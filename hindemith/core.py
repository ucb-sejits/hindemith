import ast
import inspect
import sys
import textwrap
import hindemith as hm
from hindemith.operations.core import HMOperation, DeviceLevel
from hindemith.cl import Kernel
from hindemith.types import hmarray
from hindemith.operations.array import ArrayAdd, ArraySub, ArrayMul, ArrayDiv, \
    ArrayScalarAdd, ArrayScalarSub, ArrayScalarDiv, ArrayScalarMul
import os
backend = os.getenv("HM_BACKEND", "ocl")
if backend in {"ocl", "opencl", "OCL"}:
    import pycl as cl
try:
    from graphviz import Digraph
    from profilehooks import profile
except ImportError:
    pass


def get_ast(obj):
    """
    Return the Python ast for the given object, which may
    be anything that inspect.getsource accepts (incl.
    a module, class, method, function, traceback, frame,
    or code object).
    """
    indented_program_txt = inspect.getsource(obj)
    program_txt = textwrap.dedent(indented_program_txt)
    return ast.parse(program_txt)


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.gen = set()
        self.kill = set()
        self.return_values = set()

    def visit_Call(self, node):
        for arg in node.args:
            self.visit(arg)

    def visit_Assign(self, node):
        # Need to visit loads before store
        self.visit(node.value)
        self.visit(node.targets[0])

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id not in self.kill:
                self.gen.add(node.id)
        else:
            self.kill.add(node.id)

    def visit_Return(self, node):
        if isinstance(node, ast.Tuple):
            for val in node.value.elts:
                self.return_values.add(val.id)
        else:
            self.return_values.add(node.value.id)


class Block(list):
    def __init__(self):
        super(Block, self).__init__()
        self.live_ins = set()
        self.live_outs = set()


class ComposableBlock(Block):
    pass


class NonComposableBlock(Block):
    pass


class Param(object):
    def __init__(self, node):
        self.name = node.id
        self.node = node
        self.level = 'buffer'

    def get_element(self):
        if self.level == 'register':
            return self.name
        else:
            return "{}[index]".format(self.name)


class Source(Param):
    pass


class Sink(Param):
    pass


class Compose(object):
    unique_id = -1

    def __init__(self, func, symbol_table, fusion):
        self.symbol_table = symbol_table
        self.symbol_table.update(globals())
        self.tree = get_ast(func)
        self.compiled = None
        self.fusion = fusion
        self.kernels = []

    def process_hm_ops(self, statements):
        processed = []
        for statement in statements:
            if self.is_hindemith_operation(statement):
                if len(processed) < 1 or not isinstance(processed[-1], ComposableBlock):
                    processed.append(ComposableBlock())
                processed[-1].append(statement)
            else:
                if isinstance(statement, ast.For):
                    statement.body = self.process_hm_ops(statement.body)
                if len(processed) < 1 or not isinstance(processed[-1], NonComposableBlock):
                    processed.append(NonComposableBlock())
                processed[-1].append(statement)
        for index, block in enumerate(reversed(processed)):
            analyzer = Analyzer()
            for statement in block:
                analyzer.visit(statement)
                if isinstance(statement, ast.For):
                    block.live_ins |= statement.body[0].live_ins
                    block.live_outs |= statement.body[-1].live_outs
            block.live_outs |= analyzer.return_values
            if index < len(processed):
                block.live_outs = block.live_outs.union(*(block.live_ins for block in processed[-index:]))
            block.live_ins = analyzer.gen | (block.live_outs - analyzer.kill)
        return processed

    def gen_blocks(self, body):
        processed = []
        for block in body:
            if isinstance(block, ComposableBlock):
                processed.append(self.gen_hm_func(block))
            else:
                for statement in block:
                    if isinstance(statement, ast.For):
                        statement.body = self.gen_blocks(statement.body)
                processed.extend(block)
        return processed

    def compile(self):
        tree = self.tree
        tree = UnpackBinOps().visit(tree)
        tree = ReplaceArrayOps(self.symbol_table).visit(tree)
        func_def = tree.body[0]
        new_body = self.process_hm_ops(func_def.body)
        processed = self.gen_blocks(new_body)

        func_def.body = processed
        # self.symbol_table['profile'] = profile
        # func_def.decorator_list = [ast.Name('profile', ast.Load())]
        func_def.decorator_list = []
        ast.fix_missing_locations(tree)
        exec(compile(tree, filename="<nofile>", mode="exec"),
             self.symbol_table, self.symbol_table)
        self.func_name = func_def.name
        self.compiled = self.symbol_table[func_def.name]

    def __call__(self, *args, **kwargs):
        if not self.compiled:
            self.compile()
        return self.compiled(*args, **kwargs)

    def gen_hm_func(self, block):
        sources = []
        sinks = []
        # Uncomment to show graph
        # dot = Digraph()
        # dot.body.append('size="6,6"')
        # sink_map = {}
        block_params = []
        for index, op in enumerate(block):
            _sinks, _sources = self.get_sinks_and_sources(op)
            sinks.extend(_sinks)
            sources.extend(_sources)
            block_params.append((_sinks, _sources))
        # Uncomment to show graph
        #     node_id = "node_{}".format(index)
        #     dot.node(node_id, op.value.func.id)
        #     for source in _sources:
        #         if source.name in sink_map:
        #             dot.edge(sink_map[source.name], node_id)
        #     for sink in _sinks:
        #         sink_map[sink.name] = node_id
        # dot.render('tmp.gv')

        kernels = []
        filtered_sinks = []
        if self.fusion:
            for sink in sinks:
                if sink.name not in block.live_outs:
                    sink.level = 'register'
                else:
                    filtered_sinks.append(sink)

            filtered_sources = []
            for source in sources:
                cont = False
                for sink in sinks:
                    if sink.name == source.name:
                        source.level = sink.level
                        cont = True
                        break
                for s in filtered_sources:
                    if source.name == s.name:
                        cont = True
                        break
                if cont:
                    continue
                filtered_sources.append(source)
        else:
            filtered_sinks = sinks
            filtered_sources = sources

        def fn(*args, **kwargs):
            for source, arg in zip(filtered_sources, args):
                self.symbol_table[source.name] = arg
            if len(kernels) == 0:
                for op, params in zip(block, block_params):
                    _sinks, _sources = params
                    # if len(kernels) < 1 or \
                    #    kernels[-1].launch_paramaters != launch_params:
                    #    kernels.append(Kernel(launch_params))
                    # else:
                    #     raise NotImplementedError()
                    if self.is_not_device_level(op):
                        launch_params = self.get_launch_params(
                            op, _sources, _sinks)
                        if len(kernels) == 0 or \
                                not isinstance(kernels[-1], Kernel) or \
                                kernels[-1].launch_parameters[0] != launch_params[0] \
                                or len(launch_params) > 1 and launch_params[1]:
                            kernels.append(Kernel(launch_params))
                        kernels[-1].append_body(
                            self.get_emit(op, _sources, _sinks)
                        )
                        for source in _sources:
                            if isinstance(self.symbol_table[source.name], hmarray):
                                kernels[-1].sources.add(source)
                        for sink in _sinks:
                            if isinstance(self.symbol_table[sink.name], hmarray):
                                kernels[-1].sinks.add(sink)
                    else:
                        kernels.append(self.get_launcher(op, _sources, _sinks))
                for kernel in kernels:
                    kernel.compile()
                    self.kernels.append(kernel)
            kernel_map = {}
            for kernel in kernels:
                evts = []
                for source in kernel.sources:
                    if source.name in kernel_map:
                        evts.extend(kernel_map[source.name])
                evts = kernel.launch(self.symbol_table, evts)
                for sink in kernel.sinks:
                    kernel_map[sink.name] = evts

            if backend in {"ocl", "opencl", "OCL"}:
                cl.clWaitForEvents(*evts)
            ret = tuple(self.symbol_table[sink.name] for sink in filtered_sinks)
            if len(ret) == 1:
                return ret[0]
            return ret

        self.unique_id += 1
        name = "_f{}".format(self.unique_id)
        self.symbol_table[name] = fn
        func = ast.Call(
            ast.Name(name, ast.Load()),
            [ast.Name(source.name, ast.Load()) for source in filtered_sources],
            # [],
            [],
            None,
            None,
        )
        if len(sinks) > 1:
            targets = [ast.Tuple([sink.node for sink in filtered_sinks], ast.Store())]
        else:
            targets = [sinks[0].node]
        return ast.Assign(targets, func)
        # return ast.Expr(func)

    def is_not_device_level(self, op):
        func = self.eval_in_symbol_table(op.value.func)
        return not issubclass(func, DeviceLevel)

    def get_keywords(self, operation):
        keywords = {}
        for keyword in operation.value.keywords:
            value = keyword.value
            if isinstance(value, ast.Tuple):
                keywords[keyword.arg] = [
                    self.eval_in_symbol_table(elt) for elt in value.elts]
            else:
                keywords[keyword.arg] = self.eval_in_symbol_table(value)
        return keywords

    def get_sinks_and_sources(self, operation):
        if isinstance(operation.targets[0], ast.Name):
            sources = [Sink(operation.targets[0])]
        else:
            sources = [Sink(elt) for elt in operation.targets[0].elts]
        if isinstance(operation.value, ast.Call):
            sinks = [Source(arg) for arg in operation.value.args]
        else:
            raise NotImplementedError()
        return sources, sinks

    def get_emit(self, operation, sources, sinks):
        func = self.eval_in_symbol_table(operation.value.func)
        keywords = self.get_keywords(operation)
        return func.emit(sources, sinks, keywords, self.symbol_table)

    def get_launcher(self, operation, sources, sinks):
        func = self.eval_in_symbol_table(operation.value.func)
        keywords = self.get_keywords(operation)
        return func.get_launcher(sources, sinks, keywords, self.symbol_table)

    def get_launch_params(self, operation, sources, sinks):
        func = self.eval_in_symbol_table(operation.value.func)
        sources = [self.symbol_table[src.name] for src in sources]
        sinks = [self.symbol_table[sink.name] for sink in sinks]
        return func.get_launch_parameters(sources, sinks)

    def eval_in_symbol_table(self, val):
        if isinstance(val, ast.Num):
            return val.n
        elif isinstance(val, ast.Name):
            return self.symbol_table[val.id]
        elif isinstance(val, ast.Attribute):
            return getattr(self.eval_in_symbol_table(val.value), val.attr)
        else:
            raise NotImplementedError()

    def is_hindemith_operation(self, statement):
        if not isinstance(statement, ast.Assign):
            return False
        value = statement.value
        if isinstance(value, ast.Call):
            func = self.eval_in_symbol_table(value.func)
            if not inspect.isclass(func):
                return False
            elif issubclass(func, HMOperation):
                return True
        return False

    def dump_kernels(self):
        with open("kernel.cl", "w") as f:
            for kernel in self.kernels:
                f.write(kernel.kernel_str)


def compose(fn=None, fusion=True):
    def composer(fn):
        tree = get_ast(fn)
        symbol_table = {}
        frame = inspect.stack()[1][0]
        while frame is not None:
            symbol_table.update(frame.f_locals)
            symbol_table.update(frame.f_globals)
            frame = frame.f_back
        composed = Compose(fn, symbol_table, fusion)

        def wrapped(*args, **kwargs):
            for index, arg in enumerate(tree.body[0].args.args):
                if sys.version_info < (3, 0):
                    symbol_table[arg.id] = args[index]
                else:
                    symbol_table[arg.arg] = args[index]
            return composed(*args, **kwargs)
        wrapped.composed = composed
        return wrapped
    if fn is not None:
        return composer(fn)
    return composer


class UnpackBinOps(ast.NodeTransformer):
    unique_id = -1

    def gen_tmp(self):
        self.unique_id += 1
        return "_hm_generated_{}".format(self.unique_id)

    def visit_FunctionDecl(self, node):
        new_body = []
        for statement in node.body:
            result = self.visit(statement)
            if isinstance(result, list):
                new_body.extend(result)
            else:
                new_body.append(result)
        node.body = new_body
        return node

    def visit_BinOp(self, node):
        result = []
        if not isinstance(node.right, ast.Name):
            target = self.gen_tmp()
            new_right = self.visit(node.right)
            if isinstance(new_right, list):
                result.extend(new_right)
            else:
                result.append(new_right)
            result[-1] = ast.Assign([ast.Name(target, ast.Store())],
                                    result[-1])
            node.right = ast.Name(target, ast.Load())
        if not isinstance(node.left, ast.Name):
            target = self.gen_tmp()
            new_left = self.visit(node.left)
            if isinstance(new_left, list):
                result.extend(new_left)
            else:
                result.append(new_left)
            result[-1] = ast.Assign([ast.Name(target, ast.Store())],
                                    result[-1])
            node.left = ast.Name(target, ast.Load())
        result.append(node)
        return result

    def visit_If(self, node):
        # Ignore If statements
        return node

    def visit_For(self, node):
        new_body = []
        for statement in node.body:
            result = self.visit(statement)
            if isinstance(result, list):
                new_body.extend(result)
            else:
                new_body.append(result)
        node.body = new_body
        return node

    def visit_Call(self, node):
        result = []
        new_args = []
        for arg in node.args:
            if not isinstance(arg, ast.Name):
                tmp = self.gen_tmp()
                new_arg = self.visit(arg)
                if isinstance(new_arg, list):
                    result.extend(new_arg)
                else:
                    result.append(new_arg)
                result[-1] = ast.Assign([ast.Name(tmp, ast.Store())],
                                        result[-1])
                new_args.append(ast.Name(tmp, ast.Load()))
            else:
                new_args.append(arg)
        node.args = new_args
        result.append(node)
        return result

    def visit_Assign(self, node):
        value = self.visit(node.value)
        if isinstance(value, list):
            node.value = value[-1]
            if len(value) > 1:
                value[-1] = node
                return value
        else:
            node.value = value
        return node

    def visit_Return(self, node):
        result = []
        target = self.gen_tmp()
        old_target = self.visit(node.value)
        if isinstance(old_target, list):
            result.extend(old_target)
        else:
            result.append(old_target)
        result[-1] = ast.Assign([ast.Name(target, ast.Store())], result[-1])
        node.value = ast.Name(target, ast.Load())
        result.append(node)
        return result


class ReplaceArrayOps(ast.NodeTransformer):
    array_op_map = {
        ast.Add: 'ArrayAdd',
        ast.Sub: 'ArraySub',
        ast.Div: 'ArrayDiv',
        ast.Mult: 'ArrayMul',
    }

    array_scalar_op_map = {
        ast.Add: 'ArrayScalarAdd',
        ast.Sub: 'ArrayScalarSub',
        ast.Div: 'ArrayScalarDiv',
        ast.Mult: 'ArrayScalarMul',
    }

    def __init__(self, symbol_table):
        super(ReplaceArrayOps, self).__init__()
        self.symbol_table = symbol_table

    def visit_Assign(self, node):
        node.value = self.visit(node.value)
        if not isinstance(node.targets[0], ast.Name) or node.targets[0].id in self.symbol_table:
            return node
        if isinstance(node.value, ast.BinOp) and len(node.targets) == 1:
            if isinstance(node.value, ast.Call):
                # TODO: Operations should specify an output generator
                self.symbol_table[node.targets[0].id] = hm.zeros(
                    self.symbol_table[node.value.args[0].id].shape)
        else:
            if isinstance(node.value, ast.Call) and \
                    inspect.isclass(self.symbol_table[node.value.func.id]) and \
                    issubclass(self.symbol_table[node.value.func.id], HMOperation):
                # TODO: Operations should specify an output generator
                self.symbol_table[node.targets[0].id] = hm.zeros(
                    self.symbol_table[node.value.args[0].id].shape)
        return node

    def visit_BinOp(self, node):
        if isinstance(self.symbol_table[node.left.id], hmarray):
            if isinstance(self.symbol_table[node.right.id], hmarray):
                node = ast.Call(ast.Name(self.array_op_map[node.op.__class__],
                                         ast.Load()),
                                [node.left, node.right], [], None, None)
            elif isinstance(self.symbol_table[node.right.id], (int, float)):
                node = ast.Call(ast.Name(self.array_scalar_op_map[node.op.__class__],
                                         ast.Load()),
                                [node.left, node.right], [], None, None)
        return node
