import ast
import inspect
import sys
import textwrap
from hindemith.operations.core import HMOperation, DeviceLevel
from hindemith.cl import Kernel, queue
import pycl as cl


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


class Block(list):
    pass


class Compose(object):
    unique_id = -1

    def __init__(self, func, symbol_table):
        self.symbol_table = symbol_table
        self.tree = get_ast(func)
        self.compiled = None

    def compile(self):
        tree = self.tree
        func_def = tree.body[0]
        new_body = []
        for statement in func_def.body:
            if self.is_hindemith_operation(statement):
                if len(new_body) < 1 or not isinstance(new_body[-1], Block):
                    new_body.append(Block())
                new_body[-1].append(statement)
            else:
                new_body.append(statement)
        processed = []

        for block_or_statement in new_body:
            if isinstance(block_or_statement, Block):
                processed.append(self.gen_hm_func(block_or_statement))
            else:
                processed.append(block_or_statement)
        func_def.body = processed
        func_def.decorator_list = []
        ast.fix_missing_locations(tree)
        exec(compile(tree, filename="<nofile>", mode="exec"),
             self.symbol_table, self.symbol_table)
        self.func_name = func_def.name
        self.compiled = self.symbol_table[func_def.name]

    def __call__(self, *args, **kwargs):
        if not self.compiled:
            self.compile()
        self.compiled(*args, **kwargs)

    def gen_hm_func(self, block):
        sources = []
        sinks = []
        for op in block:
            _sinks, _sources = self.get_sinks_and_sources(op)
            sinks.extend(_sinks)
            sources.extend(_sources)

        kernels = []

        def fn(*args, **kwargs):
            for source, arg in zip(sources, args):
                self.symbol_table[source.id] = arg
            if len(kernels) == 0:
                for op in block:
                    _sinks, _sources = self.get_sinks_and_sources(op)
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
                        kernels[-1].sources |= set(_sources)
                        kernels[-1].sinks |= set(_sinks)
                    else:
                        kernels.append(self.get_launcher(op, _sources, _sinks))
                for kernel in kernels:
                    kernel.compile()
            for kernel in kernels:
                kernel.launch(self.symbol_table)
                # cl.clFinish(queue)
            # ret = tuple(self.symbol_table[sink.id] for sink in sinks)
            # if len(ret) == 1:
            #     return ret[0]
            # return ret

        self.unique_id += 1
        name = "_f{}".format(self.unique_id)
        self.symbol_table[name] = fn
        func = ast.Call(
            ast.Name(name, ast.Load()),
            sources,
            [],
            None,
            None,
        )
        if len(sinks) > 1:
            targets = [ast.Tuple(sinks, ast.Store())]
        else:
            targets = [sinks[0]]
        # return ast.Assign(targets, func)
        return ast.Expr(func)

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
            sources = [operation.targets[0]]
        else:
            sources = [elt for elt in operation.targets[0].elts]
        if isinstance(operation.value, ast.Call):
            sinks = [arg for arg in operation.value.args]
        else:
            raise NotImplementedError()
        return sources, sinks

    def get_emit(self, operation, sources, sinks):
        func = self.eval_in_symbol_table(operation.value.func)
        sources = [src.id for src in sources]
        sinks = [sink.id for sink in sinks]
        keywords = self.get_keywords(operation)
        return func.emit(sources, sinks, keywords, self.symbol_table)

    def get_launcher(self, operation, sources, sinks):
        func = self.eval_in_symbol_table(operation.value.func)
        sources = [src.id for src in sources]
        sinks = [sink.id for sink in sinks]
        keywords = self.get_keywords(operation)
        return func.get_launcher(sources, sinks, keywords, self.symbol_table)

    def get_launch_params(self, operation, sources, sinks):
        func = self.eval_in_symbol_table(operation.value.func)
        sources = [self.eval_in_symbol_table(src) for src in sources]
        sinks = [self.eval_in_symbol_table(sink) for sink in sinks]
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


def compose(fn):
    tree = get_ast(fn)
    symbol_table = {}
    frame = inspect.stack()[1][0]
    while frame is not None:
        symbol_table.update(frame.f_locals)
        symbol_table.update(frame.f_globals)
        frame = frame.f_back
    composed = Compose(fn, symbol_table)

    def wrapped(*args, **kwargs):
        for index, arg in enumerate(tree.body[0].args.args):
            if sys.version_info < (3, 0):
                symbol_table[arg.id] = args[index]
            else:
                symbol_table[arg.arg] = args[index]
        return composed(*args, **kwargs)
    return wrapped
