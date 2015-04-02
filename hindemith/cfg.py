import ast
from hindemith.operations.core import operations
from hindemith.cl import context, queue
import ctree.c.nodes as C
from ctree.types import get_c_type_from_numpy_dtype
import numpy as np
import ctypes as ct
import pycl as cl


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.gen = set()
        self.kill = set()
        self.return_values = set()

    def visit_Call(self, node):
        for arg in node.args:
            self.visit(arg)

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


def perform_liveness_analysis(basic_blocks):
    for index, block in enumerate(reversed(basic_blocks)):
        analyzer = Analyzer()
        for statement in block:
            if isinstance(statement, ast.AST):
                analyzer.visit(statement)
            else:
                analyzer.visit(statement.statement)
        if index == 0:
            block.live_outs = set()
        else:
            block.live_outs = set().union(
                *(b.live_ins for b in basic_blocks[-index:]))
        block.live_outs |= analyzer.return_values
        block.live_ins = analyzer.gen.union(
            block.live_outs.difference(analyzer.kill))


class CFGBuilder(ast.NodeTransformer):
    def __init__(self):
        self.funcs = []
        self.tmp = -1
        self.curr_target = None

    def _gen_tmp(self):
        self.tmp += 1
        return "_t{}".format(self.tmp)

    def visit_FunctionDef(self, node):
        new_body = []
        for statement in node.body:
            result = self.visit(statement)
            if isinstance(result, list):
                new_body.extend(result)
            else:
                new_body.append(result)
        node.body = [BasicBlock(new_body)]
        return node

    def visit_Call(self, node):
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Name):
                args.append(arg)
            else:
                raise NotImplementedError()
        node.args = args
        return [ast.Assign([ast.Name(self.curr_target, ast.Store())], node)]

    def visit_BinOp(self, node):
        operands = ()
        ret = []
        for operand in (node.right, node.left):
            if isinstance(operand, ast.Name):
                operands += (operand, )
            else:
                old_target = self.curr_target
                self.curr_target = self._gen_tmp()
                ret.extend(self.visit(operand))
                operands += (ast.Name(self.curr_target, ast.Load()), )
                self.curr_target = old_target
        node.right = operands[0]
        node.left = operands[1]
        ret.append(ast.Assign([ast.Name(self.curr_target, ast.Store())], node))
        return ret

    def visit_Assign(self, node):
        old_target = self.curr_target
        self.curr_target = node.targets[0].id
        ret = self.visit(node.value)
        self.curr_target = old_target
        return ret

    def visit_Return(self, node):
        if isinstance(node.value, ast.Name):
            return node
        elif isinstance(node.value, ast.Tuple):
            raise NotImplementedError()
        tmp = self._gen_tmp()
        self.curr_target = tmp
        value = self.visit(node.value)
        node.value = ast.Name(tmp, ast.Load())
        return value + [node]


op2str = {
    ast.Add: "+",
    ast.Mult: "*",
    ast.Sub: "-",
    ast.Div: "/"
}


def dump_op(op):
    if isinstance(op, ast.Call):
        return "{}({})".format(op.func.id, ", ".join([arg.id for arg in
                                                      op.args]))
    elif isinstance(op, ast.BinOp):
        return "{} {} {}".format(op.left.id, op2str[op.op.__class__],
                                 op.right.id)
    else:
        raise NotImplementedError(op)


class BasicBlock(object):

    """Docstring for BasicBlock. """

    def __init__(self, statements):
        """TODO: to be defined1. """
        self.statements = statements
        self.live_ins = set()
        self.live_outs = set()

    def add_statement(self, statement):
        self.statements.append(statement)

    def __getitem__(self, index):
        return self.statements[index]

    def __len__(self):
        return len(self.statements)

    def dump(self, tab):
        output = tab + self.__class__.__name__ + "\n"
        tab += "  "
        output += tab + "live ins: {}\n".format(", ".join(self.live_ins))
        output += tab + "live outs: {}\n".format(", ".join(self.live_outs))
        output += tab + "body:\n"
        tab += "  "
        for expr in self.statements:
            if isinstance(expr, ast.Assign):
                if isinstance(expr.targets[0], ast.Tuple):
                    output += tab + "{} = {}\n".format(
                        ", ".join(target.id for target in
                                  expr.targets[0].elts),
                        dump_op(expr.value))
                else:
                    output += tab + "{} = {}\n".format(
                        expr.targets[0].id, dump_op(expr.value))
            elif isinstance(expr, ast.Return):
                output += tab + "return {}\n".format(expr.value.id)
        return output


class ComposableBasicBlock(BasicBlock):
    def find_matching_op(self, statement, env):
        for op in operations:
            if op.match(statement, env):
                return op
        raise Exception("Found non unsupported statement in composable block")

    def compile(self, name, env):
        body = []
        global_size = None
        for statement in self.statements:
            for elem in statement.sources + statement.sinks:
                if elem not in self.live_ins.union(self.live_outs):
                    decl = env[elem].promote_to_register(elem)
                    if decl is not None:
                        body.insert(0, decl)
            global_size = statement.get_global_size()
            body.append(statement.compile())
        params = []
        for arg in self.live_ins:
            ptr = ct.POINTER(get_c_type_from_numpy_dtype(
                np.dtype(env[arg].dtype)))()
            params.append(C.SymbolRef(arg, ptr, _global=True))
        for arg in self.live_outs:
            ptr = ct.POINTER(get_c_type_from_numpy_dtype(
                np.dtype(env[arg].dtype)))()
            params.append(C.SymbolRef(arg, ptr, _global=True))
        kernel = C.FunctionDecl(
            None,
            C.SymbolRef(name),
            params,
            body
        )
        kernel.set_kernel()
        print(kernel)

        def compiled(*args, **kwargs):
            types = []
            bufs = []
            outs = []
            for arg in args:
                types.append(cl.cl_mem)
                bufs.append(arg.ocl_buf)
            for arg in self.live_outs:
                types.append(cl.cl_mem)
                outs.append(env[arg].ocl_buf)
                env[arg].host_dirty = True
            program = cl.clCreateProgramWithSource(
                context, kernel.codegen()
            ).build()

            kern = program[kernel.name.name]
            kern.argtypes = types
            evt = kern(*(bufs + outs)).on(queue, global_size)
            rets = ()
            for arg in self.live_outs:
                rets += (env[arg], )
            if len(rets) == 1:
                return rets[0]
            return rets
        return compiled


class NonComposableBasicBlock(BasicBlock):
    pass


class ControlFlowGraph(object):
    """
    A datastructure to represent the control flow graph of a function as a
    graph of basic blocks.
    """
    def __init__(self, func):
        """
        :param ast.FunctionDef func:
        """
        self.name = func.name
        self.params = func.args
        self.graph = CFGBuilder().visit(func)
        self.func_id = -1

    def __str__(self):
        output = ""
        tab = ""
        for block in self.graph.body:
            if isinstance(block, BasicBlock):
                output += block.dump(tab)
            else:
                raise NotImplementedError(block)
        return output

    def compile_to_fn(self, env):
        # TODO: Should we create a new module/funcdef or just reuse the one
        # passed in
        tree = ast.Module(
            [ast.FunctionDef(self.name, self.params,
                             self.graph.body[0].statements, [])]
        )
        ast.fix_missing_locations(tree)
        exec(compile(tree, filename="<nofile>", mode="exec"), env, env)
        return env[self.name]

    def find_matching_op(self, ops, statement, env):
        for op in operations:
            if op.match(statement, env):
                return op
        return None

    def build_composable_blocks(self, env):
        blocks = []
        for block in self.graph.body:
            for statement in block:
                op = self.find_matching_op(operations, statement, env)
                if op is not None:
                    if len(blocks) < 1 or \
                            not isinstance(blocks[-1],
                                           ComposableBasicBlock):
                        blocks.append(ComposableBasicBlock([]))
                    statement = op(statement, env)
                else:
                    if len(blocks) < 1 or \
                            not isinstance(blocks[-1],
                                           NonComposableBasicBlock):
                        blocks.append(NonComposableBasicBlock([]))
                blocks[-1].add_statement(statement)
        self.graph.body = blocks

        perform_liveness_analysis(self.graph.body)

    def gen_func_name(self):
        self.func_id += 1
        return "_f{}".format(self.func_id)

    def compile_composable_blocks(self, env):
        blocks = []
        for block in self.graph.body:
            if isinstance(block, NonComposableBasicBlock):
                blocks.extend(block.statements)
            else:
                func_name = self.gen_func_name()
                env[func_name] = block.compile(func_name, env)
                if len(block.live_outs) > 1:
                    target = [ast.Tuple(
                        [ast.Name(sink, ast.Store()) for sink in
                         block.live_outs], ast.Store()
                    )]
                else:
                    target = [ast.Name(next(iter(block.live_outs)),
                                       ast.Store())]
                blocks.append(ast.Assign(
                    target,
                    ast.Call(ast.Name(func_name, ast.Load()),
                             [ast.Name(source, ast.Load()) for source in
                              block.live_ins],
                             [], None, None)
                ))
                ast.fix_missing_locations(blocks[-1])
        self.graph.body = [BasicBlock(blocks)]
