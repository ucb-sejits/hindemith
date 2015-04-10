import ast
from hindemith.operations.core import operations, DeviceLevel, ElementLevel
from hindemith.cl import ElementLevelKernel, queue
import sys
import pycl as cl


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


def perform_liveness_analysis(basic_block, live_outs=None):
    if basic_block.next_block is not None:
        perform_liveness_analysis(basic_block.next_block)
        curr_block = basic_block.next_block
        while curr_block is not None:
            basic_block.live_outs = basic_block.live_outs | curr_block.live_ins
            curr_block = curr_block.next_block
    elif live_outs is not None:
        basic_block.live_outs = live_outs
    else:
        basic_block.live_outs = set()
    if isinstance(basic_block, LoopBlock):
        perform_liveness_analysis(basic_block.next_block)
        perform_liveness_analysis(basic_block.start_block,
                                  basic_block.live_outs)
        basic_block.live_ins = basic_block.start_block.live_ins
        end_block = basic_block.start_block
        while end_block.next_block is not None:
            end_block = end_block.next_block
        basic_block.live_outs = end_block.live_outs
        return
    analyzer = Analyzer()
    for statement in basic_block:
        if isinstance(statement, ast.AST):
            analyzer.visit(statement)
        else:
            analyzer.gen |= set(statement.sources) - analyzer.kill
            analyzer.kill |= set(statement.sinks)
    basic_block.live_outs |= analyzer.return_values
    basic_block.live_ins = analyzer.gen | (basic_block.live_outs -
                                           analyzer.kill)


class LoopBlock(object):
    def __init__(self, ast_node, start_block):
        self.ast_node = ast_node
        self.start_block = start_block
        self.next_block = None
        self.live_ins = set()
        self.live_outs = set()

    def dump(self, tab):
        output = tab + "LoopBlock\n"
        output += self.start_block.dump(tab + "  ")
        if self.next_block is not None:
            output += self.next_block.dump(tab)
        return output


class CFGBuilder(ast.NodeVisitor):
    def __init__(self):
        self.funcs = []
        self.tmp = -1
        self.curr_target = None
        self.curr_basic_block = None

    def _gen_tmp(self):
        self.tmp += 1
        return "_t{}".format(self.tmp)

    def visit_FunctionDef(self, node):
        self.start_block = BasicBlock([])
        self.curr_basic_block = self.start_block
        for statement in node.body:
            self.visit(statement)

    def visit_For(self, node):
        start_block = BasicBlock([])
        loop = LoopBlock(node, start_block)
        self.curr_basic_block.next_block = loop
        self.curr_basic_block = start_block
        for statement in node.body:
            self.visit(statement)
        after_block = BasicBlock([])
        loop.next_block = after_block
        self.curr_basic_block = after_block

    def visit_Call(self, node):
        if node.func.id in {'range', 'print'}:
            return node
        self.curr_basic_block.add_statement(
            ast.Assign([ast.Name(self.curr_target, ast.Store())], node))

    def visit_BinOp(self, node):
        operands = ()
        for operand in (node.right, node.left):
            if isinstance(operand, (ast.Name, ast.Num)):
                operands += (operand, )
            else:
                old_target = self.curr_target
                self.curr_target = self._gen_tmp()
                self.visit(operand)
                operands += (ast.Name(self.curr_target, ast.Load()), )
                self.curr_target = old_target
        node.right = operands[0]
        node.left = operands[1]
        self.curr_basic_block.add_statement(
            ast.Assign([ast.Name(self.curr_target, ast.Store())], node))

    def visit_Assign(self, node):
        old_target = self.curr_target
        self.curr_target = node.targets[0].id
        self.visit(node.value)
        self.curr_target = old_target
        # self.curr_basic_block.add_statement(ret)

    def visit_Return(self, node):
        if not isinstance(node.value, ast.Name):
            tmp = self._gen_tmp()
            self.curr_target = tmp
            self.visit(node.value)
            node.value = ast.Name(tmp, ast.Load())
        self.curr_basic_block.add_statement(node)


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
        if isinstance(op.left, ast.Name):
            left = op.left.id
        else:
            left = op.left.n
        if isinstance(op.right, ast.Name):
            right = op.right.id
        else:
            right = op.right.n
        return "{} {} {}".format(left, op2str[op.op.__class__], right)
    else:
        raise NotImplementedError(op)


class BasicBlock(object):

    """Docstring for BasicBlock. """

    def __init__(self, statements):
        """TODO: to be defined1. """
        self.statements = statements
        self.live_ins = set()
        self.live_outs = set()
        self.next_block = None

    def add_statement(self, statement):
        self.statements.append(statement)

    def __getitem__(self, index):
        return self.statements[index]

    def __len__(self):
        return len(self.statements)

    def dump(self, tab):
        orig_tab = tab
        output = tab + self.__class__.__name__ + "\n"
        tab += "  "
        output += tab + "live ins: {}\n".format(", ".join(self.live_ins))
        output += tab + "live outs: {}\n".format(", ".join(self.live_outs))
        output += tab + "body:\n"
        tab += "  "
        for expr in self.statements:
            if not isinstance(expr, ast.AST):
                expr = expr.statement
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
        if self.next_block is not None:
            output += self.next_block.dump(orig_tab)
        return output


class ComposableBasicBlock(BasicBlock):
    def find_matching_op(self, statement, env):
        for op in operations:
            if op.match(statement, env):
                return op
        raise Exception("Found non unsupported statement in composable block")

    def compile(self, name, env):
        # body = []
        kernels = []
        # global_size = None
        for statement in self.statements:
            if isinstance(statement, DeviceLevel):
                kernels.extend(statement.compile())
            elif isinstance(statement, ElementLevel):
                body, global_size, sources, sinks = statement.compile()
                if len(kernels) < 1 or \
                   not isinstance(kernels[-1], ElementLevelKernel) or \
                   global_size != kernels[-1].global_size:
                    kernels.append(ElementLevelKernel(global_size))
                kernels[-1].body += "\n" + body
                kernels[-1].sources |= set(sources)
                kernels[-1].sinks |= set(sinks)
            # for elem in statement.sources + statement.sinks:
            #     if elem not in (self.live_ins | self.live_outs):
            #         decl = env[elem].promote_to_register(elem)
            #         if decl is not None:
            #             body.insert(0, decl)
            # body.append(statement.compile())
            # global_size = statement.get_global_size()
        # param_set = self.live_outs | self.live_ins
        # params = []
        # for arg in param_set:
        #     ptr = ct.POINTER(get_c_type_from_numpy_dtype(
        #         np.dtype(env[arg].dtype)))()
        #     params.append(C.SymbolRef(arg, ptr, _global=True))
        # kernel = C.FunctionDecl(
        #     None,
        #     C.SymbolRef(name),
        #     params,
        #     body
        # )
        # kernel.set_kernel()
        # print(kernel)

        def compiled(*args, **kwargs):
            # types = []
            # bufs = []
            # outs = []
            # for arg in self.live_ins:
            #     types.append(cl.cl_mem)
            #     bufs.append(arg.ocl_buf)
            # for arg in self.live_outs | self.live_ins:
            #     types.append(cl.cl_mem)
            #     outs.append(env[arg].ocl_buf)
            #     env[arg].host_dirty = True
            # program = cl.clCreateProgramWithSource(
            #     context, kernel.codegen()
            # ).build()

            # kern = program[kernel.name.name]
            # kern.argtypes = types
            # kern(*(bufs + outs)).on(queue, global_size)
            for kernel in kernels:
                kernel.launch(env)
                cl.clFinish(queue)
            rets = tuple(env[out] for out in self.live_outs)
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
        self.compiled = None
        builder = CFGBuilder()
        builder.visit(func)
        self.start_block = builder.start_block
        self.func_id = -1

    def __str__(self):
        return self.start_block.dump("")

    def compile_blocks(self, block):
        body = []
        while block is not None:
            if isinstance(block, LoopBlock):
                body.append(block.ast_node)
                block.ast_node.body = self.compile_blocks(block.start_block)
            else:
                body.extend(block.statements)
            block = block.next_block
        return body

    def compile_to_fn(self, env):
        # TODO: Should we create a new module/funcdef or just reuse the one
        # passed in
        if sys.version_info > (3, 0):
            tree = ast.Module(
                [ast.FunctionDef(self.name, self.params,
                                 self.compiled, [], None)]
            )
        else:
            tree = ast.Module(
                [ast.FunctionDef(self.name, self.params,
                                 self.compiled, [])]
            )
        ast.fix_missing_locations(tree)
        exec(compile(tree, filename="<nofile>", mode="exec"), env, env)
        return env[self.name]

    def find_matching_op(self, ops, statement, env):
        for op in operations:
            if op.match(statement, env):
                return op
        return None

    def build_composable_blocks(self, env, block):
        if block is None or not isinstance(block, LoopBlock) and \
           len(block.statements) == 0:
            return
        elif isinstance(block, LoopBlock):
            block.start_block = self.build_composable_blocks(
                env, block.start_block)
            new_block = block
            start_block = new_block
        else:
            new_block = None
            start_block = None
            for statement in block:
                op = self.find_matching_op(operations, statement, env)
                if op is not None:
                    if new_block is None:
                        new_block = ComposableBasicBlock([])
                        start_block = new_block
                    elif not isinstance(new_block, ComposableBasicBlock):
                        new_block.next_block = ComposableBasicBlock([])
                        new_block = new_block.next_block
                    statement = op(statement, env)
                else:
                    if new_block is None:
                        new_block = NonComposableBasicBlock([])
                        start_block = new_block
                    elif not isinstance(new_block, NonComposableBasicBlock):
                        new_block.next_block = NonComposableBasicBlock([])
                        new_block = new_block.next_block
                new_block.add_statement(statement)
        new_block.next_block = self.build_composable_blocks(
            env, block.next_block)
        return start_block

    def perform_liveness_analysis(self):
        perform_liveness_analysis(self.start_block)

    def gen_func_name(self):
        self.func_id += 1
        return "_f{}".format(self.func_id)

    def compile_composable_blocks(self, env, block):
        statements = []
        while block is not None:
            if isinstance(block, NonComposableBasicBlock):
                statements.extend(block.statements)
            elif isinstance(block, LoopBlock):
                statements.append(block.ast_node)
                block.ast_node.body = self.compile_composable_blocks(
                    env, block.start_block)
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
                statements.append(ast.Assign(
                    target,
                    ast.Call(ast.Name(func_name, ast.Load()),
                             [ast.Name(source, ast.Load()) for source in
                              block.live_ins],
                             [], None, None)
                ))
            block = block.next_block
        return statements
