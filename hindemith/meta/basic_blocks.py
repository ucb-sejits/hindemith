import ast
from functools import reduce
import sys
from ctree.jit import LazySpecializedFunction
from .merge import merge_entry_points


def eval_in_env(env, expr):
    if isinstance(expr, ast.Name):
        return env[expr.id]
    elif isinstance(expr, ast.Attribute):
        return getattr(env[expr.value.id], expr.attr)
    raise Exception("Unhandled type for eval_in_env {}".format(type(expr)))


def str_dump(item, tab=0):
    if isinstance(item, ast.Assign):
        return "{} = {}".format(", ".join(map(str_dump, item.targets)),
                                str_dump(item.value))
    elif isinstance(item, ast.Return):
        return "return {}".format(str_dump(item.value))
    elif isinstance(item, ast.Attribute):
        return "{}.{}".format(str_dump(item.value), item.attr)
    elif isinstance(item, ast.Call):
        return "{}({})".format(str_dump(item.func),
                               ", ".join(map(str_dump, item.args)))
    elif isinstance(item, ast.Expr):
        return str_dump(item.value)
    elif isinstance(item, ast.Tuple):
        return "({})".format(", ".join(map(str_dump, item.elts)))
    elif isinstance(item, ast.Num):
        return str(item.n)
    elif isinstance(item, ast.Name):
        return item.id
    elif isinstance(item, ComposableBlock):
        tab = "\n" + "".join([" " for _ in range(tab + 2)])
        return "ComposableBlock:{}{}".format(tab, tab.join(
            map(lambda x: str_dump(x.node), item.statements)))
    elif isinstance(item, NonComposableBlock):
        tab = "\n" + "".join([" " for _ in range(tab + 2)])
        return "NonComposableBlock:{}{}".format(tab, tab.join(
            map(str_dump, item.statements)))
    elif isinstance(item, ast.arguments):
        if sys.version_info >= (3, 0):
            return ", ".join(arg.arg for arg in item.args)
        else:
            return ", ".join(arg.id for arg in item.args)
    raise Exception("Unsupport type for dumping {}: {}".format(type(item),
                                                               item))


class Statement(object):
    _fields = ['node']

    def __init__(self, node, specializer):
        self.node = node
        self.specializer = specializer
        # Infer sources and sinks for statement for now
        self.sinks = [node.targets[0].id]
        self.sources = [arg.id for arg in node.value.args]
        if isinstance(node.value.func, ast.Attribute):
            self.sources.insert(0, node.value.func.value.id)


class BasicBlock(object):
    def __init__(self, name, params, body, composable_blocks=None):
        self.name = name
        self.params = params
        self.body = body
        if composable_blocks is None:
            self.composable_blocks = ()
        else:
            self.composable_blocks = composable_blocks

    def __len__(self):
        return len(self.body)

    def __getitem__(self, item):
        return self.body[item]

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        return iter(self.body)

    def __repr__(self):
        return """
BasicBlock
  Name: {name}
  Params: {params}
  Body:
    {body}
""".format(name=self.name,
           params=str_dump(self.params),
           body="\n    ".join(map(lambda x: str_dump(x, 4), self.body)))


def get_if_composable(statement, env):
    if isinstance(statement, ast.Assign) and \
       isinstance(statement.value, ast.Call):
        if isinstance(statement.value.func, ast.Attribute):
            value = eval_in_env(env, statement.value.func.value)
            func = getattr(value, statement.value.func.attr)
        else:
            func = eval_in_env(env, statement.value.func)
        if isinstance(func, LazySpecializedFunction):
            return func
        elif hasattr(func, 'composable'):
            return func.specializer


def separate_composable_blocks(basic_block, env):
    # TODO: This is a pretty convoluted function, simplify it to a
    # reduction across the block
    statements = []
    for statement in basic_block.body:
        func = get_if_composable(statement, env)
        if func is not None:
            statement = Statement(statement, func)
            arg_vals = tuple(env[id] for id in statement.sources)
            env[statement.sinks[0]] = func.get_placeholder_output(arg_vals)
            if len(statements) > 0 and \
                    isinstance(statements[-1], ComposableBlock):
                statements[-1].add_statement(statement)
            else:
                statements.append(ComposableBlock([statement]))
        else:
            if len(statements) > 0 and \
                    isinstance(statements[-1], NonComposableBlock):
                statements[-1].add_statement(statement)
            else:
                statements.append(NonComposableBlock([statement]))

    return BasicBlock(basic_block.name, basic_block.params,
                      statements)


class SubBlock(object):
    def __init__(self, statements):
        super(SubBlock, self).__init__()
        self.statements = statements
        self.live_ins = set()
        self.live_outs = set()

    def add_statement(self, item):
        self.statements.append(item)

    def __iter__(self):
        return iter(self.statements)

    def __getitem__(self, item):
        return self.statements[item]


class ComposableBlock(SubBlock):
    """docstring for ComposableBlock"""
    pass


class NonComposableBlock(SubBlock):
    """docstring for NonComposableBlock"""
    pass


def gen_tmp():
    gen_tmp.tmp += 1
    return "_t{}".format(gen_tmp.tmp)

gen_tmp.tmp = -1


def decompose(expr):
    def visit(expr, curr_target=None):
        if isinstance(expr, ast.Return):
            if isinstance(expr.value, (ast.Name, ast.Tuple)):
                body = (expr, )
            else:
                tmp = gen_tmp()
                body = visit(expr.value, ast.Name(tmp, ast.Store()))
                body += (ast.Return(ast.Name(tmp, ast.Load())), )
        elif isinstance(expr, ast.Name):
            return expr
        elif isinstance(expr, ast.BinOp):
            body = ()
            operands = []

            if isinstance(expr.left, ast.Num):
                body += (ast.Assign([curr_target], expr), )
            else:
                for operand in [expr.left, expr.right]:
                    if isinstance(operand, (ast.Name, ast.Num)):
                        operands += (operand, )
                    else:
                        tmp = gen_tmp()
                        body += visit(operand,
                                      ast.Name(tmp, ast.Store()))
                        operands.append(ast.Name(tmp, ast.Load()))
                if isinstance(expr.op, ast.Add):
                    op = ast.Attribute(operands[0], '__add__', ast.Load())
                elif isinstance(expr.op, ast.Mult):
                    op = ast.Attribute(operands[0], '__mul__', ast.Load())
                elif isinstance(expr.op, ast.Sub):
                    op = ast.Attribute(operands[0], '__sub__', ast.Load())
                elif isinstance(expr.op, ast.Div):
                    op = ast.Attribute(operands[0], '__div__', ast.Load())
                else:
                    raise Exception("Unsupported BinOp {}".format(expr.op))
                operands.pop(0)
                body += (ast.Assign([curr_target],
                                    ast.Call(op, operands, [], None, None)), )
        elif isinstance(expr, ast.Assign):
            target = expr.targets[0]
            if isinstance(target, ast.Tuple):
                body = reduce(lambda x, y: x + y,
                              map(visit, expr.value.elts, target.elts), ())
            else:
                body = visit(expr.value, target)
        elif isinstance(expr, ast.Call):
            body = ()
            args = []
            for arg in expr.args:
                val = visit(arg)
                if isinstance(val, tuple):
                    tmp = gen_tmp()
                    val = visit(arg, ast.Name(tmp, ast.Store))
                    body += val
                    args.append(ast.Name(tmp, ast.Load()))
                elif isinstance(val, (ast.Name, ast.Num)):
                    args.append(val)
                else:
                    raise Exception("Call argument returned\
                                     unsupported type {}".format(type(val)))
            if curr_target is not None:
                body += (ast.Assign(
                    [curr_target],
                    ast.Call(visit(expr.func), args, [], None, None)
                ), )
            else:
                body += (ast.Call(visit(expr.func), args, [], None, None), )
        elif isinstance(expr, ast.Expr):
            return (ast.Expr(visit(expr.value)[0]), )
        else:
            raise Exception("Unsupported expression {}".format(expr))
        return body

    return visit(expr)


def get_basic_block(module):
    func = module.body[0]
    params = func.args
    body = map(decompose, func.body)
    body = reduce(lambda x, y: x + y, body, ())
    return BasicBlock(func.name, params, body)


def process_composable_blocks(basic_block, env):
    body = []
    for sub_block in basic_block:
        if isinstance(sub_block, ComposableBlock):
            body.append(merge_entry_points(sub_block, env))
        else:
            body.extend(sub_block.statements)
    return BasicBlock(basic_block.name, basic_block.params, body)
