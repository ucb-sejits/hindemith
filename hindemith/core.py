from ctree import browser_show_ast
from ctree.frontend import get_ast
from ctree.jit import LazySpecializedFunction
from hindemith.operations.dense_linear_algebra import Float32, Int, Scalar, Array
from hindemith.types.stencil import Stencil
from numpy import ndarray
from hindemith.utils import UnsupportedTypeError, unique_kernel_name, unique_python_name
import ast
import logging

LOG = logging.getLogger('Hindemith')

__author__ = 'leonardtruong'


def coercer(arg):
    name, value = arg
    if isinstance(value, float):
        return name, Float32(name, value)
    elif isinstance(value, int):
        return name, Int(name, value)
    elif isinstance(value, Stencil):
        value.name = name
        return name, value
    elif isinstance(value, Array):
        value.name = name
        return name, value
    elif isinstance(value, Scalar):
        value.name = name
        return name, value
    elif isinstance(value, ndarray):
        return name, Array(name, value)
    else:
        raise UnsupportedTypeError("Hindemith found unsupported type: {0}".format(type(value)))


def fuse(fn):
    def fused_fn(*args, **kwargs):
        symbol_table = {}
        for name, value in map(coercer, kwargs.items()):
            symbol_table[name] = value
        tree = get_ast(fn)
        blocks = tree.body[0].body

        MagicMethodProcessor().visit(tree)
        decls = []
        BlockBuilder(symbol_table, decls).visit(tree)
        # init = [get_specializer(blocks[0], symbol_table)]
        # fused_blocks = reduce(fuse_blocks_wrapper(symbol_table), blocks[1:], init)
        decls.extend(tree.body[0].body)
        tree.body[0].body = decls
        tree = ast.fix_missing_locations(tree)
        browser_show_ast(tree, 'tmp.png')
        exec(compile(tree, filename='', mode='exec')) in globals(), locals()
        return fn(**symbol_table)
    return fused_fn


def fuse_blocks_wrapper(symbol_table):
    def fuse_blocks(result, next):
        prev = result[-1]
        next = get_specializer(next, symbol_table)
        if do_fusion(prev, next):
            return result
        else:
            return result + [next]
    return fuse_blocks


def fusable(prev, next):
    return True


def do_fusion(prev, next):
    if True:  # TODO: Fusability check
        return True
    else:
        return False


class BlockBuilder(ast.NodeTransformer):
    def __init__(self, symbol_table, decls):
        self.symbol_table = symbol_table
        self.decls = decls
        self.result = False
        self.prev = None

    def get_if_specializer(self, name, attr):
        func = self.symbol_table[name]
        if attr is not None:
            func = getattr(func, attr)
        if isinstance(func, LazySpecializedFunction):
            self.result = func
        else:
            self.result = None

    def get_specializer(self, node):
        if isinstance(node.func, ast.Attribute):
            arg = getattr(self.symbol_table[node.func.value.id], node.func.attr)
            name = ast.Str(node.func.value.id)
            attr = ast.Str(node.func.attr)
        else:
            name = ast.Str(node.func.id)
            attr = None
        expr = ast.Expression(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name('self', ast.Load()),
                    attr='get_if_specializer',
                    ctx=ast.Load()
                ),
                args=[name, attr],
                keywords=[]
            )
        )
        ast.fix_missing_locations(expr)
        exec(compile(expr, filename='', mode='eval')) in globals(), locals()
        return self.result

    def attempt_fusion(self, previous, next_tree):
        prev = self.get_specializer(previous.value)
        next = self.get_specializer(next_tree.value)
        if not prev or not next:
            return

        fused_name = unique_python_name()
        fused = ast.Call(
            func=ast.Name(fused_name, ast.Load()),
            args=previous.value.args + [next_tree.value.func.value],
            keywords=[]
        )
        previous.value = ast.copy_location(fused, previous.value)
        previous.targets = next_tree.targets
        return True

    def visit_FunctionDef(self, node):
        body = []
        for child in node.body:
            result = self.visit(child)
            if isinstance(result, ast.Assign):
                if isinstance(result.value, ast.Call):
                    if self.prev:
                        if self.attempt_fusion(self.prev, child):
                            self.prev = child
                            continue
                    self.prev = child
            else:
                self.prev = None
            body.append(child)
        node.body = body
        return node


    # def visit_Call(self, node):
    #     if isinstance(node.func, ast.Attribute):
    #         arg = getattr(self.symbol_table[node.func.value.id], node.func.attr)
    #         name = ast.Str(node.func.value.id)
    #         attr = ast.Str(node.func.attr)
    #     else:
    #         name = ast.Str(node.func.id)
    #         attr = None
    #     expr = ast.Expression(
    #         ast.Call(
    #             func=ast.Attribute(
    #                 value=ast.Name('self', ast.Load()),
    #                 attr='is_specializer',
    #                 ctx=ast.Load()
    #             ),
    #             args=[name, attr],
    #             keywords=[]
    #         )
    #     )
    #     ast.fix_missing_locations(expr)
    #     exec(compile(expr, filename='', mode='eval')) in globals(), locals()
    #
    #     if self.result:
    #         new_func = unique_kernel_name()
    #         self.decls.append(ast.FunctionDef(
    #             name=new_func,
    #             args=ast.arguments(
    #                 args=[],
    #                 defaults=[]
    #             ),
    #             body=[ast.Expr(node)],
    #             decorator_list=[]
    #         ))
    #         return ast.copy_location(ast.Call(
    #             func=ast.Name(new_func, ast.Load()),
    #             args=[],
    #             keywords=[]
    #         ), node)
    #     return node


class MagicMethodProcessor(ast.NodeTransformer):
    def __init__(self):
        self.result = False

    def visit_BinOp(self, node):
        attr_map = {
            ast.Mult: '__mul__',
            ast.Div: '__div__',
            ast.Sub: '__sub__',
            ast.Add: '__add__'
        }
        attr = attr_map[type(node.op)]
        expr = ast.Call(
            func=ast.Attribute(
                value=node.left,
                attr=attr,
                ctx=ast.Load()
            ),
            args=[node.right],
            keywords=[]
        )

        expr = ast.fix_missing_locations(expr)
        return ast.copy_location(expr, node)
