import numpy
import ast
import pycl as cl

from ctree.frontend import get_ast
from ctree.jit import ConcreteSpecializedFunction
import ctree
import ctree.np

from hindemith.utils import unique_kernel_name, unique_name

import inspect

ctree.np  # Make PEP happy

import logging
LOG = logging.getLogger('Hindemith')


def fuse(fn):
    """Decorator that will fuse specializers in the body of a function.  This
    optimizer will attempt to fuse specializer calls on various levels to
    improve runtime performance.  It will execute regular python code normally,
    and will output the same result as running the non-fused version of the
    function.

    :fn: A python function.
    :returns: `fused`, higher order python function that takes in the same
    paramaters as `fn` and returns the same result(s) as `fn`.

    """
    def fused(*args, **kwargs):
        """Fused wrapper around `fn`.  First it get all variables defined in the
        local, global scope.  It then traverses the body of the function
        looking for places where specializer calls are made.  Any specializer
        calls found that can be fused will be (@todo: Tuning should occur here
        instead).

        :*args: Arguments to `fn`
        :**kwargs: Keyword arguments to `fn`.
        :returns: The same return value(s) as `fn`.

        """
        tree = get_ast(fn)
        blocks = get_blocks(tree)

        # Build a symbol table with keyword arguments and the function's global
        # scope.
        symbol_table = dict(fn.__globals__, **kwargs)
        # Add all locally defined variables in the current stack
        for frame in inspect.stack()[1:]:
            symbol_table.update(frame[0].f_locals)

        # Add non keyword arguments
        for index, arg in enumerate(args):
            symbol_table[tree.body[0].args.args[index].id] = arg

        fuser = Fuser(blocks, symbol_table)
        fused_blocks = fuser.do_fusion()
        tree.body[0].body = fused_blocks

        # Remove Decorator
        tree.body[0].decorator_list = []
        tree = ast.fix_missing_locations(tree)
        exec(compile(tree, '', 'exec')) in symbol_table
        return symbol_table[fn.__name__](*args, **kwargs)
    return fused


def get_blocks(tree):
    """Convenience method for getting the blocks from an ast

    :param ast.Node tree: A Python AST.
    :returns: A list of AST nodes.
    :rtype: list
    """
    blocks = []
    BlockBuilder(blocks).visit(tree)
    return blocks


class BlockBuilder(ast.NodeVisitor):

    """Docstring for BlockBuilder. """

    def __init__(self, blocks):
        """@todo: to be defined1.

        :blocks: @todo

        """
        ast.NodeVisitor.__init__(self)

        self._blocks = blocks

    def visit_FunctionDef(self, node):
        """@todo: Docstring for visit_FunctionDef.

        :node: @todo
        :returns: @todo

        """
        self._blocks.extend(node.body)


class UniqueNamer(ast.NodeTransformer):
    uid = 0

    def __init__(self):
        ast.NodeTransformer.__init__(self)
        self.seen = {}

    def visit_FunctionCall(self, node):
        # Don't rename functions
        return node

    def visit_SymbolRef(self, node):
        if node.name not in self.seen:
            UniqueNamer.uid += 1
            self.seen[node.name] = '_f%d' % UniqueNamer.uid
        node.name = self.seen[node.name]
        return node


class Fuser(object):

    """Docstring for Fuser. """

    def __init__(self, blocks, symbol_table):
        """@todo: to be defined1.

        :blocks: @todo
        :_symbol_table: @todo

        """
        self._blocks = blocks
        self._symbol_table = symbol_table
        self._defns = []

    def do_fusion(self):
        """@todo: Docstring for do_fusion.

        :blocks: @todo
        :returns: @todo

        """
        fused_blocks = [[self._blocks.pop(0)]]
        for block in self._blocks:
            if self._is_fusable(fused_blocks[-1][-1], block):
                fused_blocks[-1].append(block)
            else:
                fused_blocks.append([block])
        return list(map(self._fuse, fused_blocks))

    def _is_fusable(self, block_1, block_2):
        """@todo: Docstring for _is_fusable.

        :block_1: @todo
        :block_2: @todo
        :returns: @todo

        """
        if isinstance(block_1, ast.Assign) and \
                isinstance(block_2, ast.Assign) or \
                isinstance(block_2, ast.Return):
            if isinstance(block_1.value, ast.Call) and \
               isinstance(block_2.value, ast.Call):
                func_1 = self._symbol_table[block_1.value.func.id]
                func_2 = self._symbol_table[block_2.value.func.id]
                return isinstance(func_1, Fusable) and \
                    isinstance(func_2, Fusable)
        return False

    def _fuse(self, blocks):
        """@todo: Docstring for _fuse.

        :block_1: @todo
        :block_2: @todo
        :returns: @todo

        """
        if len(blocks) == 1:
            return blocks[0]

        num_args = []
        specializers = []
        kernels = []
        arg_list = []
        arg_nodes_list = []
        outputs = []
        is_return = False
        for block in blocks:
            if isinstance(block, ast.Return):
                is_return = True
            specializer = self._symbol_table[block.value.func.id]
            arg_nodes_list.extend(block.value.args)
            args = tuple(
                self._symbol_table[arg.id] if isinstance(arg, ast.Name) else
                arg.n for arg in block.value.args
            )
            arg_list.extend(args)
            program_cfg = (specializer.args_to_subconfig(args), None)
            output = specializer.generate_output(program_cfg)
            arg_list.append(output)
            if not is_return:
                target = block.targets[0].id
            else:
                target = unique_name()

            self._symbol_table[target] = output
            arg_nodes_list.append(ast.Name(target, ast.Load()))

            outputs.append(output)
            kernels.append(
                specializer.transform(
                    specializer.original_tree,
                    (specializer.args_to_subconfig(args), None)
                ).files[-1]  # TODO: This needs to be more general
            )
            specializers.append(
                specializer.finalize(
                    specializer.transform(
                        specializer.original_tree,
                        (specializer.args_to_subconfig(args), None)
                    ),
                    (specializer.args_to_subconfig(args), None)
                )
            )
            num_args.append(len(block.value.args))

        kernel = UniqueNamer().visit(kernels[0])
        for kern in kernels[1:]:
            UniqueNamer().visit(kern)
            kernel.body[0].defn.extend(kern.body[0].defn)
            kernel.body[0].params.extend(kern.body[0].params)

        fn = FusedFn(specializers, num_args, outputs, is_return)
        program = cl.clCreateProgramWithSource(fn.context,
                                               kernel.codegen()).build()
        # FIXME: Assuming OpenCL
        func_name = unique_kernel_name()
        self._symbol_table[func_name] = fn.finalize(
            program[kernel.body[0].name],
            reduce(lambda x, y: x * y, arg_list[0].shape, 1)
        )
        tree = ast.Call(
            func=ast.Name(id=func_name, ctx=ast.Load()),
            args=arg_nodes_list, keywords=[]
        )
        if is_return:
            tree = ast.Return(tree)
        else:
            tree = ast.Expr(tree)
        return tree


class FusedFn(ConcreteSpecializedFunction):

    """Docstring for FusedFn. """

    def __init__(self, specializers, num_args, outputs, is_return):
        """@todo: to be defined1. """
        ConcreteSpecializedFunction.__init__(self)
        self.specializers = specializers
        self.num_args = num_args
        self.device = cl.clGetDeviceIDs()[-1]
        self.context = cl.clCreateContext([self.device])
        self.queue = cl.clCreateCommandQueue(self.context)
        self.orig_args = ()
        self.arg_buf_map = {}
        self.outputs = outputs
        self.is_return = is_return

    def finalize(self, kernel, global_size):
        self.kernel = kernel
        self.global_size = global_size
        return self

    def _process_args(self, args):
        processed = ()
        argtypes = ()
        for arg in args:
            if isinstance(arg, numpy.ndarray):
                if arg.ctypes.data in self.arg_buf_map:
                    processed += (self.arg_buf_map[arg.ctypes.data],)
                else:
                    buf, evt = cl.buffer_from_ndarray(self.queue, arg,
                                                      blocking=False)
                    evt.wait()
                    processed += (buf,)
                    self.arg_buf_map[arg.ctypes.data] = buf
                argtypes += (cl.cl_mem,)
            else:
                processed += (arg,)
                if isinstance(arg, int):
                    argtypes += (cl.cl_int,)
                elif isinstance(arg, float):
                    argtypes += (cl.cl_float,)
                else:
                    raise NotImplementedError(
                        "UnsupportedType: %s" % type(arg)
                    )
        return processed, argtypes

    def __call__(self, *args):
        processed, argtypes = self._process_args(args)
        self.kernel.argtypes = argtypes
        run_evt = self.kernel(*processed).on(self.queue, self.global_size)
        run_evt.wait()
        return self._process_outputs()

    def _process_outputs(self):
        retvals = ()
        if self.is_return:
            output = self.outputs[-1]
            buf = self.arg_buf_map[output.ctypes.data]
            out, evt = cl.buffer_to_ndarray(self.queue, buf, output)
            evt.wait()
            return out
        for output in self.outputs:
            try:
                buf = self.arg_buf_map[output.ctypes.data]
                out, evt = cl.buffer_to_ndarray(self.queue, buf, output)
                evt.wait()
                retvals += (out,)
            except KeyError:
                # TODO: Make this a better exception
                raise Exception("Could not find corresponding buffer")
        return retvals
        # ret_vals = []
        # for output, out_like in zip(outputs, out_likes):
        #     if isinstance(output, cl.cl_mem):
        #         out, evt = cl.buffer_to_ndarray(self.queue, output,
        #                                         like=out_like)
        #         evt.wait()
        #         ret_vals.append(out)
        #     else:
        #         ret_vals.append(output.value)
        # return ret_vals


class Fusable(object):

    """Docstring for Fusable. """

    def __init__(self):
        """@todo: to be defined1. """
        pass
