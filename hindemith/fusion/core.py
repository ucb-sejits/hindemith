import numpy
import ast
import ctypes as ct
import pycl as cl

from ctree.frontend import get_ast
from ctree.jit import ConcreteSpecializedFunction
import ctree
import ctree.np

from hindemith.utils import unique_kernel_name

ctree.np  # Make PEP happy

import logging
LOG = logging.getLogger('Hindemith')


def fuse(fn_locals, fn_globals):
    def wrapped_fuser(fn):
        def fused(*args, **kwargs):
            tree = get_ast(fn)
            blocks = get_blocks(tree)
            fuser = Fuser(blocks, dict(fn_locals, **fn_globals))
            fused_blocks = fuser.do_fusion()
            tree.body[0].body = fused_blocks
            # Remove Decorator
            tree.body[0].decorator_list = []
            tree = ast.fix_missing_locations(tree)
            exec(compile(tree, '', 'exec')) in fuser._symbol_table
            return fuser._symbol_table[fn.__name__](*args, **kwargs)
        return fused
    return wrapped_fuser


def get_blocks(tree):
    """Convenience method for getting the blocks from an ast

    :tree: @todo
    :returns: @todo

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
        if isinstance(block_1, ast.Assign) and isinstance(block_2, ast.Assign):
            if isinstance(block_1.value, ast.Call) and \
               isinstance(block_2.value, ast.Call):
                func_1 = self._symbol_table[block_1.value.func.id]
                func_2 = self._symbol_table[block_2.value.func.id]
                return hasattr(func_1, 'fusable') and func_1.fusable() and \
                    hasattr(func_2, 'fusable') and func_2.fusable()
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
        for block in blocks:
            specializer = self._symbol_table[block.value.func.id].specialized
            arg_nodes_list.extend(block.value.args)
            args = tuple(
                self._symbol_table[arg.id] if isinstance(arg, ast.Name) else
                arg.n for arg in block.value.args
            )
            arg_list.extend(args)
            output = specializer.generate_output(*args)
            arg_list.append(output)
            self._symbol_table[block.targets[0].id] = output
            arg_nodes_list.append(ast.Name(block.targets[0].id, ast.Load()))
            outputs.append(output)
            kernels.append(
                specializer.fuse_transform(
                    specializer.original_tree,
                    (specializer.args_to_subconfig(args), None)
                )
            )
            specializers.append(
                specializer.transform(
                    specializer.original_tree,
                    (specializer.args_to_subconfig(args), None)
                )
            )
            num_args.append(len(block.value.args))

        kernel = UniqueNamer().visit(kernels[0])
        for kern in kernels[1:]:
            UniqueNamer().visit(kern)
            kernel.body[0].defn.extend(kern.body[0].defn)
            kernel.body[0].params.extend(kern.body[0].params)

        fn = FusedFn(specializers, num_args, outputs)
        program = cl.clCreateProgramWithSource(fn.context,
                                               kernel.codegen()).build()
        # FIXME: Assuming OpenCL
        func_name = unique_kernel_name()
        self._symbol_table[func_name] = fn.finalize(
            program[kernel.body[0].name],
            reduce(lambda x, y: x * y, arg_list[0].shape, 1)
        )
        return ast.Expr(ast.Call(
            func=ast.Name(id=func_name, ctx=ast.Load()),
            args=arg_nodes_list, keywords=[]
        ))
        # return fn.finalize(
        #     program[kernel.body[0].name],
        #     reduce(lambda x, y: x * y, arg_list[0].shape, 1)
        # )(*arg_list)


class FusedFn(ConcreteSpecializedFunction):

    """Docstring for FusedFn. """

    def __init__(self, specializers, num_args, outputs):
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

        # offset = 0
        # processed_args = []
        # processed_types = ()
        # outputs = []
        # output_like = []
        # self.orig_args = args
        # for num_args, specializer in zip(self.num_args, self.specializers):
        #     processed, argtypes, output, out_like = \
        #         self.tmp_process_args(*args[offset:offset + num_args])
        #     offset += num_args
        #     processed_args.extend(processed)
        #     processed_types += argtypes
        #     outputs.append(output)
        #     output_like.append(out_like)
        # return processed_args, processed_types, outputs, output_like

    def tmp_process_args(self, *args):
        processed = []
        events = []
        argtypes = ()
        output = ct.c_int()
        out_like = None
        for arg in args:
            if isinstance(arg, numpy.ndarray):
                if arg.ctypes.data in self.arg_buf_map:
                    processed.append(self.arg_buf_map[arg.ctypes.data])
                    argtypes += (cl.cl_mem,)
                else:
                    buf, evt = cl.buffer_from_ndarray(self.queue, arg,
                                                      blocking=False)
                    processed.append(buf)
                    self.arg_buf_map[arg.ctypes.data] = buf
                    events.append(evt)
                    argtypes += (cl.cl_mem,)
                    output = buf.empty_like_this()
                    out_like = arg
            else:
                processed.append(arg)
                if isinstance(arg, int):
                    argtypes += (cl.cl_int,)
                elif isinstance(arg, float):
                    argtypes += (cl.cl_float,)
                    if isinstance(output, ct.c_int):
                        output = ct.c_float()
                else:
                    raise NotImplementedError(
                        "UnsupportedType: %s" % type(arg)
                    )
        if isinstance(output, cl.cl_mem):
            argtypes += (cl.cl_mem,)
            processed.append(output)
        else:
            processed.append(output.byref)
            if isinstance(output, ct.c_float):
                argtypes += (cl.cl_float,)
            else:
                argtypes += (cl.cl_int,)
        cl.clWaitForEvents(*events)
        return processed, argtypes, output, out_like

    def __call__(self, *args):
        processed, argtypes = self._process_args(args)
        self.kernel.argtypes = argtypes
        run_evt = self.kernel(*processed).on(self.queue, self.global_size)
        run_evt.wait()
        return self._process_outputs()

    def _process_outputs(self):
        retvals = ()
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
