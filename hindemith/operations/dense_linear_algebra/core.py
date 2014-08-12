from ctree.jit import ConcreteSpecializedFunction, LazySpecializedFunction
from ctree.frontend import get_ast
from ctree.transformations import PyBasicConversions
from ctree.c.nodes import FunctionDecl, SymbolRef

from ctree.ocl.nodes import OclFile
from ctree.ocl.macros import get_global_id

import ctree.np

ctree.np  # Make PEP8 happy

import pycl as cl
import numpy as np
import ctypes as ct
import ast
from numpy import zeros_like
from numpy.ctypeslib import ndpointer
from collections import namedtuple

from hindemith.utils import unique_kernel_name
from hindemith.fusion.core import Fusable

__author__ = 'leonardtruong'


class DLAConcreteOCL(ConcreteSpecializedFunction):
    device = cl.clGetDeviceIDs()[-1]
    context = cl.clCreateContext([device])
    queue = cl.clCreateCommandQueue(context)

    def __init__(self, output=None):
        self.context = DLAConcreteOCL.context
        self.queue = DLAConcreteOCL.queue
        self.output = output

    def finalize(self, kernel, global_size):
        self.kernel = kernel
        self.global_size = global_size
        return self

    def process_args(self, *args):
        processed = []
        events = []
        argtypes = ()
        output = ct.c_int()
        out_like = None
        for arg in args:
            if isinstance(arg, np.ndarray):
                buf, evt = cl.buffer_from_ndarray(self.queue, arg,
                                                  blocking=False)
                processed.append(buf)
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
        if self.output is not None:
            output, evt = cl.buffer_from_ndarray(self.queue, self.output,
                                                 blocking=False)
            out_like = self.output
            evt.wait()
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
        processed, argtypes, output, out_like = self.process_args(*args)
        self.kernel.argtypes = argtypes
        run_evt = self.kernel(*processed).on(self.queue, self.global_size)
        run_evt.wait()
        return self.process_output(output, out_like)

    def process_output(self, output, out_like=None):
        if isinstance(output, cl.cl_mem):
            out, evt = cl.buffer_to_ndarray(self.queue, output,
                                            like=out_like)
            evt.wait()
            return out
        else:
            return output.value


class PointsLoop(ast.AST):
    _fields = ['loop_var', 'iter_target', 'body']

    def __init__(self, loop_var=None, iter_target=None, body=None):
        self.loop_var = loop_var
        self.iter_target = iter_target
        if body is not None:
            self.body = body
        else:
            self.body = []

    def label(self):
        return r"loop_var: %s\n iter_target: %s" % (self.loop_var,
                                                    self.iter_target)


class DLASemanticTransformer(PyBasicConversions):
    def visit_For(self, node):
        if isinstance(node.iter, ast.Call) and\
           isinstance(node.iter.func, ast.Attribute):
            if node.iter.func.attr == 'points':
                return PointsLoop(
                    node.target.id, node.iter.func.value.id,
                    list(map(self.visit, node.body))
                )
        return node


class DLAOclTransformer(ast.NodeTransformer):
    def __init__(self, arg_cfg):
        self.arg_cfg = arg_cfg
        self.arg_cfg_dict = {}
        self.params = []
        self.project = None
        self.loop_var = None

    def visit_Project(self, node):
        self.project = node
        node.files = list(map(self.visit, node.files))
        return node

    def visit_FunctionDecl(self, node):
        """
        :param node:
        :type node: FunctionDef
        """
        for index, arg in enumerate(node.params):
            self.arg_cfg_dict[arg.name] = self.arg_cfg[index]
            arg.type = self.arg_cfg[index].type
        self.params = node.params
        node.defn = list(filter(None, map(self.visit, node.defn)))
        return node

    def visit_PointsLoop(self, node):
        self.loop_var = node.loop_var
        body = list(map(self.visit, node.body))
        params = [SymbolRef(x.name, x.type,
                            _global=self.arg_cfg_dict[x.name].is_global)
                  for x in self.params]
        kernel_name = unique_kernel_name()
        kernel = FunctionDecl(None, kernel_name, params, body)
        kernel.set_kernel()
        self.project.files.append(OclFile(kernel_name, [kernel]))
        self.loop_var = None

    def visit_SymbolRef(self, node):
        if node.name == self.loop_var:
            return get_global_id(0)
        return node


HMArray = namedtuple("HMArray", ['type', 'ndpointer', 'shape', 'ndim',
                                 'length', 'is_global', 'data'])
HMScalar = namedtuple("HMScalar", ['type', 'value', 'is_global'])


class DLALazy(LazySpecializedFunction, Fusable):
    def __init__(self, tree, backend):
        super(DLALazy, self).__init__(tree)
        self.backend = backend
        self.output = None

    def _process_arg(self, arg):
        if isinstance(arg, np.ndarray):
            return HMArray(
                ndpointer(arg.dtype, arg.ndim, arg.shape)(),
                ndpointer(arg.dtype, arg.ndim, arg.shape), arg.shape, arg.ndim,
                reduce(lambda x, y: x * y, arg.shape, 1), True, arg
            )
        elif isinstance(arg, int):
            return HMScalar(ct.c_int(), arg, False)
        elif isinstance(arg, float):
            return HMScalar(ct.c_float(), arg, False)
        else:
            raise NotImplementedError(
                "UnsupportedType: %s" % type(arg)
            )

    def args_to_subconfig(self, args):
        return tuple(map(self._process_arg, args))

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        # FIXME: Assumes all scalars are floats
        if self.output is None:
            self.generate_output(program_cfg)
        output_type = (HMScalar(ct.POINTER(ct.c_float)(), 0, True),)
        for arg in arg_cfg:
            if hasattr(arg, 'ndpointer'):
                output_type = (
                    HMArray(arg.type, arg.ndpointer, arg.shape,
                            arg.ndim, arg.length, True, self.output),)
                break

        tree = DLASemanticTransformer().visit(tree)
        tree = DLAOclTransformer(arg_cfg + output_type).visit(tree)
        return tree

    def finalize(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        global_size = 1
        for arg in arg_cfg:
            if hasattr(arg, 'ndpointer'):
                global_size = arg.length
                break
        fn = DLAConcreteOCL(self.output)
        self.output = None
        kernel = tree.files[-1]
        program = cl.clCreateProgramWithSource(fn.context,
                                               kernel.codegen()).build()
        return fn.finalize(program[kernel.body[0].name], global_size)

    def generate_output(self, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        for arg in arg_cfg:
            if hasattr(arg, 'ndpointer'):
                self.output = zeros_like(arg.data)
                return self.output
        return 0


class DLAOp(object):
    def __new__(cls, backend='ocl'):
        return DLALazy(get_ast(cls.op), backend)


class ArrayAdd(DLAOp):
    def op(input1, input2, output):
        for x in input1.points():
            output[x] = input1[x] + input2[x]


class ArrayMul(DLAOp):
    def op(input1, input2, output):
        for x in input1.points():
            output[x] = input1[x] * input2[x]


class ArraySub(DLAOp):
    def op(input1, input2, output):
        for x in input1.points():
            output[x] = input1[x] - input2[x]


class ArrayDiv(DLAOp):
    def op(input1, input2, output):
        for x in input1.points():
            output[x] = input1[x] / input2[x]


class ScalarArrayAdd(DLAOp):
    def op(input1, input2, output):
        for x in input2.points():
            output[x] = input1 + input2[x]


class ScalarArrayMul(DLAOp):
    def op(input1, input2, output):
        for x in input2.points():
            output[x] = input1 * input2[x]


class ScalarArraySub(DLAOp):
    def op(input1, input2, output):
        for x in input2.points():
            output[x] = input1 - input2[x]


class ScalarArrayDiv(DLAOp):
    def op(input1, input2, output):
        for x in input2.points():
            output[x] = input1 / input2[x]


class ArrayScalarAdd(DLAOp):
    def op(input1, input2, output):
        for x in input1.points():
            output[x] = input1[x] + input2


class ArrayScalarMul(DLAOp):
    def op(input1, input2, output):
        for x in input1.points():
            output[x] = input1[x] * input2


class ArrayScalarSub(DLAOp):
    def op(input1, input2, output):
        for x in input1.points():
            output[x] = input1[x] - input2


class ArrayScalarDiv(DLAOp):
    def op(input1, input2, output):
        for x in input1.points():
            output[x] = input1[x] / input2

array_add = ArrayAdd()
array_sub = ArraySub()
array_mul = ArrayMul()
array_div = ArrayDiv()

scalar_array_add = ScalarArrayAdd()
scalar_array_sub = ScalarArraySub()
scalar_array_mul = ScalarArrayMul()
scalar_array_div = ScalarArrayDiv()

array_scalar_add = ArrayScalarAdd()
array_scalar_sub = ArrayScalarSub()
array_scalar_mul = ArrayScalarMul()
array_scalar_div = ArrayScalarDiv()
