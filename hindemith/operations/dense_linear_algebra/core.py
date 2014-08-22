from ctree.jit import ConcreteSpecializedFunction, LazySpecializedFunction
from ctree.frontend import get_ast
from ctree.transformations import PyBasicConversions
from ctree.c.nodes import FunctionDecl, SymbolRef, ArrayDef, \
    FunctionCall, Ref, Constant, Add, Mul
from ctree.c.macros import NULL
from ctree.templates.nodes import StringTemplate

from ctree.ocl import get_context_from_device
from ctree.ocl.nodes import OclFile
from ctree.ocl.macros import get_global_id, clSetKernelArg

import ctree.np

ctree.np  # Make PEP8 happy

import pycl as cl
import numpy as np
import ctypes as ct
import ast
from numpy import zeros
from numpy.ctypeslib import ndpointer
from collections import namedtuple

from hindemith.utils import unique_kernel_name
from hindemith.fusion.core import Fusable, KernelCall

__author__ = 'leonardtruong'


class DLAConcreteOCL(ConcreteSpecializedFunction):
    device = cl.clGetDeviceIDs()[-1]
    context = get_context_from_device(device)
    queue = cl.clCreateCommandQueue(context)

    def __init__(self, output=None):
        self.context = DLAConcreteOCL.context
        self.queue = DLAConcreteOCL.queue
        self.output = output

    def finalize(self, tree, entry_type, entry_point, kernel):
        self.kernel = kernel
        self._c_function = self._compile(entry_point, tree, entry_type)
        return self

    def process_args(self, *args):
        processed = []
        events = []
        output = ct.c_int()
        out_like = None
        for arg in args:
            if isinstance(arg, np.ndarray):
                buf, evt = cl.buffer_from_ndarray(self.queue, arg,
                                                  blocking=False)
                processed.append(buf)
                events.append(evt)
                output = buf.empty_like_this()
                out_like = arg
            else:
                if isinstance(arg, int):
                    processed.append(arg)
                elif isinstance(arg, float) and isinstance(output, ct.c_int):
                    processed.append(arg)
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
            processed.append(output)
        else:
            processed.append(output.byref)
        cl.clWaitForEvents(*events)
        return processed, output, out_like

    def __call__(self, *args):
        processed, output, out_like = self.process_args(*args)
        self._c_function(self.queue, self.kernel, *processed)
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
    def __init__(self, arg_cfg, fusable_nodes):
        self.arg_cfg = arg_cfg
        self.arg_cfg_dict = {}
        for cfg in arg_cfg:
            if hasattr(cfg, 'ndpointer'):
                self.ndim = cfg.ndim
                self.shape = cfg.shape
                break
        self.params = []
        self.project = None
        self.loop_var = None
        self.fusable_nodes = fusable_nodes

    def visit_Project(self, node):
        self.project = node
        node.files = list(map(self.visit, node.files))
        return node

    def visit_FunctionDecl(self, node):
        """
        :param node:
        :type node: FunctionDef
        """
        if node.kernel is True:
            return node
        for index, arg in enumerate(node.params):
            self.arg_cfg_dict[arg.name] = self.arg_cfg[index]
            if hasattr(self.arg_cfg[index], 'ndpointer'):
                arg.type = self.arg_cfg[index].ndpointer()
            else:
                arg.type = self.arg_cfg[index].ctype()

        self.params = node.params
        node.defn = list(filter(None, map(self.visit, node.defn)))
        params = [
            SymbolRef('queue', cl.cl_command_queue()),
            SymbolRef('kernel', cl.cl_kernel())
        ]
        params.extend(SymbolRef('buf%d' % d, cl.cl_mem())
                      for d in range(len(self.arg_cfg)))
        local_size = 4
        defn = [
            ArrayDef(
                SymbolRef('global', ct.c_ulong()), Constant(self.ndim),
                [Constant(d) for d in self.shape]
            ),
            ArrayDef(
                SymbolRef('local', ct.c_ulong()), Constant(self.ndim),
                [Constant(local_size) for _ in self.shape]
            )
        ]
        setargs = [clSetKernelArg(
                SymbolRef('kernel'), Constant(index),
                Constant(ct.sizeof(arg.ctype)),
                Ref(SymbolRef('buf%d' % index))
            ) for index, arg in enumerate(self.arg_cfg)]
        defn.extend(setargs)
        enqueue_call = FunctionCall(SymbolRef('clEnqueueNDRangeKernel'), [
            SymbolRef('queue'), SymbolRef('kernel'),
            Constant(self.ndim), NULL(),
            SymbolRef('global'), SymbolRef('local'),
            Constant(0), NULL(), NULL()
        ])
        finish_call = FunctionCall(SymbolRef('clFinish'), [SymbolRef('queue')])
        defn.extend([enqueue_call, finish_call])
        header = StringTemplate("""
            #ifdef __APPLE__
            #include <OpenCL/opencl.h>
            #else
            #include <CL/cl.h>
            #endif
            """)
        node.params = params
        node.defn = defn
        self.fusable_nodes.append(
            KernelCall(node, self.project.files[-1].body[0], self.shape, defn[0],
                       tuple(local_size for _ in self.shape), defn[1], enqueue_call,
                       finish_call, setargs)
        )
        return [header, node]

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
            index = get_global_id(self.ndim - 1)
            for d in reversed(range(self.ndim - 1)):
                index = Add(
                    Mul(
                        index,
                        Constant(self.shape[d])
                    ), get_global_id(d)
                )
            return index
        return node


HMArray = namedtuple("HMArray", ['ndpointer', 'shape', 'ndim', 'dtype',
                                 'is_global', 'ctype'])
HMScalar = namedtuple("HMScalar", ['is_global', 'ctype'])


class DLALazy(LazySpecializedFunction, Fusable):
    def __init__(self, tree, backend):
        super(DLALazy, self).__init__(tree)
        self.backend = backend
        self.output = None
        self.fusable_nodes = []

    def _process_arg(self, arg):
        if isinstance(arg, np.ndarray):
            return HMArray(
                ndpointer(arg.dtype, arg.ndim, arg.shape), arg.shape, arg.ndim,
                arg.dtype, True, cl.cl_mem
            )
        elif isinstance(arg, int):
            return HMScalar(False, ct.c_int)
        elif isinstance(arg, float):
            return HMScalar(False, ct.c_float)
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
        output_type = self._process_arg(self.output)
        tree = DLASemanticTransformer().visit(tree)
        tree = DLAOclTransformer(arg_cfg + (output_type, ),
                                 self.fusable_nodes).visit(tree)
        entry_type = [None, cl.cl_command_queue, cl.cl_kernel]
        entry_type.extend(arg.ctype for arg in arg_cfg + (output_type, ))
        entry_point = 'op'
        return tree, entry_type, entry_point

    def finalize(self, tree, entry_type, entry_point):
        fn = DLAConcreteOCL(self.output)
        self.output = None
        self.fusable_nodes = []
        kernel = tree.files[-1]
        program = cl.clCreateProgramWithSource(fn.context,
                                               kernel.codegen()).build()
        kernel_ptr = program[kernel.body[0].name]

        return fn.finalize(tree, ct.CFUNCTYPE(*entry_type), entry_point,
                           kernel_ptr)

    def generate_output(self, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        for arg in arg_cfg:
            if hasattr(arg, 'ndpointer'):
                self.output = zeros(arg.shape, arg.dtype)
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
