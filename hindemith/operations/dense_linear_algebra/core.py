from ctree.jit import ConcreteSpecializedFunction, LazySpecializedFunction
from ctree.frontend import get_ast
from ctree.transformations import PyBasicConversions
from ctree.c.nodes import FunctionDecl, SymbolRef

from ctree.ocl.nodes import OclFile
from ctree.ocl.macros import get_global_id

import ctree.np

import pycl as cl
import numpy as np
import ctypes as ct
import ast
from numpy.ctypeslib import ndpointer
from collections import namedtuple

from hindemith.utils import unique_kernel_name

__author__ = 'leonardtruong'


class DLAConcreteOCL(ConcreteSpecializedFunction):
    def __init__(self):
        self.device = cl.clGetDeviceIDs()[-1]
        self.context = cl.clCreateContext([self.device])
        self.queue = cl.clCreateCommandQueue(self.context)

    def finalize(self, kernel, global_size):
        self.kernel = kernel
        self.global_size = global_size
        return self

    def __call__(self, *args):
        args += (np.empty_like(args[0]), )
        bufs = []
        events = []
        argtypes = ()
        for arg in args:
            buf, evt = cl.buffer_from_ndarray(self.queue, arg, blocking=False)
            bufs.append(buf)
            events.append(evt)
            argtypes += (cl.cl_mem, )
        cl.clWaitForEvents(*events)
        self.kernel.argtypes = argtypes
        run_evt = self.kernel(*bufs).on(self.queue, self.global_size)
        out, evt = cl.buffer_to_ndarray(self.queue, bufs[-1], wait_for=run_evt,
                                        like=args[0])
        evt.wait()
        return out


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


class DLASemanticTransformer(ast.NodeTransformer):
    def visit_For(self, node):
        if isinstance(node.iter, ast.Call) and\
           isinstance(node.iter.func, ast.Attribute):
            if node.iter.func.attr == 'points':
                return PointsLoop(
                    node.target.id, node.iter.func.value.id, node.body
                )
        return node


class DLAOclTransformer(ast.NodeTransformer):
    def __init__(self, arg_cfg):
        self.arg_cfg = arg_cfg
        self.arg_cfg_dict = {}
        self.params = []
        self.project = None

    def visit_Project(self, node):
        self.project = node
        node.files = map(self.visit, node.files)
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
        node.defn = map(self.visit, node.defn)
        return node

    def visit_PointsLoop(self, node):
        self.loop_var = node.loop_var
        body = list(map(self.visit, node.body))
        params = [SymbolRef(x.name, x.type, _global=True)
                  for x in self.params]
        kernel_name = unique_kernel_name()
        kernel = FunctionDecl(None, kernel_name, params, body)
        kernel.set_kernel()
        self.project.files.append(OclFile(kernel_name, [kernel]))

    def visit_SymbolRef(self, node):
        if node.name == self.loop_var:
            return get_global_id(0)
        return node



HMArray = namedtuple("HMArray", ['type', 'ndpointer', 'shape', 'ndim',
                                 'length'])
HMScalar = namedtuple("HMScalar", ['type', 'value'])


class DLALazy(LazySpecializedFunction):
    def __init__(self, tree, backend):
        super(DLALazy, self).__init__(tree)
        self.backend = backend

    def _process_arg(self, arg):
        if isinstance(arg, np.ndarray):
            return HMArray(
                ndpointer(arg.dtype, arg.ndim, arg.shape)(),
                ndpointer(arg.dtype, arg.ndim, arg.shape), arg.shape, arg.ndim,
                reduce(lambda x, y: x * y, arg.shape, 1)
            )
        elif isinstance(arg, int):
            return HMScalar(ct.c_int(), arg)
        elif isinstance(arg, float):
            return HMScalar(ct.c_float(), arg)
        else:
            raise NotImplementedError()

    def args_to_subconfig(self, args):
        return tuple(map(self._process_arg, args))

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        arg_cfg += (HMArray(arg_cfg[0].type, arg_cfg[0].ndpointer,
                            arg_cfg[0].shape, arg_cfg[0].ndim, arg_cfg[0].length), )
        tree = PyBasicConversions().visit(tree)
        tree = DLASemanticTransformer().visit(tree)
        tree = DLAOclTransformer(arg_cfg).visit(tree)
        fn = DLAConcreteOCL()
        kernel = tree.files[-1]
        program = cl.clCreateProgramWithSource(fn.context,
                                            kernel.codegen()).build()
        return fn.finalize(program[kernel.body[0].name], arg_cfg[0].length)


class DLAOp(object):
    def __init__(self, backend='ocl'):
        self.specialized = DLALazy(get_ast(self.op), backend)

    def __call__(self, input1, input2):
        return self.specialized(input1, input2)


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
