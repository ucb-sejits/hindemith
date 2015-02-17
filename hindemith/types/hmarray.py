import numpy as np
import pycl as cl
from ctree.ocl import get_context_and_queue_from_devices
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.c.nodes import FunctionDecl, SymbolRef, For, ArrayRef, Add, Assign, \
    Constant, AddAssign, Lt, Mul, Sub, Div, CFile
from ctree.templates.nodes import StringTemplate
from ctree.ocl.nodes import OclFile
from ctree.nodes import Project
from ctree.omp.nodes import OmpParallelFor
from ctree.omp.macros import IncludeOmpHeader
import ctree.np
ctree.np
from collections import namedtuple
import ctypes as ct
from functools import reduce

from hindemith.nodes import kernel_range
# from hindemith.meta.merge import MergeableInfo, FusableKernel, kernel_range
# , LoopDependence

import copy
import ast


class hmarray(np.ndarray):
    buffer_cache = {}

    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        """
        ---IMPORTANT---
        Read this link before making changes,
        http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        ---------------

        The hmarray constructor accepts the normal ndarray prototype,
        but also allows marshalling an existing ndarray into the
        hmarray subclass by passing it in as the first argument.
        """
        if isinstance(shape, np.ndarray):
            # Set to subtype if marshalling an existing ndarray
            obj = np.asarray(shape).view(subtype)
        else:
            obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset,
                                     strides, order)
        subtype.__add__ = add
        subtype.__radd__ = add
        subtype.__sub__ = sub
        subtype.__rsub__ = sub
        subtype.__mul__ = mul
        subtype.__rmul__ = mul
        subtype.__div__ = div
        subtype.__rdiv__ = div
        obj._ocl_buf = None
        obj._host_dirty = False
        obj._ocl_dirty = True
        devices = cl.clGetDeviceIDs()
        obj.context, obj.queue = get_context_and_queue_from_devices(
            [devices[-1]])
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._ocl_buf = getattr(obj, '_ocl_buf', None)
        self._host_dirty = getattr(obj, '_host_dirty', False)
        self._ocl_dirty = getattr(obj, '_ocl_dirty', True)
        self.queue = getattr(obj, 'queue', None)
        devices = cl.clGetDeviceIDs()
        self.context, self.queue = get_context_and_queue_from_devices(
            [devices[-1]])

    @property
    def ocl_buf(self):
        if self._ocl_dirty is True:
            if self._ocl_buf is None:
                # try:
                #     self._ocl_buf = self.buffer_cache[self.nbytes].pop()
                # except (KeyError, IndexError):
                self._ocl_buf = cl.clCreateBuffer(self.context,
                                                  self.nbytes)

            _, evt = cl.buffer_from_ndarray(self.queue, self,
                                            self._ocl_buf, blocking=False)
            evt.wait()
            self._ocl_dirty = False
        return self._ocl_buf

    def copy_to_host_if_dirty(self):
        if self._host_dirty:
            _, evt = cl.buffer_to_ndarray(self.queue, self._ocl_buf,
                                          self, blocking=False)
            evt.wait()
            self._host_dirty = False
        return self

    # def __del__(self):
    #     if self.shape not in self.buffer_cache:
    #         self.buffer_cache[self.shape] = [self]
    #     else:
    #         self.buffer_cache[self.shape].append(self)
        # if self._ocl_buf is not None:
        #     try:
        #         self.buffer_cache[self.nbytes].append(self._ocl_buf)
        #     except KeyError:
        #         self.buffer_cache[self.nbytes] = [self._ocl_buf]


def empty(shape, _type):
    if shape in hmarray.buffer_cache:
        bucket = hmarray.buffer_cache[shape]
        if len(bucket) > 0:
            # print('Cached')
            arr = bucket.pop()
            arr._ocl_buf = cl.clCreateBuffer(arr.context, arr.nbytes)
            arr._ocl_dirty = False
            arr._host_dirty = False
            return arr
    arr = hmarray(shape, _type)
    arr._ocl_buf = cl.clCreateBuffer(arr.context, arr.nbytes)
    arr._ocl_dirty = False
    return arr


def empty_like(arr):
    return empty(arr.shape, arr.dtype)


def zeros(size, dtype):
    return hmarray(np.zeros(size, dtype))


indices_cache = {}


def indices(size):
    if size not in indices_cache:
        y, x = np.indices(size).astype(np.float32)
        indices_cache[size] = (hmarray(y), hmarray(x))
    return indices_cache[size]


NdArrCfg = namedtuple('NdArrCfg', ['dtype', 'ndim', 'shape'])
ScalarCfg = namedtuple('ScalarCfg', ['value'])


class LoopVarGenerator():
    def __init__(self):
        self.curr = 0

    def __call__(self):
        self.curr += 1
        return "_l{}".format(self.curr)

next_loop_var = LoopVarGenerator()


def gen_loop_index(loop_vars, shape):
    base = SymbolRef(loop_vars[-1])
    for index, var in reversed(list(enumerate(loop_vars[:-1]))):
        curr = Mul(SymbolRef(var),
                   Constant(reduce(lambda x, y: x * y, shape[:index + 1], 1)))
        base = Add(curr, base)
    return Assign(SymbolRef('loop_idx', ct.c_int()), base)


def for_range(r, step, body):
    loop_vars = []
    curr_body = []
    loop_vars.append(next_loop_var())
    node = For(Assign(SymbolRef(loop_vars[-1], ct.c_int()), Constant(0)),
               Lt(SymbolRef(loop_vars[-1]), Constant(r[-1])),
               AddAssign(SymbolRef(loop_vars[-1]), step),
               curr_body)
    for dim in r[:-1]:
        next_body = []
        loop_vars.append(next_loop_var())
        curr_body.append(
            For(Assign(SymbolRef(loop_vars[-1], ct.c_int()), Constant(0)),
                Lt(SymbolRef(loop_vars[-1]), Constant(dim)),
                AddAssign(SymbolRef(loop_vars[-1]), step),
                next_body)
        )
        curr_body = next_body
    curr_body.append(gen_loop_index(loop_vars, r))
    curr_body.extend(body)
    return node


py_to_ctypes = {
    int: ct.c_int,
    float: ct.c_float
}

op_map = {
    ast.Add: Add,
    ast.Sub: Sub,
    ast.Mult: Mul,
    ast.Div: Div
}


class CConcreteEltOp(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        self._c_function = self._compile(entry_name, proj, entry_type)

    def __call__(self, *args):
        output = None
        processed = []
        for arg in args:
            if isinstance(arg, hmarray):
                arg.copy_to_host_if_dirty()
                processed.append(arg)
                if output is None:
                    output = hmarray(np.zeros_like(arg))
        self._c_function(*(processed + [output]))
        return output


class OclConcreteEltOp(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        self._c_function = self._compile(entry_name, proj, entry_type)
        devices = cl.clGetDeviceIDs()
        self.context, self.queue = get_context_and_queue_from_devices(
            [devices[-1]])

    def finalize(self, kernel):
        self.kernel = kernel
        return self

    def __call__(self, *args):
        output = None
        out_buf = None
        processed = []
        for arg in args:
            if isinstance(arg, hmarray):
                if output is None:
                    output = hmarray(np.zeros_like(arg))
                    out_buf = cl.clCreateBuffer(self.context, output.nbytes)
                    output._ocl_buf = out_buf
                    output._ocl_dirty = False
                    output._host_dirty = True
                processed.append(arg.ocl_buf)
            # else:
            #     processed.append(arg)
        self._c_function(*([self.queue, self.kernel] + processed + [out_buf]))
        return output


class EltWiseArrayOp(LazySpecializedFunction):
    backend = 'ocl'
    fusable = True

    def args_to_subconfig(self, args):
        arg_cfgs = ()
        out_cfg = None
        for arg in args:
            if isinstance(arg, hmarray):
                arg_cfgs += (NdArrCfg(arg.dtype, arg.ndim, arg.shape), )
                out_cfg = (NdArrCfg(arg.dtype, arg.ndim, arg.shape), )
            elif type(arg) in {int, float}:
                arg_cfgs += (ScalarCfg(arg), )
            else:
                raise Exception("EltWiseArrayOp can't handle this arg " +
                                "type: {}".format(type(arg)))
        return arg_cfgs + out_cfg

    def process_arg_cfg(self, arg_cfg):
        arg_types = ()
        op_args = ()
        kernel_params = ()
        params = []
        for index, cfg in enumerate(arg_cfg):
            if isinstance(cfg, NdArrCfg):
                if self.backend in {'c', 'omp'}:
                    arg_types += (np.ctypeslib.ndpointer(
                        cfg.dtype, cfg.ndim, cfg.shape), )
                    unique = SymbolRef.unique(sym_type=arg_types[-1]())
                    params.append(unique)
                    if index < 2:
                        op_args += (ArrayRef(SymbolRef(unique.name),
                                             SymbolRef('loop_idx')), )
                else:
                    arg_types += (cl.cl_mem, )
                    unique = SymbolRef.unique(sym_type=arg_types[-1]())
                    params.append(unique)
                    if index < 2:
                        op_args += (ArrayRef(SymbolRef(unique.name),
                                             SymbolRef('loop_idx')), )
                    kernel_params += (
                        SymbolRef(unique.name,
                                  np.ctypeslib.ndpointer(
                                      cfg.dtype, cfg.ndim, cfg.shape)()), )
            else:
                # arg_types += (py_to_ctypes[cfg.dtype], )
                if index < 2:
                    op_args += (Constant(cfg.value), )
                # if EltWiseArrayOp.backend == 'ocl':
                #     kernel_arg_types += (py_to_ctypes[cfg.dtype], )
        return arg_types, op_args, kernel_params, params

    def transform(self, tree, program_cfg):
        op = op_map[tree.__class__]
        arg_cfg, tune_cfg = program_cfg
        arg_types, op_args, kernel_params, params = \
            self.process_arg_cfg(arg_cfg)
        loop_body = [
            Assign(ArrayRef(SymbolRef(params[-1].name), SymbolRef('loop_idx')),
                   op(*op_args))]
        func = FunctionDecl(
            None,
            SymbolRef('op'),
            params,
            []
        )
        cfile = CFile('op', [func])
        if self.backend in {'c', 'omp'}:
            if self.backend == 'omp':
                cfile.body.insert(0, IncludeOmpHeader())
                func.defn.append(OmpParallelFor())
                cfile.config_target = 'omp'
            func.defn.append(for_range(arg_cfg[2].shape, 1, loop_body))
            return [cfile]
        elif self.backend == 'ocl':
            cfile.config_target = 'opencl'
            cfile.body.insert(0, StringTemplate("""
                #ifdef __APPLE__
                #include <OpenCL/opencl.h>
                #else
                #include <CL/cl.h>
                #endif
                """))
            shape = arg_cfg[2].shape
            control, kernel = kernel_range(shape, shape,
                                           kernel_params, loop_body)
            func.defn = control

            func.params.insert(0, SymbolRef('queue', cl.cl_command_queue()))
            func.params.insert(1, SymbolRef(kernel.body[0].name.name,
                                            cl.cl_kernel()))
            return [cfile, kernel]

    def finalize(self, files, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        proj = Project(files)
        arg_types, op_args, kernel_params, params = \
            self.process_arg_cfg(arg_cfg)
        entry_name = 'op'
        if self.backend == 'c':
            entry_type = ct.CFUNCTYPE(*((None,) + arg_types))
            return CConcreteEltOp(entry_name, proj, entry_type)
        elif self.backend == 'ocl':
            arg_types = (cl.cl_command_queue, cl.cl_kernel) + arg_types
            entry_type = ct.CFUNCTYPE(*((None,) + arg_types))
            fn = OclConcreteEltOp(entry_name, proj, entry_type)
            kernel = proj.find(OclFile)
            program = cl.clCreateProgramWithSource(
                fn.context, kernel.codegen()).build()
            return fn.finalize(program[kernel.name])

    def get_placeholder_output(self, args):
        return hmarray(np.empty_like(args[0]))

    def get_ir_nodes(self, args):
        tree = copy.deepcopy(self.original_tree)
        arg_cfg = self.args_to_subconfig(args)
        op = op_map[tree.__class__]
        params = []
        op_args = ()
        types = []
        for index, cfg in enumerate(arg_cfg):
            if isinstance(cfg, NdArrCfg):
                types += (np.ctypeslib.ndpointer(
                    cfg.dtype, cfg.ndim, cfg.shape), )
                unique = SymbolRef.unique(sym_type=types[-1]())
                params.append(unique)
                if index < 2:
                    op_args += (ArrayRef(SymbolRef(unique.name),
                                         SymbolRef('loop_idx')), )
            else:
                if index < 2:
                    op_args += (Constant(cfg.value), )
        loop_body = [
            Assign(ArrayRef(SymbolRef(params[-1].name), SymbolRef('loop_idx')),
                   op(*op_args))]
        shape = arg_cfg[2].shape
        return [Loop(shape, params[:-1], [params[-1]], types, loop_body)]


class HmIRNode(object):
    pass


class Loop(HmIRNode):
    def __init__(self, shape, sources, sinks, types, body, local_mem=None):
        self.shape = shape
        self.sources = sources
        self.sinks = sinks
        self.types = types
        self.body = body
        self.local_mem = local_mem
        if local_mem is None:
            self.local_mem = []


spec_add = EltWiseArrayOp(ast.Add())
spec_sub = EltWiseArrayOp(ast.Sub())
spec_mul = EltWiseArrayOp(ast.Mult())
spec_div = EltWiseArrayOp(ast.Div())


def composable(specializer):
    def wrapper(fn):
        fn.composable = True
        fn.specializer = specializer
        return fn
    return wrapper


@composable(spec_add)
def add(a, b):
    return spec_add(a, b)


@composable(spec_sub)
def sub(a, b):
    return spec_sub(a, b)


@composable(spec_mul)
def mul(a, b):
    return spec_mul(a, b)


@composable(spec_div)
def div(a, b):
    return spec_div(a, b)


def square(a):
    return spec_mul(a, a)
