import numpy as np
import pycl as cl
from ctree.ocl import get_context_and_queue_from_devices
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.c.nodes import FunctionDecl, SymbolRef, For, ArrayRef, Add, Assign, \
    Constant, AddAssign, Lt, Mul, Sub, Div, CFile, FunctionCall, ArrayDef, If, \
    And
from ctree.templates.nodes import StringTemplate
from ctree.ocl.macros import clSetKernelArg, NULL, get_global_id
from ctree.ocl.nodes import OclFile
from ctree.nodes import Project
from ctree.omp.nodes import OmpParallelFor
from ctree.omp.macros import IncludeOmpHeader
import ctree.np
ctree.np
from collections import namedtuple
import ctypes as ct
from functools import reduce


class hmarray(np.ndarray):
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
        subtype.__sub__ = sub
        subtype.__mul__ = mul
        subtype.__div__ = div
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
        self.__add__ = getattr(obj, '__add__')

    @property
    def ocl_buf(self):
        if self._ocl_dirty is True:
            buf, evt = cl.buffer_from_ndarray(self.queue, self,
                                              blocking=True)
            evt.wait()
            self._ocl_buf = buf
            self._ocl_dirty = False
        return self._ocl_buf

    def copy_to_host_if_dirty(self):
        if self._host_dirty:
            _, evt = cl.buffer_to_ndarray(self.queue, self._ocl_buf,
                                          self, blocking=True)
            evt.wait()
            self._host_dirty = False
            return _

    def __getitem__(self, item):
        if self._host_dirty:
            _, evt = cl.buffer_to_ndarray(self.queue, self._ocl_buf, self,
                                          blocking=True)
            evt.wait()
            self._host_dirty = False

        return np.ndarray.__getitem__(self, item)


NdArrCfg = namedtuple('NdArrCfg', ['dtype', 'ndim', 'shape'])
ScalarCfg = namedtuple('ScalarCfg', ['dtype'])


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


def gen_ocl_loop_index(shape):
    base = get_global_id(0)
    for index in range(1, len(shape)):
        curr = Mul(get_global_id(index),
                   Constant(reduce(lambda x, y: x * y, shape[:index], 1)))
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


def gen_kernel_cond(global_size, shape):
    conds = ()
    for index, g, s in zip(range(len(global_size)), global_size, shape):
        if s < g:
            conds += (Lt(get_global_id(index), Constant(s)), )
    if len(conds) == 0:
        return None
    cond = conds[0]
    for c in conds[1:]:
        cond = And(c, cond)
    return cond


def kernel_range(r, arg_types, body):
    """TODO: Make local size computation dynamic"""
    control = []
    params = []
    for index, arg in enumerate(arg_types):
        control.append(
            clSetKernelArg('kernel', index, ct.sizeof(cl.cl_mem),
                           'arg{}'.format(index)))
        params.append(SymbolRef('arg{}'.format(index), arg()))
    # devices = cl.clGetDeviceIDs()
    # max_sizes = cl.clGetDeviceInfo(
    #     devices[-1], cl.cl_device_info.CL_DEVICE_MAX_WORK_ITEM_SIZES)
    # max_total = cl.clGetDeviceInfo(
    #     devices[-1], cl.cl_device_info.CL_DEVICE_MAX_WORK_GROUP_SIZE)

    if len(arg_types[0]._shape_) == 2:
        # x_len, y_len = 1, 1
        # while True:
        #     if arg_types[0]._shape_[0] % 2 == 1:
        #         x_len = 1
        #     else:
        #         x_len = min(max_sizes[0], x_len * 2)
        #         if arg_types[0]._shape_[0] % x_len != 0:
        #             x_len /= 2
        #             break
        #     if max_total - x_len * y_len <= 0:
        #         break
        #     if arg_types[0]._shape_[1] % 2 == 1:
        #         y_len = 1
        #     else:
        #         y_len = min(max_sizes[1], y_len * 2)
        #         if arg_types[0]._shape_[1] % y_len != 0:
        #             y_len /= 2
        #             break
        #     if max_total - x_len * y_len <= 0:
        #         break
        #     if x_len == arg_types[0]._shape_[0] or \
        #             y_len == arg_types[0]._shape_[1]:
        #         break

        # local_size = (x_len, y_len)
        local_size = (32, 32)
    else:
        local_size = (32, )
        # local_size = (min(
        #     max_total, max_sizes[0], arg_types[0]._shape_[0] / 2), )
    global_size = ()
    for d in r:
        if d % 32 != 0:
            global_size += ((d + 31) & (~31),)
        else:
            global_size += (d,)
    control.extend([
        ArrayDef(SymbolRef('global_size', ct.c_size_t()),
                 Constant(len(r)), global_size),
        ArrayDef(SymbolRef('local_size', ct.c_size_t()),
                 Constant(len(r)), local_size),
        FunctionCall(
            SymbolRef('clEnqueueNDRangeKernel'), [
                SymbolRef('queue'), SymbolRef('kernel'), Constant(len(r)),
                Constant(0), SymbolRef('global_size'), SymbolRef('local_size'),
                Constant(0), NULL(), NULL()
            ]
        ),
        FunctionCall(SymbolRef('clFinish'), [SymbolRef('queue')])
    ])
    body.insert(0, gen_ocl_loop_index(r))
    cond = gen_kernel_cond(global_size, r)
    if cond:
        body = If(cond, body)
    kernel = FunctionDecl(
        None,
        SymbolRef('kern'),
        params,
        body
    )
    for index, arg in enumerate(arg_types):
        if isinstance(arg(), np.ctypeslib._ndptr):
            kernel.params[index].set_global()
            if index < len(arg_types) - 1:
                kernel.params[index].set_const()
    kernel.set_kernel()
    return control, OclFile('op_file', [kernel])


py_to_ctypes = {
    int: ct.c_int,
    float: ct.c_float
}

op_map = {
    '+': Add,
    '-': Sub,
    '*': Mul,
    '/': Div
}


class CConcreteEltOp(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        print(proj.files[0])
        self._c_function = self._compile(entry_name, proj, entry_type)

    def __call__(self, *args):
        output = None
        for arg in args:
            if isinstance(arg, hmarray):
                if output is None:
                    output = hmarray(np.zeros_like(arg))
                arg.copy_to_host_if_dirty()
        self._c_function(args[0], args[1], output)
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
                    output = hmarray(np.empty_like(arg))
                    out_buf, evt = cl.buffer_from_ndarray(self.queue, output,
                                                          blocking=True)
                    output._ocl_buf = out_buf
                    output._ocl_dirty = False
                    output._host_dirty = True
                evt.wait()
                processed.append(arg.ocl_buf)
            else:
                processed.append(arg)
        self._c_function(processed[0], processed[1], out_buf, self.queue,
                         self.kernel)
        return output


class EltWiseArrayOp(LazySpecializedFunction):
    backend = 'c'

    def args_to_subconfig(self, args):
        arg_cfgs = ()
        out_cfg = None
        for arg in args:
            if isinstance(arg, hmarray):
                arg_cfgs += (NdArrCfg(arg.dtype, arg.ndim, arg.shape), )
                out_cfg = (NdArrCfg(arg.dtype, arg.ndim, arg.shape), )
            else:
                arg_cfgs += (ScalarCfg(type(arg)), )
        return arg_cfgs + out_cfg

    def process_arg_cfg(self, arg_cfg):
        arg_types = ()
        op_args = ()
        kernel_arg_types = ()
        for index, cfg in enumerate(arg_cfg):
            if isinstance(cfg, NdArrCfg):
                if self.backend in {'c', 'omp'}:
                    arg_types += (np.ctypeslib.ndpointer(
                        cfg.dtype, cfg.ndim, cfg.shape), )
                    if index < 2:
                        op_args += (ArrayRef(SymbolRef('arg{}'.format(index)),
                                             SymbolRef('loop_idx')), )
                else:
                    arg_types += (cl.cl_mem, )
                    if index < 2:
                        op_args += (ArrayRef(SymbolRef('arg{}'.format(index)),
                                             SymbolRef('loop_idx')), )
                    kernel_arg_types += (np.ctypeslib.ndpointer(
                        cfg.dtype, cfg.ndim, cfg.shape), )
            else:
                arg_types += (py_to_ctypes[cfg.dtype], )
                if index < 2:
                    op_args += (SymbolRef('arg{}'.format(index)), )
                if EltWiseArrayOp.backend == 'ocl':
                    kernel_arg_types += (py_to_ctypes[cfg.dtype], )
        return arg_types, op_args, kernel_arg_types

    def transform(self, tree, program_cfg):
        op = op_map[tree]
        arg_cfg, tune_cfg = program_cfg
        arg_types, op_args, kernel_arg_types = self.process_arg_cfg(arg_cfg)
        loop_body = [
            Assign(ArrayRef(SymbolRef('arg2'), SymbolRef('loop_idx')),
                   op(*op_args))]
        func = FunctionDecl(
            None,
            SymbolRef('op'),
            [SymbolRef('arg0', arg_types[0]()),
             SymbolRef('arg1', arg_types[1]()),
             SymbolRef('arg2', arg_types[2]())],
            []
        )
        proj = Project([CFile('op', [func])])
        if self.backend in {'c', 'omp'}:
            if self.backend == 'omp':
                func.defn.append(OmpParallelFor())
                proj.files[0].config_target = 'omp'
            proj.files[0].body.insert(0, IncludeOmpHeader())
            func.defn.append(for_range(arg_cfg[2].shape, 1, loop_body))
            entry_type = ct.CFUNCTYPE(*((None,) + arg_types))
            return CConcreteEltOp('op', proj, entry_type)
        elif self.backend == 'ocl':
            proj.files[0].body.insert(0, StringTemplate("""
                #ifdef __APPLE__
                #include <OpenCL/opencl.h>
                #else
                #include <CL/cl.h>
                #endif
                """))
            func.params.extend((
                SymbolRef('queue', cl.cl_command_queue()),
                SymbolRef('kernel', cl.cl_kernel())
            ))
            arg_types += (cl.cl_command_queue, cl.cl_kernel)
            control, kernel = kernel_range(arg_cfg[2].shape,
                                           kernel_arg_types, loop_body)
            func.defn = control
            entry_type = ct.CFUNCTYPE(*((None,) + arg_types))
            fn = OclConcreteEltOp('op', proj, entry_type)
            print(func)
            print(kernel)
            program = cl.clCreateProgramWithSource(
                fn.context, kernel.codegen()).build()
            return fn.finalize(program['kern'])


spec_add = EltWiseArrayOp('+')
spec_sub = EltWiseArrayOp('-')
spec_mul = EltWiseArrayOp('*')
spec_div = EltWiseArrayOp('/')


def add(a, b):
    return spec_add(a, b)


def sub(a, b):
    return spec_sub(a, b)


def mul(a, b):
    return spec_mul(a, b)


def div(a, b):
    return spec_div(a, b)
