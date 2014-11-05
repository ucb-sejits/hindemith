import numpy as np
# from hindemith.linalg import add, sub, mul, div
import pycl as cl
from ctree.ocl import get_context_and_queue_from_devices


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
        obj.__add__ = add
        obj.__sub__ = sub
        obj.__mul__ = mul
        obj.__div__ = div
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

    def __getitem__(self, item):
        if self._host_dirty:
            cl.buffer_to_ndarray(self.queue, self._ocl_buf, self,
                                 blocking=True)
        return np.ndarray.__getitem__(self, item)


from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.c.nodes import FunctionDecl, SymbolRef, For, ArrayRef, Add, Assign, \
    Constant, AddAssign, Lt, Mul, Sub, Div, CFile, FunctionCall, ArrayDef
from ctree.templates.nodes import StringTemplate
from ctree.ocl.macros import clSetKernelArg, NULL, get_global_id
from ctree.ocl.nodes import OclFile
from ctree.ocl import get_context_and_queue_from_devices
from ctree.nodes import Project
import ctree.np
ctree.np
import numpy as np
from collections import namedtuple
import ctypes as ct
import pycl as cl
from functools import reduce
# from hindemith.types import hmarray


NdArrCfg = namedtuple('NdArrCfg', ['dtype', 'ndim', 'shape'])
ScalarCfg = namedtuple('ScalarCfg', ['dtype'])


curr_u = 0


def next_loop_var():
    global curr_u
    curr_u += 1
    return "_l{}".format(curr_u)


def gen_loop_index(loop_vars, shape):
    base = SymbolRef(loop_vars[0])
    for index, var in enumerate(loop_vars[1:]):
        curr = Mul(SymbolRef(var),
                   Constant(reduce(lambda x, y: x * y, shape[:index], 1)))
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
                Lt(SymbolRef(loop_vars[-1]), Constant(r[-1])),
                AddAssign(SymbolRef(loop_vars[-1]), step),
                next_body)
        )
        curr_body = next_body
    curr_body.append(gen_loop_index(loop_vars, r))
    curr_body.extend(body)
    return node


def kernel_range(r, arg_types, body):
    control = [
        clSetKernelArg('kernel', 0, ct.sizeof(cl.cl_mem), 'arg0'),
        clSetKernelArg('kernel', 1, ct.sizeof(cl.cl_mem), 'arg1'),
        clSetKernelArg('kernel', 2, ct.sizeof(cl.cl_mem), 'output'),
        ArrayDef(SymbolRef('global_size', ct.c_size_t()),
                 Constant(len(r)), r),
        ArrayDef(SymbolRef('local_size', ct.c_size_t()),
                 Constant(len(r)), [32 for _ in r]),
        FunctionCall(
            SymbolRef('clEnqueueNDRangeKernel'), [
                SymbolRef('queue'), SymbolRef('kernel'), Constant(len(r)),
                Constant(0), SymbolRef('global_size'), SymbolRef('local_size'),
                Constant(0), NULL(), NULL()
            ]
        )
    ]
    body.insert(0, gen_ocl_loop_index(r))
    kernel = FunctionDecl(
        None,
        SymbolRef('op_kernel'),
        [SymbolRef('arg0', arg_types[0]()),
         SymbolRef('arg1', arg_types[1]()),
         SymbolRef('output', arg_types[2]())],
        body,
    )
    for index, arg in enumerate(arg_types):
        if isinstance(arg(), np.ctypeslib._ndptr):
            kernel.params[index].set_global()
            if index < 2:
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
        self._c_function = self._compile(entry_name, proj, entry_type)

    def __call__(self, *args):
        output = None
        for arg in args:
            if isinstance(arg, np.ndarray):
                output = np.zeros_like(arg)
                break
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
                if arg._ocl_dirty is True:
                    buf, evt = cl.buffer_from_ndarray(self.queue, arg,
                                                      blocking=True)
                    arg._ocl_buf = buf
                    arg._ocl_dirty = False
                evt.wait()
                processed.append(arg._ocl_buf)
            else:
                processed.append(arg)
        print(processed)
        self._c_function(processed[0], processed[1], out_buf, self.queue,
                         self.kernel)
        # buf, evt = cl.buffer_to_ndarray(self.queue, out_buf, output,
        #                                 blockking=True)
        cl.clFinish(self.queue)
        # evt.wait()
        return output


class EltWiseArrayOp(LazySpecializedFunction):
    backend = 'ocl'

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

    def transform(self, tree, program_cfg):
        op = op_map[tree]
        arg_cfg, tune_cfg = program_cfg
        arg_types = ()
        op_args = ()
        kernel_arg_types = ()
        for index, cfg in enumerate(arg_cfg):
            if isinstance(cfg, NdArrCfg):
                if EltWiseArrayOp.backend == 'c':
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
        loop_body = [
            Assign(ArrayRef(SymbolRef('output'), SymbolRef('loop_idx')),
                   op(*op_args))]
        func = FunctionDecl(
            None,
            SymbolRef('op'),
            [SymbolRef('arg0', arg_types[0]()),
             SymbolRef('arg1', arg_types[1]()),
             SymbolRef('output', arg_types[2]())],
            []
        )
        proj = Project([CFile('op', [func])])
        if EltWiseArrayOp.backend == 'c':
            func.defn.append(for_range(arg_cfg[2].shape, 1, loop_body))
            entry_type = ct.CFUNCTYPE(*((None,) + arg_types))
            return CConcreteEltOp('op', proj, entry_type)
        elif EltWiseArrayOp.backend == 'ocl':
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
            return fn.finalize(program['op_kernel'])


add = EltWiseArrayOp('+')
sub = EltWiseArrayOp('-')
mul = EltWiseArrayOp('*')
div = EltWiseArrayOp('/')
