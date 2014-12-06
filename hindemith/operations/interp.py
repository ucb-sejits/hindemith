from ctree.templates.nodes import StringTemplate
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from hindemith.types.hmarray import NdArrCfg, kernel_range, hmarray

lerp_kern_body = """
float x = $x_map[loop_idx];
float y = $y_map[loop_idx];
int xx = (int) x;
int yy = (int) y;
float tx = x - xx;
float ty = y - yy;
if (xx > $x_dim - 2 || yy > $y_dim - 2) {
  $output[loop_idx] = 0;
} else {
  $output[loop_idx] = $input[yy * $x_dim + xx] * (1 - tx) * (1 - ty) + \
                     $input[yy * $x_dim + xx + 1] * tx * (1 - ty) + \
                     $input[(yy + 1) * $x_dim + xx] * (1 - tx) * ty + \
                     $input[(yy + 1) * $x_dim + xx + 1] * tx * ty;
}
"""


ocl_header = StringTemplate("""
                #ifdef __APPLE__
                #include <OpenCL/opencl.h>
                #else
                #include <CL/cl.h>
                #endif
                """)


import pycl as cl
from ctree.c.nodes import SymbolRef, FunctionDecl, CFile, Constant
from ctree.nodes import Project
from ctree.ocl.nodes import OclFile
from ctree.ocl import get_context_and_queue_from_devices
import numpy as np
import ctypes as ct


class OclConcreteLerp(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        self._c_function = self._compile(entry_name, proj, entry_type)
        devices = cl.clGetDeviceIDs()
        self.context, self.queue = get_context_and_queue_from_devices(
            [devices[-1]])

    def finalize(self, kernel):
        self.kernel = kernel
        return self

    def __call__(self, *args):
        output = hmarray(np.empty_like(args[0]))
        output._ocl_buf = cl.clCreateBuffer(self.context, output.nbytes)
        output._ocl_dirty = False
        output._host_dirty = True
        processed = [arg.ocl_buf for arg in args]
        processed = [self.queue, self.kernel] + processed + [output._ocl_buf]
        self._c_function(*processed)
        return output


class LinearInterp(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        arg_cfgs = ()
        out_cfg = None
        for arg in args:
            arg_cfgs += (NdArrCfg(arg.dtype, arg.ndim, arg.shape), )
            out_cfg = (NdArrCfg(arg.dtype, arg.ndim, arg.shape), )
        return arg_cfgs + out_cfg

    def process_arg_types(self, arg_cfg):
        arg_types, params, kernel_params = (), [], ()
        for cfg in arg_cfg:
            arg_types += (cl.cl_mem, )
            params.append(SymbolRef.unique(sym_type=arg_types[-1]()))
            kernel_params += (SymbolRef(
                params[-1].name,
                np.ctypeslib.ndpointer(cfg.dtype, cfg.ndim, cfg.shape)()),)
        return arg_types, params, kernel_params

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        arg_types, params, kernel_params = self.process_arg_types(arg_cfg)
        func = FunctionDecl(
            None,
            SymbolRef('lerp'),
            params,
            []
        )
        proj = Project([CFile('lerp', [func])])
        proj.files[0].body.insert(0, ocl_header)
        arg_types = (cl.cl_command_queue, cl.cl_kernel) + arg_types
        shape = arg_cfg[0].shape[::-1]
        control, kernel = kernel_range(shape, shape,
                                       kernel_params, [StringTemplate(lerp_kern_body, {
                                           "input": SymbolRef(kernel_params[0].name),
                                           "x_map": SymbolRef(kernel_params[1].name),
                                           "y_map": SymbolRef(kernel_params[2].name),
                                           "output": SymbolRef(kernel_params[3].name),
                                           "x_dim": Constant(shape[0]),
                                           "y_dim": Constant(shape[1])
                                       })])
        func.params.insert(0, SymbolRef('queue', cl.cl_command_queue()))
        func.params.insert(1, SymbolRef(kernel.body[0].name.name,
                                        cl.cl_kernel()))
        func.defn = control
        print(func)
        print(kernel)
        proj.files.append(kernel)
        entry_type = (None,) + arg_types
        fn = OclConcreteLerp('lerp', proj, ct.CFUNCTYPE(*entry_type))
        kernel = proj.find(OclFile)
        program = cl.clCreateProgramWithSource(
            fn.context, kernel.codegen()).build()
        return fn.finalize(program[kernel.body[0].name.name])

interp_linear = LinearInterp(None)
