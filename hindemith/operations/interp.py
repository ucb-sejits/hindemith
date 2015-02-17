from ctree.templates.nodes import StringTemplate
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from hindemith.types.hmarray import NdArrCfg, kernel_range, hmarray, empty_like, Loop
import copy
import ast

# lerp_kern_body = """
# float x = $x_map[loop_idx];
# float y = $y_map[loop_idx];
# int xx = (int) x;
# int yy = (int) y;
# float tx = x - xx;
# float ty = y - yy;
# if (xx > $x_dim - 5 || yy > $y_dim - 5 || xx < 1 || yy < 1) {
#   $output[loop_idx] = 0;
# } else {
#     int indx1 = (yy - 1) * $x_dim + xx - 1;
#     float val1 = $input[indx1 + 1] + 0.5 * ty*($input[indx1 + 2] - $input[indx1] + ty*(2.0*$input[indx1] - 5.0*$input[indx1 + 1] + 4.0*$input[indx1 + 2] - $input[indx1 + 3] + ty*(3.0*($input[indx1 + 1] - $input[indx1 + 2]) + $input[indx1 + 3] - $input[indx1])));

#     int indx2 = yy * $x_dim + xx - 1;
#     float val2 = $input[indx2 + 1] + 0.5 * ty*($input[indx2 + 2] - $input[indx2] + ty*(2.0*$input[indx2] - 5.0*$input[indx2 + 1] + 4.0*$input[indx2 + 2] - $input[indx2 + 3] + ty*(3.0*($input[indx2 + 1] - $input[indx2 + 2]) + $input[indx2 + 3] - $input[indx2])));

#     int indx3 = (yy + 1) * $x_dim + xx - 1;
#     float val3 = $input[indx3 + 1] + 0.5 * ty*($input[indx3 + 2] - $input[indx3] + ty*(2.0*$input[indx3] - 5.0*$input[indx3 + 1] + 4.0*$input[indx3 + 2] - $input[indx3 + 3] + ty*(3.0*($input[indx3 + 1] - $input[indx3 + 2]) + $input[indx3 + 3] - $input[indx3])));

#     int indx4 = (yy + 2) * $x_dim + xx - 1;
#     float val4 = $input[indx4 + 1] + 0.5 * ty*($input[indx4 + 2] - $input[indx4] + ty*(2.0*$input[indx4] - 5.0*$input[indx4 + 1] + 4.0*$input[indx4 + 2] - $input[indx4 + 3] + ty*(3.0*($input[indx4 + 1] - $input[indx4 + 2]) + $input[indx4 + 3] - $input[indx4])));
#     $output[loop_idx] = val2 + 0.5 * tx*(val3 - val1 + tx*(2.0*val1 - 5.0*val2 + 4.0*val3 - val4 + tx*(3.0*(val2 - val3) + val4 - val1)));
# }
# """
lerp_kern_body = """
float x = $x_map[loop_idx];
float y = $y_map[loop_idx];
int xx = (int) x;
int yy = (int) y;
float tx = x - xx;
float ty = y - yy;
if (xx > $x_dim - 2 || yy > $y_dim - 2 || xx < 0 || yy < 0) {
  $output[loop_idx] = 0;
} else {
  int idx = xx + yy * $x_dim;
  $output[loop_idx] = $input[idx] * (1 - tx) * (1 - ty) + \
                      $input[idx + 1] * tx * (1 - ty) + \
                      $input[idx + $x_dim] * (1 - tx) * ty + \
                      $input[idx + 1 + $x_dim] * tx * ty;
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
        output = empty_like(args[1])
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
        proj = Project([CFile('lerp', [func], config_target='opencl')])
        proj.files[0].body.insert(0, ocl_header)
        shape = arg_cfg[1].shape
        control, kernel = kernel_range(shape, shape,
                                       kernel_params, [StringTemplate(lerp_kern_body, {
                                           "input": SymbolRef(kernel_params[0].name),
                                           "x_map": SymbolRef(kernel_params[1].name),
                                           "y_map": SymbolRef(kernel_params[2].name),
                                           "output": SymbolRef(kernel_params[3].name),
                                           "x_dim": Constant(arg_cfg[0].shape[1]),
                                           "y_dim": Constant(arg_cfg[0].shape[0])
                                       })])
        func.params.insert(0, SymbolRef('queue', cl.cl_command_queue()))
        func.params.insert(1, SymbolRef(kernel.body[0].name.name,
                                        cl.cl_kernel()))
        func.defn = control
        proj.files.append(kernel)
        return proj.files

    def finalize(self, files, program_cfg):
        proj = Project(files)
        arg_cfg, tune_cfg = program_cfg
        arg_types, params, kernel_params = self.process_arg_types(arg_cfg)
        arg_types = (cl.cl_command_queue, cl.cl_kernel) + arg_types
        entry_type = (None,) + arg_types
        fn = OclConcreteLerp('lerp', proj, ct.CFUNCTYPE(*entry_type))
        kernel = proj.find(OclFile)
        program = cl.clCreateProgramWithSource(
            fn.context, kernel.codegen()).build()
        return fn.finalize(program[kernel.name])

    def get_placeholder_output(self, args):
        return empty_like(args[-1])

    def get_ir_nodes(self, args):
        tree = copy.deepcopy(self.original_tree)
        arg_cfg = self.args_to_subconfig(args)
        arg_types, params = (), []
        for cfg in arg_cfg:
            arg_types += (np.ctypeslib.ndpointer(cfg.dtype, cfg.ndim, cfg.shape), )
            params.append(SymbolRef.unique(sym_type=arg_types[-1]()))
        shape = arg_cfg[1]
        return [Loop(shape, params[:-1], [params[-1]], arg_types,
                     [StringTemplate(lerp_kern_body, {
                         "input": SymbolRef(params[0].name),
                         "x_map": SymbolRef(params[1].name),
                         "y_map": SymbolRef(params[2].name),
                         "output": SymbolRef(params[3].name),
                         "x_dim": Constant(arg_cfg[0].shape[1]),
                         "y_dim": Constant(arg_cfg[0].shape[0])})])]

interp_linear = LinearInterp(ast.Module())
