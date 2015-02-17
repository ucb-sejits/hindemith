from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from hindemith.types.hmarray import NdArrCfg, hmarray
from ctree.templates.nodes import StringTemplate
import numpy as np
import pycl as cl
import ctypes as ct
from ctree.c.nodes import CFile, SymbolRef, Constant, Add
from ctree.nodes import Project
from ctree.ocl.nodes import OclFile
from ctree.ocl import get_context_and_queue_from_devices


class OclConcreteReduce(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        self._c_function = self._compile(entry_name, proj, entry_type)
        devices = cl.clGetDeviceIDs()
        self.context, self.queue = get_context_and_queue_from_devices(
            [devices[-1]])

    def finalize(self, kernel):
        self.kernel = kernel
        return self

    def __call__(self, arg):
        out_buf = cl.clCreateBuffer(self.context, arg.nbytes)
        return self._c_function(self.queue, self.kernel, arg.ocl_buf, out_buf)


class Reduction(LazySpecializedFunction):
    backend = 'ocl'

    def args_to_subconfig(self, args):
        return NdArrCfg(args[0].dtype, args[0].ndim, args[0].shape)

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg

        if self.backend == 'c':
            arg_types = (np.ctypeslib.ndpointer(
                arg_cfg.dtype, arg_cfg.ndim, arg_cfg.shape),
                np.ctypeslib.ndpointer(arg_cfg.dtype, arg_cfg.ndim,
                                       arg_cfg.shape))
        else:
            arg_types = (cl.cl_mem, cl.cl_mem)
        kernel = StringTemplate(
            """
__kernel void reduce(__global float *in, __global float *out,
             __local float *buf)
{
  size_t tid = get_local_id(0);
  size_t gid = get_group_id(0);
  size_t dim = get_local_size(0);
  size_t idx = get_global_id(0);
  // Perform the first add with two loads
  // Note we are having workgroups compute two blocks at once
  size_t i = gid * dim * 2 + tid;
  buf[tid] = in[i] $op in[i + dim];
  barrier(CLK_LOCAL_MEM_FENCE);
  // Perform the reduction tree
  for (unsigned int s=dim/2; s > 0; s >>= 1) {
    // Reduce if thread is active for this level
    if (tid < s) {
      buf[tid] $augassign buf[tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // Last thread writes the output
  if (tid == 0) {
    out[gid] = buf[0];
  }
}
            """, {"augassign": SymbolRef(tree.name + "="),
                  "op": tree}
        )

        control = StringTemplate(
            """
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
float control(cl_command_queue queue, cl_kernel reduce, cl_mem input,
            cl_mem output) {
    clSetKernelArg(reduce, 0, sizeof(cl_mem), &input);
    clSetKernelArg(reduce, 1, sizeof(cl_mem), &output);
    clSetKernelArg(reduce, 2, 512 * sizeof($type), NULL);
    size_t local_work_size[1] = {512};
    size_t global_work_size[1];
    global_work_size[0] = $size / 2;
    clEnqueueNDRangeKernel(queue, reduce, 1, NULL, global_work_size,
                           local_work_size, 0, NULL, NULL);
    clSetKernelArg(reduce, 0, sizeof(cl_mem), &output);
    while (global_work_size[0] / local_work_size[0] > 1) {
        global_work_size[0] /= (local_work_size[0] * 2);

        if (local_work_size[0] > global_work_size[0]) {
            local_work_size[0] = global_work_size[0];
        }

        clEnqueueNDRangeKernel(queue, reduce, 1, NULL, global_work_size,
                                local_work_size, 0, NULL, NULL);
    }
    clFinish(queue);
    $type out[1];
    clEnqueueReadBuffer(queue, output, 1, 0, sizeof(float), out, 0, NULL,
            NULL);
    clFinish(queue);
    return out[0];
}
            """, {"type": SymbolRef('float'),
                  "size": Constant(np.prod(arg_cfg.shape))})

        return [CFile('control', [control], config_target='opencl'), 
                OclFile('kernel', [kernel])]
    
    def finalize(self, files, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        if self.backend == 'c':
            arg_types = (np.ctypeslib.ndpointer(
                arg_cfg.dtype, arg_cfg.ndim, arg_cfg.shape),
                np.ctypeslib.ndpointer(arg_cfg.dtype, arg_cfg.ndim,
                                       arg_cfg.shape))
        else:
            arg_types = (cl.cl_mem, cl.cl_mem)
        proj = Project(files)
        if self.backend == 'ocl':
            arg_types = (ct.c_float, cl.cl_command_queue,
                         cl.cl_kernel) + arg_types
            entry_type = ct.CFUNCTYPE(*arg_types)
            fn = OclConcreteReduce('control', proj, entry_type)
            kernel = proj.find(OclFile)
            program = cl.clCreateProgramWithSource(
                fn.context, kernel.codegen()).build()
            return fn.finalize(program['reduce'])


sum = Reduction(SymbolRef("+"))
