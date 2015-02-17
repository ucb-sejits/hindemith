gemm_kernel = """
   int x = get_global_id(1); 
   int y = get_global_id(0);
 
   $dtype value = 0;
   for (int k = 0; k < $K; ++k) {
      value += A[y * $K + k] * B[k * $N + x];
   }
 
   C[y * $N + x] = $alpha * value + $beta * C[y * $N + x];
"""

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.templates.nodes import StringTemplate
from ctree.c.nodes import Constant, SymbolRef, FunctionDecl, CFile
from ctree.nodes import Project
from ctree.ocl.nodes import Project, OclFile
from ctree.ocl import get_context_and_queue_from_devices
import ast
import numpy as np
import pycl as cl
import ctypes as ct
from hindemith.nodes import kernel_range


class ConcreteGemm(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        self._c_function = self._compile(entry_name, proj, entry_type)
        devices = cl.clGetDeviceIDs()
        self.context, self.queue = get_context_and_queue_from_devices(
            [devices[-1]])

    def finalize(self, kernel):
        self.kernel = kernel
        return self

    def __call__(self, A, B, C, alpha, beta):
        self._c_function(self.queue, self.kernel, A.ocl_buf, B.ocl_buf, C.ocl_buf)
        C._host_dirty = True
        return C


class Gemm(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        A, B, C, alpha, beta = args
        return {
            'A': (A.shape, A.dtype),
            'B': (B.shape, B.dtype),
            'C': (C.shape, C.dtype),
            'alpha': alpha,
            'beta': beta
        }

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        C = arg_cfg['C']
        shape = C[0]
        m = shape[0]
        n = shape[1]
        global_size = (m, n)
        loop_body = [StringTemplate(
            gemm_kernel,
            {'K': Constant(arg_cfg['A'][0][1]),
             'M': Constant(m),
             'N': Constant(n),
             'dtype': StringTemplate('float'),
             'alpha': Constant(arg_cfg['alpha']),
             'beta': Constant(arg_cfg['beta'])
             })]
        kernel_params = (
            SymbolRef('A',
                      np.ctypeslib.ndpointer(arg_cfg['A'][1],
                                             len(arg_cfg['A'][0]),
                                             arg_cfg['A'][0])()),
            SymbolRef('B',
                      np.ctypeslib.ndpointer(arg_cfg['B'][1],
                                             len(arg_cfg['B'][0]),
                                             arg_cfg['B'][0])()),
            SymbolRef('C',
                      np.ctypeslib.ndpointer(arg_cfg['C'][1],
                                             len(arg_cfg['C'][0]),
                                             arg_cfg['C'][0])()),
        )
        control, kernel = kernel_range(global_size, global_size, kernel_params,
                                       loop_body)

        params = [
            SymbolRef('queue', cl.cl_command_queue()),
            SymbolRef(kernel.body[0].name.name, cl.cl_kernel()),
            SymbolRef('A', cl.cl_mem()),
            SymbolRef('B', cl.cl_mem()),
            SymbolRef('C', cl.cl_mem()),
        ]
        func = FunctionDecl(
            None,
            SymbolRef('gemm'),
            params,
            control
        )
        entry_type = (None, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem,
                      cl.cl_mem, cl.cl_mem)
        proj = Project([CFile('gemm', [func], config_target='opencl'), kernel])
        proj.files[0].body.insert(0, StringTemplate("""
            #ifdef __APPLE__
            #include <OpenCL/opencl.h>
            #else
            #include <CL/cl.h>
            #endif
            """))
        return proj.files

    def finalize(self, files, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        proj = Project(files)
        entry_type = (None, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem,
                      cl.cl_mem, cl.cl_mem)
        entry_type = ct.CFUNCTYPE(*entry_type)
        fn = ConcreteGemm('gemm', proj, entry_type)
        kernel = proj.find(OclFile)
        program = cl.clCreateProgramWithSource(
            fn.context, kernel.codegen()).build()
        return fn.finalize(program[kernel.name])


gemm = Gemm(ast.Module())
