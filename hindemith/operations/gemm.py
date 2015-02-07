"""
kernel void gemm_nn (
    global const T * restrict A,
    int lda,    // column stride in elements for matrix A
    global const T * restrict B,
    int ldb,    // column stride in elements for matrix B
    global T * restrict C,
    int ldc,    // column stride in elements for matrix C
    int k,
    T alpha,
    T beta
)
"""
gemm_kernel = """
    // Indices for matrices A and B are calculated differently
    // because they have the same format (both column-major) and
    // matrix multiplication involves "natural transpose" for
    // one of the matrix.

    int Aind = get_group_id(0)*$TILE_GROUP_M*$TILE_SIZE_M + get_local_id(0);
    int Bind = get_group_id(1)*$TILE_GROUP_N*$TILE_SIZE_N + get_local_id(1);
    int Cind = Aind + Bind*$ldc;

    Bind *= $ldb;    // matrix B is in column-major form

    $T c[$TILE_SIZE_M*$TILE_SIZE_N] = {($T)0};

    // Main accumulation loop
    for(int l_block = 0; l_block < $k; l_block += $TILE_SIZE_K)
    {
        for(int i = 0; i < $TILE_SIZE_M; ++i)
            for(int j = 0; j < $TILE_SIZE_N; ++j)
                for(int l = 0; l < $TILE_SIZE_K; ++l)
                    c[i*$TILE_SIZE_N + j] +=
                        A[Aind + l*$lda + i*$TILE_GROUP_M] *
                        B[Bind + l + j*$ldb*$TILE_GROUP_N];
        Aind += $lda*$TILE_SIZE_K;
        Bind += $TILE_SIZE_K;
    }

    // Store accumulated results from c to C with alpha and beta multiplication
    for(int i = 0; i < $TILE_SIZE_M; ++i)
        for(int j = 0; j < $TILE_SIZE_N; ++j)
        {
            int Ccur = Cind + i*$TILE_GROUP_M + j*$TILE_GROUP_N*$ldc;
            C[Ccur] = ((float) $alpha)*c[i*$TILE_SIZE_N + j] + ((float) $beta)*C[Ccur];
        }
"""
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.templates.nodes import StringTemplate
from ctree.c.nodes import Constant, SymbolRef, FunctionDecl, CFile
from ctree.nodes import Project
from ctree.ocl import get_context_and_queue_from_devices
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
        tile_size_m = 1
        tile_group_m = 16
        tile_size_n = min(128, n)
        tile_group_n = 1
        tile_size_k = 8
        global_size = (m / tile_size_m, n / tile_size_n)
        local_size = (min(global_size[0], tile_group_m),
                      min(global_size[1], tile_group_n))
        loop_body = [StringTemplate(
            gemm_kernel,
            {'TILE_GROUP_M': Constant(tile_group_m),
             'TILE_GROUP_N': Constant(tile_group_n),
             'TILE_SIZE_M': Constant(tile_size_m),
             'TILE_SIZE_N': Constant(tile_size_n),
             'TILE_SIZE_K': Constant(tile_size_k),
             'ldc': Constant(m),
             'ldb': Constant(arg_cfg['B'][0][0]),
             'lda': Constant(arg_cfg['A'][0][0]),
             'k': Constant(arg_cfg['A'][0][1]),
             'T': StringTemplate('float'),
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
                                       loop_body, local_size=local_size)

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
        print(kernel)
        print(func)
        entry_type = (None, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem,
                      cl.cl_mem, cl.cl_mem)
        proj = Project([CFile('gemm', [func])])
        proj.files[0].body.insert(0, StringTemplate("""
            #ifdef __APPLE__
            #include <OpenCL/opencl.h>
            #else
            #include <CL/cl.h>
            #endif
            """))

        entry_type = ct.CFUNCTYPE(*entry_type)
        fn = ConcreteGemm('gemm', proj, entry_type)
        program = cl.clCreateProgramWithSource(
            fn.context, kernel.codegen()).build()
        return fn.finalize(program[kernel.body[0].name.name])


gemm = Gemm(None)
