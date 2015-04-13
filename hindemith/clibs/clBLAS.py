import ctypes as ct
import pycl as cl
import os

from hindemith.cl import queue

try:
    from sys import platform as _platform
    if _platform == "linux" or _platform == "linux2":
        ext = "so"
    elif _platform == "darwin":
        ext = "dylib"
    path = os.path.dirname(os.path.abspath(__file__))
    _clblaslib = ct.cdll.LoadLibrary(path + "/libclBLAS.{}".format(ext))
except OSError:
    raise Exception("Could not load clBLAS, build it with ./build_clBLAS.sh")

err = _clblaslib.clblasSetup()
if err:
    raise Exception("Error setting up clBLAS: {}".format(err))

_clblaslib.clblasSgemm.restype = ct.c_void_p
_clblaslib.clblasSgemm.argtypes = (
    ct.c_int, ct.c_int, ct.c_int, ct.c_size_t, ct.c_size_t, ct.c_size_t,
    ct.c_float, cl.cl_mem, ct.c_size_t, ct.c_size_t, cl.cl_mem, ct.c_size_t,
    ct.c_size_t, ct.c_float, cl.cl_mem, ct.c_size_t, ct.c_size_t, ct.c_size_t,
    ct.POINTER(cl.cl_command_queue), ct.c_size_t, ct.c_void_p, ct.c_void_p
)


def sgemm(transA, transB, alpha, A, A_offset, lda, B, B_offset, ldb, beta, C,
          C_offset, ldc, m, n, k):
    cblas_row_major = ct.c_int(0)
    transA = ct.c_int(1 if transA else 0)
    transB = ct.c_int(1 if transB else 0)
    lda = ct.c_size_t(int(lda))
    ldb = ct.c_size_t(int(ldb))
    ldc = ct.c_size_t(int(ldc))
    m = ct.c_size_t(int(m))
    n = ct.c_size_t(int(n))
    k = ct.c_size_t(int(k))
    alpha = ct.c_float(alpha)
    beta = ct.c_float(beta)
    err = _clblaslib.clblasSgemm(cblas_row_major, transA, transB, m, n, k,
                                 alpha, A.ocl_buf, ct.c_size_t(A_offset), lda,
                                 B.ocl_buf, ct.c_size_t(B_offset), ldb, beta,
                                 C.ocl_buf, ct.c_size_t(C_offset), ldc,
                                 ct.c_size_t(1), ct.byref(queue),
                                 ct.c_size_t(0), None, None)
    if err:
        raise Exception("clBLAS sgemm returned error code {}".format(err))
