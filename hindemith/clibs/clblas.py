import ctypes as ct
import pycl as cl
import os

from hindemith.cl import queues, context
import random

try:
    from sys import platform as _platform
    if _platform == "linux" or _platform == "linux2":
        ext = "so"
    elif _platform == "darwin":  # pragma: no cover
        ext = "dylib"
    path = os.path.dirname(os.path.abspath(__file__))
    _clblaslib = ct.cdll.LoadLibrary(path + "/libclBLAS.{}".format(ext))
except OSError:  # pragma: no cover
    raise Exception("Could not load clBLAS, build it with ./build_clBLAS.sh")

err = _clblaslib.clblasSetup()
if err:  # pragma: no cover
    raise Exception("Error setting up clBLAS: {}".format(err))

_clblaslib.clblasSgemm.restype = ct.c_void_p
_clblaslib.clblasSgemm.argtypes = (
    ct.c_int, ct.c_int, ct.c_int, ct.c_size_t, ct.c_size_t, ct.c_size_t,
    ct.c_float, cl.cl_mem, ct.c_size_t, ct.c_size_t, cl.cl_mem, ct.c_size_t,
    ct.c_size_t, ct.c_float, cl.cl_mem, ct.c_size_t, ct.c_size_t,
    ct.c_size_t, ct.POINTER(cl.cl_command_queue), ct.c_size_t, ct.c_void_p,
    ct.c_void_p
)


def make_event_array(events):
    if isinstance(events, cl.cl_event):
        events = [events]
    valid_events = [e for e in events if e]
    numevents = len(valid_events)
    event_array = (cl.cl_event * numevents)()
    for i, e in enumerate(valid_events):
        event_array[i] = e
    return numevents, event_array


def sgemm(transA, transB, alpha, A, A_offset, lda, B, B_offset, ldb, beta, C,
          C_offset, ldc, m, n, k, _queue=None, wait_for=None):
    if _queue is None:
        _queue = queues[random.randint(0, len(queues) - 1)]
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
    if wait_for is None:
        num_wait = 0
    else:
        num_wait, wait_for = make_event_array(wait_for)
    done_evt = cl.cl_event()
    err = _clblaslib.clblasSgemm(cblas_row_major, transA, transB, m, n, k,
                                 alpha, A.ocl_buf, ct.c_size_t(A_offset),
                                 lda, B.ocl_buf, ct.c_size_t(B_offset), ldb,
                                 beta, C.ocl_buf, ct.c_size_t(C_offset), ldc,
                                 ct.c_size_t(1), ct.byref(_queue),
                                 ct.c_size_t(num_wait), wait_for,
                                 ct.byref(done_evt))
    if err:
        raise Exception("clBLAS sgemm returned error code {}".format(err))
    return done_evt


def sgemv(transA, M, N, alpha, bufA, offA, lda, bufX, offX, incx, beta, bufY,
          offY, incy, wait_for=None):
    cblas_row_major = ct.c_int(0)
    transA = ct.c_int(1 if transA else 0)
    lda = ct.c_size_t(int(lda))
    incx = ct.c_size_t(int(incx))
    incy = ct.c_size_t(int(incy))
    M = ct.c_size_t(int(M))
    N = ct.c_size_t(int(N))
    alpha = ct.c_float(alpha)
    beta = ct.c_float(beta)
    if wait_for is None:
        num_wait = 0
    else:
        num_wait = 1
        wait_for = ct.byref(wait_for)
    done_evt = cl.cl_event()
    err = _clblaslib.clblasSgemv(cblas_row_major, transA, M, N,
                                 alpha, bufA.ocl_buf, ct.c_size_t(offA), lda,
                                 bufX.ocl_buf, ct.c_size_t(offX), incx, beta,
                                 bufY.ocl_buf, ct.c_size_t(offY), incy,
                                 ct.c_size_t(1), ct.byref(queues[0]),
                                 ct.c_size_t(num_wait), wait_for,
                                 ct.byref(done_evt))
    if err:
        raise Exception("clBLAS sgemv returned error code {}".format(err))
    return done_evt


# gemm_kernel = """
# typedef float4 floatX;
# #define WIDTH 4
# #define TSM 128         // The tile-size in dimension M
# #define TSN 128         // The tile-size in dimension N
# #define TSK 16          // The tile-size in dimension K
# #define WPTM 8          // The amount of work-per-thread in dimension M
# #define WPTN 8          // The amount of work-per-thread in dimension N
# #define RTSM (TSM/WPTM) // The reduced tile-size in dimension M
#                         // (== number of threads)
# #define RTSN (TSN/WPTN) // The reduced tile-size in dimension N
#                         // (== number of threads)
# #define LPTA ((TSK*WPTM*WPTN)/(TSN)) // The amount of loads-per-thread for A
# #define LPTB ((TSK*WPTM*WPTN)/(TSM)) // The amount of loads-per-thread for B
#
# #define MOD2(x,y) ((x) % (y))
# #define DIV2(x,y) ((x) / (y))
#
# #define BK TSK
# #define BN TSN
# #define BM TSM
# #define TX RTSM
# #define TY RTSN
# #define RX WPTM
# #define RY WPTN
#
# #define PADDINGX 16
# #define PADDINGY 16
#
# __kernel void sgemm(const int M, const int N, const int K, const float alpha,
#                     const __global floatX* A,
#                     const __global floatX* B,
#                     __global float* C) {
#
#     // Thread identifiers
#     const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
#     const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
#     const int gidm = get_group_id(0); // Work-group ID
#     const int gidn = get_group_id(1); // Work-group ID
#     const int tid = tidn*RTSM + tidm; // Global thread ID (max RTSM*RTSN)
#
#     // Local memory to fit two tiles of A and B
#     __local float Asub[2][TSK*TSM];
#     __local float Bsub[2][TSK*TSN];
#
#     // Allocate register space
#     float Areg;
#     float Breg[WPTN];
#     float acc[WPTM][WPTN];
#
#     // Initialise the accumulation registers
#     #pragma unroll
#     for (int wm=0; wm<WPTM; wm++) {
#         #pragma unroll
#         for (int wn=0; wn<WPTN; wn++) {
#             acc[wm][wn] = 0.0f;
#         }
#     }
#
#     // Tile A
#     #pragma unroll
#     for (int la=0; la<LPTA/WIDTH; la++) {
#         int id = la*RTSN*RTSM + tid;
#         int row = MOD2(id,TSM/WIDTH);
#         int col = DIV2(id,TSM/WIDTH);
#
#         // Load the value (wide vector load)
#         int tiledIndex = TSK*0 + col;
#         int indexA = tiledIndex*(M/WIDTH) + gidm*(TSM/WIDTH) + row;
#         #ifdef USE_LDG
#             floatX vecA = __ldg(&A[indexA]);
#         #else
#             floatX vecA = A[indexA];
#         #endif
#
#         // Store the loaded vector into local memory
#         #if WIDTH == 1
#             Asub[0][col*TSM + row] = vecA;
#         #elif WIDTH == 2
#             Asub[0][col*TSM + WIDTH*row + 0] = vecA.x;
#             Asub[0][col*TSM + WIDTH*row + 1] = vecA.y;
#         #elif WIDTH == 4
#             Asub[0][col*TSM + WIDTH*row + 0] = vecA.x;
#             Asub[0][col*TSM + WIDTH*row + 1] = vecA.y;
#             Asub[0][col*TSM + WIDTH*row + 2] = vecA.z;
#             Asub[0][col*TSM + WIDTH*row + 3] = vecA.w;
#         #endif
#     }
#
#     // Tile B
#     #pragma unroll
#     for (int lb=0; lb<LPTB/WIDTH; lb++) {
#         int id = lb*RTSN*RTSM + tid;
#         int row = MOD2(id,TSN/WIDTH);
#         int col = DIV2(id,TSN/WIDTH);
#
#         // Load the value (wide vector load)
#         int tiledIndex = TSK*0 + col;
#         int indexB = tiledIndex*(N/WIDTH) + gidn*(TSN/WIDTH) + row;
#         #ifdef USE_LDG
#             floatX vecB = __ldg(&B[indexB]);
#         #else
#             floatX vecB = B[indexB];
#         #endif
#
#         // Store the loaded vector into local memory
#         #if WIDTH == 1
#             Bsub[0][col*TSN + row] = vecB;
#         #elif WIDTH == 2
#             Bsub[0][col*TSN + WIDTH*row + 0] = vecB.x;
#             Bsub[0][col*TSN + WIDTH*row + 1] = vecB.y;
#         #elif WIDTH == 4
#             Bsub[0][col*TSN + WIDTH*row + 0] = vecB.x;
#             Bsub[0][col*TSN + WIDTH*row + 1] = vecB.y;
#             Bsub[0][col*TSN + WIDTH*row + 2] = vecB.z;
#             Bsub[0][col*TSN + WIDTH*row + 3] = vecB.w;
#         #endif
#     }
#
#     // Loop over all tiles
#     const int numTiles = K/TSK;
#     int t=0;
#     do {
#
#         // Synchronise
#         barrier(CLK_LOCAL_MEM_FENCE);
#
#         // Load the next tile of A and B into local memory
#         int tt = t + 1;
#         if (tt < numTiles) {
#
#             // Tile A
#             #pragma unroll
#             for (int la=0; la<LPTA/WIDTH; la++) {
#                 int id = la*RTSN*RTSM + tid;
#                 int row = MOD2(id,TSM/WIDTH);
#                 int col = DIV2(id,TSM/WIDTH);
#
#                 // Load the value (wide vector load)
#                 int tiledIndex = TSK*tt + col;
#                 int indexA = tiledIndex*(M/WIDTH) + gidm*(TSM/WIDTH) + row;
#                 #ifdef USE_LDG
#                     floatX vecA = __ldg(&A[indexA]);
#                 #else
#                     floatX vecA = A[indexA];
#                 #endif
#
#                 // Store the loaded vector into local memory
#                 #if WIDTH == 1
#                     Asub[tt%2][col*TSM + row] = vecA;
#                 #elif WIDTH == 2
#                     Asub[tt%2][col*TSM + WIDTH*row + 0] = vecA.x;
#                     Asub[tt%2][col*TSM + WIDTH*row + 1] = vecA.y;
#                 #elif WIDTH == 4
#                     Asub[tt%2][col*TSM + WIDTH*row + 0] = vecA.x;
#                     Asub[tt%2][col*TSM + WIDTH*row + 1] = vecA.y;
#                     Asub[tt%2][col*TSM + WIDTH*row + 2] = vecA.z;
#                     Asub[tt%2][col*TSM + WIDTH*row + 3] = vecA.w;
#                 #endif
#             }
#
#             // Tile B
#             #pragma unroll
#             for (int lb=0; lb<LPTB/WIDTH; lb++) {
#                 int id = lb*RTSN*RTSM + tid;
#                 int row = MOD2(id,TSN/WIDTH);
#                 int col = DIV2(id,TSN/WIDTH);
#
#                 // Load the value (wide vector load)
#                 int tiledIndex = TSK*tt + col;
#                 int indexB = tiledIndex*(N/WIDTH) + gidn*(TSN/WIDTH) + row;
#                 #ifdef USE_LDG
#                     floatX vecB = __ldg(&B[indexB]);
#                 #else
#                     floatX vecB = B[indexB];
#                 #endif
#
#                 // Store the loaded vector into local memory
#                 #if WIDTH == 1
#                     Bsub[tt%2][col*TSN + row] = vecB;
#                 #elif WIDTH == 2
#                     Bsub[tt%2][col*TSN + WIDTH*row + 0] = vecB.x;
#                     Bsub[tt%2][col*TSN + WIDTH*row + 1] = vecB.y;
#                 #elif WIDTH == 4
#                     Bsub[tt%2][col*TSN + WIDTH*row + 0] = vecB.x;
#                     Bsub[tt%2][col*TSN + WIDTH*row + 1] = vecB.y;
#                     Bsub[tt%2][col*TSN + WIDTH*row + 2] = vecB.z;
#                     Bsub[tt%2][col*TSN + WIDTH*row + 3] = vecB.w;
#                 #endif
#             }
#         }
#
#         // Loop over the values of a single tile
#         #pragma unroll
#         for (int k=0; k<TSK; k++) {
#
#             // Cache the values of Bsub in registers
#             #pragma unroll
#             for (int wn=0; wn<WPTN; wn++) {
#                 int col = tidn + wn*RTSN;
#                 Breg[wn] = Bsub[t%2][k*TSN + col];
#             }
#
#             // Perform the computation
#             #pragma unroll
#             for (int wm=0; wm<WPTM; wm++) {
#                 int row = tidm + wm*RTSM;
#                 Areg = Asub[t%2][k*TSM + row];
#                 #pragma unroll
#                 for (int wn=0; wn<WPTN; wn++) {
#                     acc[wm][wn] += Areg * Breg[wn];
#                 }
#             }
#         }
#
#         // Next tile
#         t++;
#     } while (t<numTiles);
#
#     // Store the final results in C
#     #pragma unroll
#     for (int wm=0; wm<WPTM; wm++) {
#         int globalRow = gidm*TSM + tidm + wm*RTSM;
#         #pragma unroll
#         for (int wn=0; wn<WPTN; wn++) {
#             int globalCol = gidn*TSN + tidn + wn*RTSN;
#             C[globalCol*M + globalRow] = alpha * acc[wm][wn];
#         }
#     }
# }
# #define TRANSPOSEX 16
# #define TRANSPOSEY 16
# __kernel void transpose(const int P, const int Q,
#                         const __global float* input, const int offset,
#                         __global float* output) {
#     // Thread identifiers
#     const int tx = get_local_id(0);
#     const int ty = get_local_id(1);
#     const int ID0 = get_group_id(0)*TRANSPOSEX + tx; // 0..P
#     const int ID1 = get_group_id(1)*TRANSPOSEY + ty; // 0..Q
#
#     // Set-up the local memory for shuffling
#     __local float buffer[TRANSPOSEX][TRANSPOSEY];
#
#     // Swap the x and y coordinates to perform the rotation (coalesced)
#     if (ID0 < P && ID1 < Q) {
#         buffer[ty][tx] = input[offset + ID1*P + ID0];
#     }
#
#     // Synchronise all threads
#     barrier(CLK_LOCAL_MEM_FENCE);
#
#     // We don't have to swap the x and y thread indices here,
#     // because that's already done in the local memory
#     const int newID0 = get_group_id(1)*TRANSPOSEY + tx;
#     const int newID1 = get_group_id(0)*TRANSPOSEX + ty;
#
#     // Store the transposed result (coalesced)
#     if (newID0 < Q && newID1 < P) {
#         output[newID1*Q + newID0] = buffer[tx][ty];
#     }
# }
#
# // Pad the P * Q matrix with zeroes to form a P_XL * Q_XL matrix
# __kernel void paddingAddZeroes(const int P, const int Q,
#                                const __global float* input, const int offset,
#                                const int P_XL, const int Q_XL,
#                                __global float* output) {
#
#     // Thread identifiers
#     const int tx = get_group_id(0)*PADDINGX + get_local_id(0);
#     const int ty = get_group_id(1)*PADDINGY + get_local_id(1);
#
#     // Check whether we are within bounds of the XL matrix
#     if (tx < P_XL && ty < Q_XL) {
#
#         // Copy the input or pad a zero
#         float value;
#         if (tx < P && ty < Q) {
#             value = input[offset + ty*P + tx];
#         }
#         else {
#             value = 0.0f;
#         }
#
#         // Store the result
#         output[ty*P_XL + tx] = value;
#     }
# }
#
#
# // Remove padded values from a P_XL * Q_XL matrix to form a P * Q matrix
# __kernel void paddingRemoveZeroes(const int P_XL, const int Q_XL,
#                                   const float beta,
#                                   const __global float* input,
#                                   const int P, const int Q,
#                                   __global float* output, const int offset) {
#     // Thread identifiers
#     // 0..P in blocks of PADDINGX
#     const int tx = get_group_id(0)*PADDINGX + get_local_id(0);
#     // 0..Q in blocks of PADDINGY
#     const int ty = get_group_id(1)*PADDINGY + get_local_id(1);
#
#
#     // Only store the result if within P * Q bounds
#     if (tx < P && ty < Q) {
#         output[offset + ty*P + tx] = beta * output[offset + ty*P + tx] + \
#             input[ty*P_XL + tx];
#     }
# }
# """
#
# program = cl.clCreateProgramWithSource(context, gemm_kernel).build()
# transpose = program["transpose"]
# transpose.argtypes = (cl.cl_int, cl.cl_int, cl.cl_mem, cl.cl_int, cl.cl_mem)
# sgemm_kernel = program["sgemm"]
# sgemm_kernel.argtypes = (cl.cl_int, cl.cl_int, cl.cl_int, cl.cl_float,
#                          cl.cl_mem, cl.cl_mem, cl.cl_mem)
# padding_add_zeros = program["paddingAddZeroes"]
# padding_add_zeros.argtypes = (cl.cl_int, cl.cl_int, cl.cl_mem, cl.cl_int,
#                               cl.cl_int, cl.cl_int, cl.cl_mem)
# padding_remove_zeros = program["paddingRemoveZeroes"]
# padding_remove_zeros.argtypes = (cl.cl_int, cl.cl_int, cl.cl_float, cl.cl_mem,
#                                  cl.cl_int, cl.cl_int, cl.cl_mem, cl.cl_int)
#
#
# def ceil_div(x, y):
#     return (x + y - 1) / y


# def sgemm(transA, transB, alpha, A, A_offset, lda, B, B_offset, ldb, beta, C,
#           C_offset, ldc, m, n, k, _queue=None, wait_for=None):
#     # As if column major
#     A, A_offset, B, B_offset = B, B_offset, A, A_offset
#     m, n = n, m
#     if _queue is None:
#         _queue = queues[random.randint(0, len(queues) - 1)]
#     tsm = 128
#     tsn = 128
#     tsk = 16
#     k_xl = ceil_div(k, tsk) * tsk
#     m_xl = ceil_div(m, tsm) * tsm
#     n_xl = ceil_div(n, tsn) * tsn
#
#     B_tr = cl.clCreateBuffer(context, n * k * cl.sizeof(cl.cl_float))
#     A_xl = cl.clCreateBuffer(context, m_xl * k_xl * cl.sizeof(cl.cl_float))
#     B_xl = cl.clCreateBuffer(context, n_xl * k_xl * cl.sizeof(cl.cl_float))
#     C_xl = cl.clCreateBuffer(context, m_xl * n_xl * cl.sizeof(cl.cl_float))
#     evt1 = transpose(k, n, B.ocl_buf, B_offset, B_tr).on(
#         _queue, ((k + 15) & ~15, (n + 15) & ~15), (16, 16),
#         wait_for=wait_for)
#     evt2 = padding_add_zeros(m, k, A.ocl_buf, A_offset, m_xl, k_xl, A_xl).on(
#         _queue, (m_xl, k_xl), (16, 16), wait_for=wait_for)
#     evt3 = padding_add_zeros(n, k, B_tr, 0, n_xl, k_xl, B_xl).on(
#         _queue, (n_xl, k_xl), (16, 16), wait_for=wait_for)
#     tsm, tsn = 128, 128
#     wptm, wptn = 8, 8
#     evt = sgemm_kernel(m_xl, n_xl, k_xl, alpha, A_xl, B_xl, C_xl).on(
#         _queue, (m_xl/wptm, n_xl/wptn), (tsm/wptm, tsn/wptn),
#         wait_for=[evt1, evt2, evt3])
#     evt = padding_remove_zeros(
#         m_xl, n_xl, beta, C_xl, m, n, C.ocl_buf, C_offset).on(
#             _queue, ((m + 15) & ~15, (n + 15) & ~15), (16, 16), wait_for=evt)
#     return evt
