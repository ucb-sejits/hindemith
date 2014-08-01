__author__ = 'leonardtruong'

import numpy as np

class TSGemm(object):
    def __call__(self, A, B, C):
        a = np.empty((1, 1, B.shape[0] * A.shape[0]))
        tsgemm_expansion(A, B, a)
        tsgemm_reduction(a, C)
        return C


def reduce_batch(shared, val, local_id, nthreads):
    shared[local_id] = val
    # Barrier
    offset = nthreads >> 1
    while offset > 0:
        if local_id < offset:
            shared[local_id] += shared[local_id + offset]
        # Barrier
        offset >>= 1

def tsgemm_expansion(A, B, a):
    depth1 = A.shape[0]
    depth2 = B.shape[0]

    exp_flt = np.zeros((depth1, depth2), dtype=np.float32)
    a_flt = np.empty(depth1, dtype=np.float32)
    b_flt = np.empty(depth2, dtype=np.float32)

    for x in range(A.shape[0]):
        for y in range(A.shape[1]):
            for i in range(depth1):
                a_flt[i] = A[(x, y, i)]

            for i in range(depth2):
                b_flt[i] = B[(x, y, i)]

            for i in range(depth1):
                for j in range(depth2):
                    exp_flt[i][j] = a_flt[i] * b_flt[j]

    num_threads = 1
    shared = np.empty(num_threads, dtype=np.float32)
    local_id = 0
    for i in range(depth1):
        for j in range(depth2):
            reduce_batch(shared, exp_flt[i][j], local_id, num_threads)
            if local_id == 0:
                a[(0, 0, j + i * depth2)] = shared[0]

def reduce_th(shared, val, local_id, num_threads):
    shared[local_id] = val
    # Barrier
    offset = num_threads >> 1
    while offset > 0:
        if local_id < offset:
            shared[local_id] += shared[local_id + offset]
        # Barrier
        offset >>= 1


def tsgemm_reduction(a, C):
    depth1 = C.shape[0]
    depth2 = C.shape[1]

    num_threads = 1
    shared = np.empty(num_threads)
    local_id = 0
    for i in range(depth1):
        for j in range(depth2):
            res = float(0.0)
            for x in range(a.shape[0]):
                for y in range(a.shape[1]):
                    res += a[(x, y, j + i * depth2)]
            reduce_th(shared, res, local_id, num_threads)
            if local_id == 0:
                C[(j, i)] = shared[0]

