from hindemith.fusion.core import fuse
from hindemith.operations.dense_linear_algebra.core import array_mul, \
    array_sub, array_add

import numpy
from numpy import testing
# import cProfile

from ctree.util import Timer

# import logging
# logging.basicConfig(level=20)

A = numpy.random.rand(2**10, 2**10).astype(numpy.float32)
B = numpy.random.rand(2**10, 2**10).astype(numpy.float32)
C = numpy.random.rand(2**10, 2**10).astype(numpy.float32)


@fuse
def fused_f(A, B, C):
    D = array_mul(A, B)
    E = array_sub(C, D)
    F = array_add(A, B)
    G = array_sub(F, E)
    H = array_mul(F, G)
    return H


def unfused_f(A, B, C):
    D = array_mul(A, B)
    E = array_sub(C, D)
    F = array_add(A, B)
    G = array_sub(F, E)
    H = array_mul(F, G)
    return H


def run():
    fused_f(A, B, C)
    unfused_f(A, B, C)
    with Timer() as fused_time:
        fused_result = fused_f(A, B, C)
    print("Fused time: {}".format(fused_time.interval))

    unfused_f(A, B, C)
    with Timer() as unfused_time:
        unfused_result = unfused_f(A, B, C)
    print("Unfused time: {}".format(unfused_time.interval))

    testing.assert_array_almost_equal(fused_result, unfused_result)
    print("PASSED!")

run()
# cProfile.run('run()')
