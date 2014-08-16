from hindemith.fusion.core import fuse
from hindemith.operations.dense_linear_algebra.core import array_mul, \
    array_sub, array_add

import numpy

from ctree.util import Timer


x = []
iterations = 20
results = [[] for _ in range(3)]

for width in (2**x for x in range(8, 13)):
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

    def numpy_f(A, B, C):
        D = A * B
        E = C - D
        F = A + B
        G = F - E
        H = F * G
        return H

    A = numpy.random.rand(width, width).astype(numpy.float32)
    B = numpy.random.rand(width, width).astype(numpy.float32)
    C = numpy.random.rand(width, width).astype(numpy.float32)

    fused_f(A, B, C)
    unfused_f(A, B, C)

    for _ in range(iterations):
        x.append(width)
        with Timer() as fused_time:
            a = fused_f(A, B, C)
        results[0].append(fused_time.interval)

        unfused_f(A, B, C)
        with Timer() as unfused_time:
            unfused_f(A, B, C)
        results[1].append(unfused_time.interval)

        with Timer() as numpy_time:
            c = numpy_f(A, B, C)
        results[2].append(numpy_time.interval)

        numpy.testing.assert_array_almost_equal(a, c)

colors = ['b', 'c', 'r']
import matplotlib.pyplot as plt

r1 = plt.scatter(x, results[0], marker='x', color=colors[0])
r2 = plt.scatter(x, results[1], marker='x', color=colors[1])
r3 = plt.scatter(x, results[2], marker='x', color=colors[2])

plt.legend((r1, r2, r3),
           ('Fused', 'Unfused', 'Numpy'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.show()
