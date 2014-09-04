from hindemith.fusion.core import fuse
from hindemith.operations.dense_linear_algebra.core import scalar_array_mul

from stencil_code.stencil_kernel import StencilKernel
import numpy

from ctree.util import Timer


radius = 1


class Stencil(StencilKernel):
    @property
    def dim(self):
        return 2

    @property
    def ghost_depth(self):
        return 1

    def neighbors(self, pt, defn=0):
        if defn == 0:
            for x in range(-radius, radius+1):
                for y in range(-radius, radius+1):
                    yield (pt[0] - x, pt[1] - y)

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]


x = []
iterations = 10
results = [[] for _ in range(3)]

# for width in (2**x for x in range(8, 10)):
for width in range(2**8, 2**13, 512):
    stencil1 = Stencil(backend='ocl')
    stencil2 = Stencil(backend='ocl')

    @fuse
    def fused_f(A, B):
        C = stencil1(A)
        D = scalar_array_mul(3, C)
        return D

    def unfused_f(A, B):
        C = stencil2(A)
        D = scalar_array_mul(3, C)
        return D

    A = numpy.random.rand(width, width).astype(numpy.float32)
    B = numpy.random.rand(width, width).astype(numpy.float32)
    C = numpy.random.rand(width, width).astype(numpy.float32)

    fused_f(A, B)
    unfused_f(A, B)

    x.append(width)
    fused_total = 0
    unfused_total = 0
    for _ in range(iterations):
        with Timer() as fused_time:
            a = fused_f(A, B)
        fused_total += fused_time.interval

        unfused_f(A, B)
        with Timer() as unfused_time:
            b = unfused_f(A, B)
        unfused_total += unfused_time.interval

    results[0].append(fused_total/iterations)
    results[1].append(unfused_total/iterations)

colors = ['b', 'c', 'r']
import matplotlib.pyplot as plt

# r1 = plt.scatter(x, results[0], marker='x', color=colors[0])
# r2 = plt.scatter(x, results[1], marker='x', color=colors[1])
width = .35
fig, ax = plt.subplots()
rects1 = ax.bar([index for index, _ in enumerate(x)], results[0], width, color='b')

rects2 = ax.bar([index + width for index, _ in enumerate(x)], results[1], width, color='g')

ax.set_ylabel('Times (s)')
ax.set_title('Array Width')
ax.set_xticks([index + width for index, _ in enumerate(x)])
ax.set_xticklabels(x)

ax.legend((rects1[0], rects2[0]), ('Fused', 'Unfused'))

# plt.legend((r1, r2),
#            ('Fused', 'Unfused'),
#            scatterpoints=1,
#            loc='lower left',
#            ncol=3,
#            fontsize=8)
plt.show()
