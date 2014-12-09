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
iterations = 1
results = [[] for _ in range(3)]

# for width in (2**x for x in range(8, 12)):
for width in range(256, 4096, 256):
    print("width is now {}".format(width))
    for _ in range(iterations):
        stencil1 = Stencil(backend='ocl')
        stencil2 = Stencil(backend='ocl')
        stencil3 = Stencil(backend='ocl')
        stencil4 = Stencil(backend='ocl')
        stencil5 = Stencil(backend='ocl')
        stencil6 = Stencil(backend='ocl')

        @fuse
        def fused_f(A):
            C = stencil1(A)
            D = stencil2(C)
            return stencil3(D)

        def unfused_f(A):
            return stencil5(stencil5(stencil5(A)))

        A = numpy.random.rand(width, width).astype(numpy.float32) * 100

        a = fused_f(A)
        b = unfused_f(A)
        # numpy.testing.assert_array_almost_equal(a[2:-2, 2:-2], b[2:-2, 2:-2])
        # numpy.testing.assert_array_almost_equal(a[3:-3, 3:-3], b[3:-3, 3:-3], decimal=4)
        x.append(width)
        with Timer() as fused_time:
            fused_f(A)
        results[0].append(fused_time.interval)

        # with Timer() as unfused_time:
        #     unfused_f(A)
        # results[1].append(unfused_time.interval)


colors = ['b', 'c', 'r']
import matplotlib.pyplot as plt

r1 = plt.scatter(x, results[0], marker='x', color=colors[0])
r2 = plt.scatter(x, results[0], marker='x', color=colors[1])

plt.legend((r1, r2),
           ('Fused', 'Unfused'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.show()
