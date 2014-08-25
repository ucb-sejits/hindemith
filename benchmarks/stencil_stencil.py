from hindemith.fusion.core import fuse

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
iterations = 2
results = [[] for _ in range(3)]
speedup = [[] for _ in range(4)]


for width in range(2**8, 2**13, 256):
    print("Running width: %d" % width)
# for width in (2**x for x in range(8, 11)):
    total0, total1, total2 = 0, 0, 0
    for _ in range(iterations):
        stencil1 = Stencil(backend='ocl')
        stencil2 = Stencil(backend='ocl')
        stencil3 = Stencil(backend='ocl')

        @fuse
        def fused_f(A):
            C = stencil1(A)
            return stencil2(C)

        def unfused_f(A):
            return stencil3(stencil3(A))

        A = numpy.random.rand(width, width).astype(numpy.float32) * 100

        a = fused_f(A)
        b = unfused_f(A)
        numpy.testing.assert_array_almost_equal(a[2:-2, 2:-2], b[2:-2, 2:-2])
        # x.append(width)
        with Timer() as fused_time:
            fused_f(A)
        results[0].append(fused_time.interval)
        total0 += fused_time.interval

        with Timer() as fused_time2:
            fused_f(A)
        results[1].append(fused_time2.interval)
        total1 += fused_time.interval

        with Timer() as unfused_time:
            unfused_f(A)
        results[1].append(unfused_time.interval)
        total1 += unfused_time.interval
    total0 /= iterations
    total1 /= iterations
    speedup[0].append(total1/total0)
    # x.append(width)

    for _ in range(iterations):
        stencil1 = Stencil(backend='ocl')
        stencil2 = Stencil(backend='ocl')
        stencil3 = Stencil(backend='ocl')
        stencil4 = Stencil(backend='ocl')

        @fuse
        def fused_f(A):
            B = stencil1(A)
            C = stencil2(B)
            return stencil3(C)

        def unfused_f(A):
            return stencil4(stencil4(stencil4(A)))

        A = numpy.random.rand(width, width).astype(numpy.float32) * 100

        a = fused_f(A)
        b = unfused_f(A)
        numpy.testing.assert_array_almost_equal(a[2:-2, 2:-2], b[2:-2, 2:-2])
        # x.append(width)
        with Timer() as fused_time:
            fused_f(A)
        results[0].append(fused_time.interval)
        total0 += fused_time.interval

        with Timer() as unfused_time:
            unfused_f(A)
        results[1].append(unfused_time.interval)
        total1 += unfused_time.interval
    total0 /= iterations
    total1 /= iterations
    speedup[1].append(total1/total0)
    x.append(width)

    total0, total1 = 0, 0
    for _ in range(iterations):
        stencil1 = Stencil(backend='ocl')
        stencil2 = Stencil(backend='ocl')
        stencil3 = Stencil(backend='ocl')
        stencil4 = Stencil(backend='ocl')
        stencil5 = Stencil(backend='ocl')

        @fuse
        def fused_f(A):
            B = stencil1(A)
            C = stencil2(B)
            D = stencil3(C)
            return stencil4(D)

        def unfused_f(A):
            return stencil5(stencil5(stencil5(stencil5(A))))

        A = numpy.random.rand(width, width).astype(numpy.float32) * 100

        a = fused_f(A)
        b = unfused_f(A)
        numpy.testing.assert_array_almost_equal(a[4:-4, 4:-4], b[4:-4, 4:-4])
        # x.append(width)
        with Timer() as fused_time:
            fused_f(A)
        results[0].append(fused_time.interval)
        total0 += fused_time.interval

        with Timer() as unfused_time:
            unfused_f(A)
        results[1].append(unfused_time.interval)
        total1 += unfused_time.interval
    total0 /= iterations
    total1 /= iterations
    speedup[2].append(total1/total0)
    # x.append(width)
    total0, total1 = 0, 0
    for _ in range(iterations):
        stencil1 = Stencil(backend='ocl')
        stencil2 = Stencil(backend='ocl')
        stencil3 = Stencil(backend='ocl')
        stencil4 = Stencil(backend='ocl')
        stencil5 = Stencil(backend='ocl')
        stencil6 = Stencil(backend='ocl')

        @fuse
        def fused_f(A):
            B = stencil1(A)
            C = stencil2(B)
            D = stencil3(C)
            E = stencil4(D)
            return stencil5(E)

        def unfused_f(A):
            return stencil6(stencil6(stencil6(stencil6(stencil6(A)))))

        A = numpy.random.rand(width, width).astype(numpy.float32) * 100

        a = fused_f(A)
        b = unfused_f(A)
        numpy.testing.assert_array_almost_equal(a[4:-4, 4:-4], b[4:-4, 4:-4])
        # x.append(width)
        with Timer() as fused_time:
            fused_f(A)
        results[0].append(fused_time.interval)
        total0 += fused_time.interval

        with Timer() as unfused_time:
            unfused_f(A)
        results[1].append(unfused_time.interval)
        total1 += unfused_time.interval
    total0 /= iterations
    total1 /= iterations
    speedup[3].append(total1/total0)


colors = ['b', 'c', 'r', 'g']
import matplotlib.pyplot as plt

r1 = plt.scatter(x, speedup[0], marker='x', color=colors[0])
r2 = plt.scatter(x, speedup[1], marker='x', color=colors[1])
r3 = plt.scatter(x, speedup[2], marker='x', color=colors[2])
r4 = plt.scatter(x, speedup[3], marker='x', color=colors[3])

plt.legend((r1, r2, r3, r4),
           ('2 Stencils', '3 Stencils', '4 Stencils', '5 Stencils'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.show()
