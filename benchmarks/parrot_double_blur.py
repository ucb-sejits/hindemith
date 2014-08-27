import png
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


def main():
    x = []
    iterations = 2
    results = [[] for _ in range(3)]
    speedup = [[] for _ in range(4)]

    image = png.Reader('parrot.png').read()
    raw_data = list(image[2])
    A = numpy.zeros([image[0], image[1]])
    for row in range(image[0]):
        for col in range(image[1]):
            A[row, col] = raw_data[row][col]

    A = A.astype(numpy.float32)
    print A[7]

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

        a = fused_f(A)
        b = unfused_f(A)
        print a[7]
        print b[7]

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

    print total0
    print total1

    # colors = ['b', 'c', 'r', 'g']
    # import matplotlib.pyplot as plt
    #
    # r1 = plt.scatter(x, speedup[0], marker='x', color=colors[0])
    # r2 = plt.scatter(x, speedup[1], marker='x', color=colors[1])
    # r3 = plt.scatter(x, speedup[2], marker='x', color=colors[2])
    # r4 = plt.scatter(x, speedup[3], marker='x', color=colors[3])
    #
    # plt.legend((r1, r2, r3, r4),
    #            ('2 Stencils', '3 Stencils', '4 Stencils', '5 Stencils'),
    #            scatterpoints=1,
    #            loc='lower left',
    #            ncol=3,
    #            fontsize=8)
    # plt.show()

if __name__ == '__main__':
    main()
