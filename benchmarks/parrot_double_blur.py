import png
from hindemith.fusion.core import fuse, dont_fuse_fusables

from stencil_code.stencil_kernel import StencilKernel
import numpy

from ctree.util import Timer

import logging
# logging.basicConfig(level=20)


radius = 1


class Stencil(StencilKernel):
    @property
    def dim(self):
        return 3

    @property
    def ghost_depth(self):
        return 1

    def neighbors(self, pt, defn=0):
        if defn == 0:
            for x in range(-radius, radius+1):
                for y in range(-radius, radius+1):
                    yield (pt[0] - x, pt[1] - y, pt[2])

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]
            out_grid[x] /= 5.0


def main():
    x = []
    iterations = 5
    results = [[] for _ in range(3)]
    speedup = [[] for _ in range(4)]

    width, height, pixels, metadata = png.Reader('parrot.png').read_flat()
    A = numpy.array(pixels).reshape(height, width, metadata['planes'])

    print("A.shape {}".format(A.shape))
    print("A[0][0] {}".format(A[0][0]))

    A = A.astype(numpy.float32)

    B = numpy.random.rand(A.shape[0], A.shape[1]).astype(numpy.float32) * 100
    print A[7]

    total0, total1, total2 = 0, 0, 0
    for _ in range(iterations):
        backend = 'ocl'
        pure_python = True
        stencil1 = Stencil(backend=backend)
        stencil2 = Stencil(backend=backend)
        stencil3 = Stencil(backend=backend)
        stencil4 = Stencil(backend=backend)
        # stencil1.pure_python = pure_python
        # stencil2.pure_python = pure_python
        # stencil3.pure_python = pure_python

        @fuse
        def fused_f(A):
            C = stencil1(A)
            return stencil2(C)

        # def unfused_f(A):
        #     return stencil4(stencil3(A))

        @dont_fuse_fusables
        def unfused_f(A):
            C = stencil3(A)
            return stencil4(C)


        a = fused_f(A)
        b = unfused_f(A)
        # print("A[7]:\n{}".format(A[7]))
        # print("a[7]:\n{}".format(a[7]))
        # print("b[7]:\n{}".format(b[7]))

        numpy.testing.assert_array_almost_equal(a[2:-2, 2:-2], b[2:-2, 2:-2])
        # x.append(width)
        with Timer() as fused_time:
            fused_f(A)
        results[0].append(fused_time.interval)
        total0 += fused_time.interval

        total1 += fused_time.interval

        with Timer() as unfused_time:
            unfused_f(A)
        results[1].append(unfused_time.interval)
        total1 += unfused_time.interval

    total0 /= iterations
    total1 /= iterations
    speedup[0].append(total1/total0)
    # x.append(width)

    print("total   fused {0} times {1}".format(total0, ["{:6.4f} ".format(x) for x in results[0]]))
    print("total unfused {0} times {1}".format(total1, ["{:6.4f} ".format(x) for x in results[1]]))

    with open('blurred_fused.png', 'wb') as out_file:
        m = metadata
        print("m {}".format(m))
        writer = png.Writer(width, height, alpha=m['alpha'], greyscale=m['greyscale'], bitdepth=m['bitdepth'],
                            interlace=m['interlace'], planes=m['planes'])
        output = numpy.reshape(A.astype(numpy.int64), (-1, width * height) )
        writer.write(out_file, output)

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
