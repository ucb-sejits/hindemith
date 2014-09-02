import array
import png
from hindemith.fusion.core import fuse, dont_fuse_fusables

from stencil_code.stencil_kernel import StencilKernel
import numpy

from ctree.util import Timer

import logging
logging.basicConfig(level=20)


radius = 1


class PyStencil(object):
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
            out_grid[x] /= 9.0

    def __init__(self, backend='a'):
        pass

    def __call__(self, m):
        n = numpy.empty_like(m)
        for i in range(1,m.shape[0]-1):
            for j in range(1,m.shape[1]-1):
                for k in range(0,m.shape[2]):
                    for nn in self.neighbors((i, j, k), 0):
                        n[i, j, k] += m[nn[0], nn[1], nn[2]]
                    n[i, j, k] /= 9.0
        return n


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
            out_grid[x] /= 9.0

    # def __init__(self, backend='a'):
    #     pass
    #
    # def __call__(self, m):
    #     n = numpy.empty_like(m)
    #     for i in range(1,m.shape[0]-1):
    #         for j in range(1,m.shape[1]-1):
    #             for k in range(0,m.shape[2]):
    #                 for nn in self.neighbors((i, j, k), 0):
    #                     n[i, j, k] += m[nn[0], nn[1], nn[2]]
    #                 n[i, j, k] /= 9.0
    #     return n



def main():
    iterations = 1
    results = [[] for _ in range(3)]
    speedup = [[] for _ in range(4)]

    width, height, pixels, metadata = png.Reader('pa.png').read_flat()
    A = numpy.array(pixels).reshape(height, width, metadata['planes'])

    print("A.shape {}".format(A.shape))
    print("A[0][0] {}".format(A[0][0]))

    A = A.astype(numpy.float32)

    B = numpy.random.rand(A.shape[0], A.shape[1]).astype(numpy.float32) * 100
    print A[7]

    total0, total1, total2 = 0, 0, 0
    for _ in range(iterations):
        backend = 'ocl'
        pystencil1 = PyStencil(backend=backend)
        stencil1 = Stencil(backend=backend)
        stencil2 = Stencil(backend=backend)
        stencil3 = Stencil(backend=backend)
        stencil4 = Stencil(backend=backend)

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

        # a = fused_f(A)
        # b = unfused_f(A)

        print("starting a")
        a = pystencil1(A)
        print("starting b")
        b = stencil3(A)
        print("done b")

        # numpy.testing.assert_array_almost_equal(a[2:-2, 2:-2], b[2:-2, 2:-2])

    #     with Timer() as fused_time:
    #         fused_f(A)
    #     results[0].append(fused_time.interval)
    #     total0 += fused_time.interval
    #
    #     total1 += fused_time.interval
    #
    #     with Timer() as unfused_time:
    #         unfused_f(A)
    #     results[1].append(unfused_time.interval)
    #     total1 += unfused_time.interval
    #
    # total0 /= iterations
    # total1 /= iterations
    # speedup[0].append(total1/total0)
    #
    # print("total   fused {0} times {1}".format(total0, ["{:6.4f} ".format(x) for x in results[0]]))
    # print("total unfused {0} times {1}".format(total1, ["{:6.4f} ".format(x) for x in results[1]]))

    def write_image(image_array, out_file):
        m = metadata
        print("m {}".format(m))
        writer = png.Writer(width, height, alpha=m['alpha'], greyscale=m['greyscale'], bitdepth=m['bitdepth'],
                            interlace=m['interlace'], planes=m['planes'])
        output = array.array('B', image_array.reshape(width * height * m['planes']))
        writer.write_array(out_file, output)

    index = numpy.nditer(a, flags=["multi_index"])
    while not index.finished:
        if index[0] > 255:
            print("a[{}] > {} out of range".format(index.multi_index, index[0]))
            exit(0)
        index.iternext()

    with open('blurred_control.png', 'wb') as out_file:
        write_image(A, out_file)

    with open('blurred_fused_a.png', 'wb') as out_file:
        write_image(a, out_file)

    with open('blurred_fused_b.png', 'wb') as out_file:
        write_image(b, out_file)


if __name__ == '__main__':
    main()
