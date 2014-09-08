from __future__ import print_function

import array
import png
from hindemith.fusion.core import fuse, dont_fuse_fusables

from stencil_code.stencil_kernel import StencilKernel
import numpy
from hindemith.utils import get_best_time

# import logging
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
            out_grid[x] /= 9.0


def main():
    width, height, pixels, metadata = png.Reader('parrot.png').read_flat()
    A = numpy.array(pixels).reshape(height, width, metadata['planes'])

    print("A.shape {}".format(A.shape))
    print("A[0][0] {}".format(A[0][0]))

    A = A.astype(numpy.float32)

    total0, total1, total2 = 0, 0, 0
    backend = 'ocl'
    stencil1 = Stencil(backend=backend)
    stencil2 = Stencil(backend=backend)
    backend = "c"
    stencil3 = Stencil(backend=backend)
    stencil4 = Stencil(backend=backend)

    @fuse
    def fused_ocl(A):
        C = stencil1(A)
        return stencil2(C)

    # @fuse   c does not support fusion at this time
    def fused_c(A):
        C = stencil3(A)
        return stencil4(C)

    a = fused_ocl(A)
    b = fused_c(A)

    print("array computed by ocl")
    for ii in range(10):
        for jj in range(10):
            print("{:10s}".format(":".join(["{:.0f}".format(a[ii][jj][kk]) for kk in range(3)])), end="")
        print()

    print("array computed by c")
    for ii in range(10):
        for jj in range(10):
            print("{:10s}".format(":".join(["{:.0f}".format(b[ii][jj][kk]) for kk in range(3)])), end="")
        print()

    # print("a\n{}".format(a[:10][:10][0]))
    # print("b\n{}".format(a[:10][:10][0]))

    numpy.testing.assert_array_almost_equal(a[2:-2, 2:-2], b[2:-2, 2:-2], decimal=4)

    def f1():
        fused_ocl(A)

    fused_best_time = get_best_time(f1, trials=3, iterations=10)
    print("Double blur GPU best fused time {:8.3f} ms".format(fused_best_time))

    def f2():
        fused_ocl(A)

    fused_best_time = get_best_time(f2, trials=3, iterations=10)
    print("Double blur CPU best unfused time {:8.3f} ms".format(fused_best_time))

    # def write_image(image_array, out_file):
    #     m = metadata
    #     print("m {}".format(m))
    #     writer = png.Writer(width, height, alpha=m['alpha'], greyscale=m['greyscale'], bitdepth=m['bitdepth'],
    #                         interlace=m['interlace'], planes=m['planes'])
    #     output = array.array('B', image_array.reshape(width * height * m['planes']))
    #     writer.write_array(out_file, output)
    #
    index = numpy.nditer(b, flags=["multi_index"])
    while not index.finished:
        if index[0] > 255:
            print("b[{}] > {} out of range".format(index.multi_index, index[0]))
            exit(0)
        index.iternext()
    #
    # with open('blurred_control.png', 'wb') as out_file:
    #     write_image(A, out_file)
    #
    # with open('blurred_fused_b.png', 'wb') as out_file:
    #     write_image(b, out_file)
    #
    # with open('blurred_fused_a.png', 'wb') as out_file:
    #     write_image(a, out_file)


if __name__ == '__main__':
    main()


