from __future__ import print_function

import array
import png
from hindemith.fusion.core import fuse, dont_fuse_fusables

from stencil_code.stencil_kernel import StencilKernel
import numpy
from hindemith.utils import get_best_time

import logging
logging.basicConfig(level=20)


radius = 1


class X(object):
    def __

class Blur(StencilKernel):
    @property
    def dim(self):
        return 3

    @property
    def ghost_depth(self):
        return 1

    neighbor_definition = [[(x, y, 0) for y in range(-radius, radius+1)] for x in range(-radius, radius+1)]

    # def neighbors(self, pt, defn=0):
    #     if defn == 0:
    #         for x in range(-radius, radius+1):
    #             for y in range(-radius, radius+1):
    #                 yield (pt[0] - x, pt[1] - y, pt[2])

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]
            out_grid[x] /= 9.0


def main():
    width, height, pixels, metadata = png.Reader('parrot.png').read_flat()
    A = numpy.array(pixels).reshape(height, width, metadata['planes']).astype(numpy.float32)

    blur_ocl = Blur(backend='ocl')
    blur_c = Blur(backend='c')

    a = blur_ocl(A)
    b = blur_c(A)

    numpy.testing.assert_array_almost_equal(a[2:-2, 2:-2], b[2:-2, 2:-2], decimal=4)

    def f1():
        blur_ocl(A)

    fused_best_time = get_best_time(f1, trials=3, iterations=10)
    print("Double blur GPU best fused time {:8.3f} ms".format(fused_best_time))

    def f2():
        blur_ocl(A)

    fused_best_time = get_best_time(f2, trials=3, iterations=10)
    print("Double blur CPU best unfused time {:8.3f} ms".format(fused_best_time))


if __name__ == '__main__':
    main()


