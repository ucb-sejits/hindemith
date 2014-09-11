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


class Blur(StencilKernel):
    neighbor_definition = [
        [(x, y, 0) for y in range(-radius, radius+1)]
        for x in range(-radius, radius+1)
    ]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]
            out_grid[x] /= 9.0


def main():
    width, height, pixels, metadata = png.Reader('parrot.png').read_flat()
    A = numpy.array(pixels).reshape(height, width, metadata['planes']).astype(numpy.float32)

    backend = 'ocl'
    stencil_f1 = Blur(backend=backend)
    stencil_f2 = Blur(backend=backend)
    stencil_u1 = Blur(backend=backend)
    stencil_u4 = Blur(backend=backend)

    @fuse
    def fused_ocl(A):
        C = stencil_f1(A)
        return stencil_f2(C)

    @dont_fuse_fusables
    def fused_c(A):
        C = stencil_u1(A)
        return stencil_u4(C)

    a = fused_ocl(A)
    b = fused_c(A)

    # numpy.testing.assert_array_almost_equal(a[2:-2, 2:-2], b[2:-2, 2:-2], decimal=4)

    def f1():
        fused_ocl(A)

    fused_best_time = get_best_time(f1, trials=3, iterations=10)
    print("Double blur GPU best fused time   {:8.3f} ms".format(fused_best_time))

    def f2():
        fused_c(A)

    fused_best_time = get_best_time(f2, trials=3, iterations=10)
    print("Double blur GPU best unfused time {:8.3f} ms".format(fused_best_time))


if __name__ == '__main__':
    main()


