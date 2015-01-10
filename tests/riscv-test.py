from stencil_code.stencil_kernel import Stencil


class Dx(Stencil):
    neighborhoods = [
        [(1, 0), (1, 1)],
        [(0, 0), (0, 1)]]

    def kernel(self, a, b, out_grid):
        for x in self.interior_points(out_grid):
            out_grid[x] = 0.0
            for y in self.neighbors(x, 0):
                out_grid[x] += .25 * (a[y] + b[y])
            for y in self.neighbors(x, 1):
                out_grid[x] -= .25 * (a[y] + b[y])


class Dy(Stencil):
    neighborhoods = [
        [(0, 1), (1, 1)],
        [(0, 0), (1, 0)]]

    def kernel(self, a, b, out_grid):
        for x in self.interior_points(out_grid):
            out_grid[x] = 0.0
            for y in self.neighbors(x, 0):
                out_grid[x] += .25 * (a[y] + b[y])
            for y in self.neighbors(x, 1):
                out_grid[x] -= .25 * (a[y] + b[y])


dx, dy = Dx(backend='c'), Dy(backend='c')


alpha2 = 15 ** 2
from hindemith.operations.map import square
from hindemith.meta.core import meta
from hindemith.types.hmarray import hmarray


@meta
def gradient_and_denom(im0, im1):
    Ix = dx(im0, im1)
    Iy = dy(im0, im1)
    It = im1 - im0
    denom = square(Ix) + square(Iy) + alpha2
    return Ix, Iy, It, denom

import numpy as np

a = hmarray(np.arange(16).reshape((4, 4)).astype(np.float32))
b = hmarray(np.arange(16).reshape((4, 4)).astype(np.float32) * 2)
Ix, Iy, It, denom = gradient_and_denom(a, b)

expected_Ix = np.array([[ 6.,  6.,  6.,  6.],
                        [ 6.,  6.,  6.,  6.],
                        [ 6.,  6.,  6.,  6.],
                        [ 0.,  0.,  0.,  0.]])

expected_Iy = np.array([[ 1.5,  1.5,  1.5,  0. ],
                        [ 1.5,  1.5,  1.5,  0. ],
                        [ 1.5,  1.5,  1.5,  0. ],
                        [ 1.5,  1.5,  1.5,  0. ]])

expected_It = np.array([[  0.,   1.,   2.,   3.],
                        [  4.,   5.,   6.,   7.],
                        [  8.,   9.,  10.,  11.],
                        [ 12.,  13.,  14.,  15.]])

expected_denom = np.array([[ 263.25,  263.25,  263.25,  261.  ],
                           [ 263.25,  263.25,  263.25,  261.  ],
                           [ 263.25,  263.25,  263.25,  261.  ],
                           [ 227.25,  227.25,  227.25,  225.  ]])

np.testing.assert_array_almost_equal(np.copy(Ix), expected_Ix)
np.testing.assert_array_almost_equal(np.copy(Iy), expected_Iy)
np.testing.assert_array_almost_equal(np.copy(It), expected_It)
np.testing.assert_array_almost_equal(np.copy(denom), expected_denom)

print("Passed")
