
import unittest

from stencil_code.stencil_kernel import StencilKernel
from hindemith.fusion.core import fuse
from hindemith.operations.dense_linear_algebra.core import array_scalar_add

import numpy
from numpy import testing


stdev_d = 3
stdev_s = 70
radius = 1
width = 64 + radius * 2
height = width


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


class Stencil3D(StencilKernel):
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
                    for z in range(-radius, radius+1):
                        yield (pt[0] - x, pt[1] - y, pt[2] - z)

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]


class TestFusingStencils(unittest.TestCase):
    def _check(self, actual, expected):
        try:
            testing.assert_array_almost_equal(actual, expected, decimal=3)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e)

    def test_fusing_stencil_array_op(self):
        in_grid = numpy.random.rand(1024, 1024).astype(numpy.float32) * 100
        kernel1 = Stencil(backend='ocl')
        kernel2 = Stencil(backend='ocl')

        @fuse
        def f(in_grid):
            out = kernel1(in_grid)
            return array_scalar_add(out, 4)

        actual = f(in_grid)
        expected = kernel2(in_grid) + 4
        self._check(actual, expected)

    def test_fusing_stencils(self):
        in_grid = numpy.random.rand(1024, 1024).astype(numpy.float32) * 10
        kernel1 = Stencil(backend='ocl')
        kernel2 = Stencil(backend='ocl')
        py_kernel1 = Stencil(backend='ocl')
        py_kernel2 = Stencil(backend='ocl')

        @fuse
        def f(in_grid):
            a = kernel1(in_grid)
            return kernel2(a)

        actual = f(in_grid)
        expected = py_kernel2(py_kernel1(in_grid))
        self._check(actual[2:-2, 2:-2], expected[2:-2, 2:-2])

    def test_fusing_3_stencils(self):
        in_grid = numpy.random.rand(1024, 1024).astype(numpy.float32) * 10
        kernel1 = Stencil(backend='ocl')
        kernel2 = Stencil(backend='ocl')
        kernel3 = Stencil(backend='ocl')
        py_kernel1 = Stencil(backend='ocl')
        py_kernel2 = Stencil(backend='ocl')
        py_kernel3 = Stencil(backend='ocl')

        @fuse
        def f(in_grid):
            a = kernel1(in_grid)
            b = kernel2(a)
            return kernel3(b)

        actual = f(in_grid)
        expected = py_kernel3(py_kernel2(py_kernel1(in_grid)))
        self._check(actual[2:-2, 2:-2], expected[2:-2, 2:-2])

    def test_fusing_4_stencils(self):
        in_grid = numpy.random.rand(1024, 1024).astype(numpy.float32) * 10
        kernel1 = Stencil(backend='ocl')
        kernel2 = Stencil(backend='ocl')
        kernel3 = Stencil(backend='ocl')
        kernel4 = Stencil(backend='ocl')
        py_kernel1 = Stencil(backend='ocl')
        py_kernel2 = Stencil(backend='ocl')
        py_kernel3 = Stencil(backend='ocl')
        py_kernel4 = Stencil(backend='ocl')

        @fuse
        def f(in_grid):
            a = kernel1(in_grid)
            b = kernel2(a)
            c = kernel3(b)
            return kernel4(c)

        actual = f(in_grid)
        expected = py_kernel4(py_kernel3(py_kernel2(py_kernel1(in_grid))))
        self._check(actual[4:-4, 4:-4], expected[4:-4, 4:-4])

    def test_fusing_stencils_3d(self):
        in_grid = numpy.random.rand(
            512, 512, 128).astype(numpy.float32) * 10
        kernel1 = Stencil3D(backend='ocl')
        kernel2 = Stencil3D(backend='ocl')
        py_kernel1 = Stencil3D(backend='ocl')

        @fuse
        def f(in_grid):
            a = kernel1(in_grid)
            return kernel2(a)

        actual = f(in_grid)
        expected = py_kernel1(py_kernel1(in_grid))
        self._check(actual[2:-2, 2:-2, 2:-2], expected[2:-2, 2:-2, 2:-2])

    def test_fusing_3_stencils_3d(self):
        in_grid = numpy.random.rand(
            512, 512, 128).astype(numpy.float32) * 10
        kernel1 = Stencil3D(backend='ocl')
        kernel2 = Stencil3D(backend='ocl')
        kernel3 = Stencil3D(backend='ocl')
        py_kernel1 = Stencil3D(backend='ocl')

        @fuse
        def f(in_grid):
            a = kernel1(in_grid)
            b = kernel2(a)
            return kernel3(b)

        actual = f(in_grid)
        expected = py_kernel1(py_kernel1(py_kernel1(in_grid)))
        self._check(actual[2:-2, 2:-2, 2:-2], expected[2:-2, 2:-2, 2:-2])
