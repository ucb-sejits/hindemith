import unittest
import numpy as np
from ctree.meta.core import meta
from .array_add import array_add
from .simple_stencil import simple_stencil
from .kernels import laplacian_2d, x_gradient, y_gradient


class TestMetaDecorator(unittest.TestCase):
    def _check_arrays_equal(self, actual, expected):
        try:
            np.testing.assert_array_almost_equal(actual, expected)
        except AssertionError as e:
            self.fail("Arrays not almost equal\n{}".format(e))

    def test_simple(self):
        @meta
        def func(a):
            return a + 3

        self.assertEqual(func(3), 6)

    def test_dataflow(self):
        @meta
        def func(a, b):
            c = array_add(a, b)
            return array_add(c, a)

        a = np.random.rand(256, 256).astype(np.float32) * 100
        b = np.random.rand(256, 256).astype(np.float32) * 100
        self._check_arrays_equal(func(a, b), a + b + a)

    def test_dataflow_2(self):
        @meta
        def func(a, b):
            c = array_add(a, b)
            d = array_add(c, b)
            return array_add(d, c)

        a = np.random.rand(256, 256).astype(np.float32) * 100
        b = np.random.rand(256, 256).astype(np.float32) * 100
        actual = func(a, b)
        c = a + b
        d = c + b
        expected = d + c
        self._check_arrays_equal(actual, expected)

    def test_multiblock_dataflow(self):
        @meta
        def func(a, b):
            c = array_add(a, b)
            d = array_add(c, b)
            e = a + d
            f = array_add(d, d)
            g = array_add(f, b)
            return a + g

        a = np.random.rand(256, 256).astype(np.float32) * 100
        b = np.random.rand(256, 256).astype(np.float32) * 100
        actual = func(a, b)
        c = a + b
        d = c + b
        e = a + d
        f = d + d
        g = f + b
        expected = a + g
        self._check_arrays_equal(actual, expected)

    def test_simple_stencil(self):
        @meta
        def func(a):
            b = simple_stencil(a)
            return simple_stencil(b)

        a = np.random.rand(32).astype(np.float32) * 100
        actual = func(a)
        expected = simple_stencil(simple_stencil(a))
        self._check_arrays_equal(actual[1:-1], expected[1:-1])

    def test_fused_gradient(self):
        a = np.random.rand(512, 512).astype(np.float32) * 100

        @meta
        def fused(a):
            y = y_gradient(a)
            x = x_gradient(a)
            return y, x
        actual = fused(a)
        expected = np.gradient(a)

        try:
            np.testing.assert_array_almost_equal(actual[0][0][1:-1, 1:-1],
                                                 expected[0][1:-1, 1:-1])
            np.testing.assert_array_almost_equal(actual[0][1][1:-1, 1:-1],
                                                 expected[1][1:-1, 1:-1])
        except AssertionError as e:
            self.fail("Arrays not almost equal\n{}".format(e))

    def test_fused_divergence(self):
        a = np.random.rand(512, 512).astype(np.float32) * 100

        @meta
        def fused(a):
            y = y_gradient(a)
            x = x_gradient(a)
            return array_add(x, y)
        actual = fused(a)
        expected = np.sum(np.gradient(a), axis=0)
        try:
            np.testing.assert_array_almost_equal(actual[1:-1, 1:-1],
                                                 expected[1:-1, 1:-1])
        except AssertionError as e:
            self.fail("Arrays not almost equal\n{}".format(e))


class TestKernels(unittest.TestCase):
    def test_laplacian(self):
        a = np.random.rand(256, 256).astype(np.float32) * 100
        actual = laplacian_2d(a)
        from scipy.ndimage.filters import laplace
        expected = laplace(a)

        try:
            np.testing.assert_array_almost_equal(
                actual[1:-1, 1:-1], expected[1:-1, 1:-1], decimal=3)
        except AssertionError as e:
            self.fail("Arrays not almost equal\n{}".format(e))

    def test_gradient(self):
        a = np.random.rand(256, 256).astype(np.float32) * 100
        actual = y_gradient(a), x_gradient(a)
        expected = np.gradient(a)

        try:
            np.testing.assert_array_almost_equal(actual[0][1:-1, 1:-1],
                                                 expected[0][1:-1, 1:-1])
            np.testing.assert_array_almost_equal(actual[1][1:-1, 1:-1],
                                                 expected[1][1:-1, 1:-1])
        except AssertionError as e:
            self.fail("Arrays not almost equal\n{}".format(e))
