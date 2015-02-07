import unittest
import numpy as np
from hindemith.meta.core import meta
# from .array_add import array_add
# from .simple_stencil import simple_stencil
# from .kernels import laplacian_2d, x_gradient, y_gradient

from hindemith.utils import symbols
from hindemith.operations.zip_with import zip_with, ZipWith
from hindemith.operations.map import square
from hindemith.types.hmarray import hmarray, EltWiseArrayOp


theta = .3
l = .15


@symbols({'l': l, 'theta': theta})
def ocl_th(rho_elt, gradient_elt, delta_elt, u_elt):
    threshold = float(l * theta * gradient_elt)
    if rho_elt < -threshold:
        return float(l * theta * delta_elt) + u_elt
    elif rho_elt > threshold:
        return float(-l * theta * delta_elt) + u_elt
    elif gradient_elt > 1e-10:
        return -rho_elt / gradient_elt * delta_elt + u_elt
    else:
        return float(0)


@symbols({'l': l, 'theta': theta})
def th(rho_elt, gradient_elt, delta_elt, u_elt):
    threshold = l * theta * gradient_elt
    if rho_elt < -threshold:
        return l * theta * delta_elt + u_elt
    elif rho_elt > threshold:
        return -l * theta * delta_elt + u_elt
    elif gradient_elt > 1e-10:
        return -rho_elt / gradient_elt * delta_elt + u_elt
    else:
        return 0

from sys import platform as _platform
if _platform == "darwin":
    threshold = zip_with(ocl_th)
else:
    threshold = zip_with(th)


class TestFusion(unittest.TestCase):
    def _check_arrays_equal(self, actual, expected):
        try:
            np.testing.assert_allclose(actual, expected, atol=1e-2)
        except AssertionError as e:
            self.fail(e)

    def setUp(self):
        self.a = np.random.rand(480, 640).astype(np.float32) * 255
        self.b = np.random.rand(480, 640).astype(np.float32) * 255
        self.c = np.random.rand(480, 640).astype(np.float32) * 255

    def test_simple(self):
        a, b, c = hmarray(self.a), hmarray(self.b), hmarray(self.c)

        def unfused(a, b, c):
            d = a + b
            e = square(c) + b
            return d + e

        fused = meta(unfused)

        for _ in range(10):
            actual = fused(a, b, c)
            expected = unfused(a, b, c)
            actual.copy_to_host_if_dirty()
            expected.copy_to_host_if_dirty()
            self._check_arrays_equal(actual, expected)

    def test_threshold(self):
        ZipWith.backend = 'ocl'
        EltWiseArrayOp.backend = 'ocl'
        a = hmarray(np.random.rand(480, 640).astype(np.float32) * 255)
        b = hmarray(np.random.rand(480, 640).astype(np.float32) * 255)
        c = hmarray(np.random.rand(480, 640).astype(np.float32) * 255)
        d = hmarray(np.random.rand(480, 640).astype(np.float32) * 255)
        e = hmarray(np.random.rand(480, 640).astype(np.float32) * 255)
        f = hmarray(np.random.rand(480, 640).astype(np.float32) * 255)

        def unfused(u1, u2, rho_c, gradient, I1wx, I1wy):
            rho = rho_c + I1wx * u1 + I1wy * u2
            v1 = threshold(rho, gradient, I1wx, u1)
            v2 = threshold(rho, gradient, I1wy, u2)
            return v1, v2

        fused = meta(unfused)

        actual = fused(a, b, c, d, e, f)
        expected = unfused(a, b, c, d, e, f)
        expected[0].copy_to_host_if_dirty()
        expected[1].copy_to_host_if_dirty()
        actual[0].copy_to_host_if_dirty()
        actual[1].copy_to_host_if_dirty()
        self._check_arrays_equal(actual[0], expected[0])
        self._check_arrays_equal(actual[1], expected[1])


# @unittest.skip("deprecated")
# class TestMetaDecorator(unittest.TestCase):
#     def _check_arrays_equal(self, actual, expected):
#         try:
#             np.testing.assert_array_almost_equal(actual, expected)
#         except AssertionError as e:
#             self.fail("Arrays not almost equal\n{}".format(e))
#
#     def test_simple(self):
#         @meta
#         def func(a):
#             return a + 3
#
#         self.assertEqual(func(3), 6)
#
#     def test_dataflow(self):
#         @meta
#         def func(a, b):
#             c = array_add(a, b)
#             return array_add(c, a)
#
#         a = np.random.rand(256, 256).astype(np.float32) * 100
#         b = np.random.rand(256, 256).astype(np.float32) * 100
#         self._check_arrays_equal(func(a, b), a + b + a)
#
#     def test_dataflow_2(self):
#         @meta
#         def func(a, b):
#             c = array_add(a, b)
#             d = array_add(c, b)
#             return array_add(d, c)
#
#         a = np.random.rand(256, 256).astype(np.float32) * 100
#         b = np.random.rand(256, 256).astype(np.float32) * 100
#         actual = func(a, b)
#         c = a + b
#         d = c + b
#         expected = d + c
#         self._check_arrays_equal(actual, expected)
#
#     def test_multiblock_dataflow(self):
#         @meta
#         def func(a, b):
#             c = array_add(a, b)
#             d = array_add(c, b)
#             e = a + d
#             print(e)
#             f = array_add(d, d)
#             g = array_add(f, b)
#             return a + g
#
#         a = np.random.rand(256, 256).astype(np.float32) * 100
#         b = np.random.rand(256, 256).astype(np.float32) * 100
#         actual = func(a, b)
#         c = a + b
#         d = c + b
#         e = a + d
#         print(e)
#         f = d + d
#         g = f + b
#         expected = a + g
#         self._check_arrays_equal(actual, expected)
#
#     def test_simple_stencil(self):
#         @meta
#         def func(a):
#             b = simple_stencil(a)
#             return simple_stencil(b)
#
#         a = np.random.rand(32).astype(np.float32) * 100
#         actual = func(a)
#         expected = simple_stencil(simple_stencil(a))
#         self._check_arrays_equal(actual[1:-1], expected[1:-1])
#
#     def test_fused_gradient(self):
#         a = np.random.rand(512, 512).astype(np.float32) * 100
#
#         @meta
#         def fused(a):
#             y = y_gradient(a)
#             x = x_gradient(a)
#             return y, x
#         actual = fused(a)
#         expected = np.gradient(a)
#
#         try:
#             np.testing.assert_array_almost_equal(actual[0][0][1:-1, 1:-1],
#                                                  expected[0][1:-1, 1:-1])
#             np.testing.assert_array_almost_equal(actual[0][1][1:-1, 1:-1],
#                                                  expected[1][1:-1, 1:-1])
#         except AssertionError as e:
#             self.fail("Arrays not almost equal\n{}".format(e))
#
#     def test_fused_divergence(self):
#         a = np.random.rand(512, 512).astype(np.float32) * 100
#
#         @meta
#         def fused(a):
#             y = y_gradient(a)
#             x = x_gradient(a)
#             return array_add(x, y)
#         actual = fused(a)
#         expected = np.sum(np.gradient(a), axis=0)
#         try:
#             np.testing.assert_array_almost_equal(actual[1:-1, 1:-1],
#                                                  expected[1:-1, 1:-1])
#         except AssertionError as e:
#             self.fail("Arrays not almost equal\n{}".format(e))
#
#
# @unittest.skip("deprecated")
# class TestKernels(unittest.TestCase):
#     @unittest.skip("Dependency issues")
#     def test_laplacian(self):
#         a = np.random.rand(256, 256).astype(np.float32) * 100
#         actual = laplacian_2d(a)
#         from scipy.ndimage.filters import laplace
#         expected = laplace(a)
#
#         try:
#             np.testing.assert_array_almost_equal(
#                 actual[1:-1, 1:-1], expected[1:-1, 1:-1], decimal=3)
#         except AssertionError as e:
#             self.fail("Arrays not almost equal\n{}".format(e))
#
#     def test_gradient(self):
#         a = np.random.rand(256, 256).astype(np.float32) * 100
#         actual = y_gradient(a), x_gradient(a)
#         expected = np.gradient(a)
#
#         try:
#             np.testing.assert_array_almost_equal(actual[0][1:-1, 1:-1],
#                                                  expected[0][1:-1, 1:-1])
#             np.testing.assert_array_almost_equal(actual[1][1:-1, 1:-1],
#                                                  expected[1][1:-1, 1:-1])
#         except AssertionError as e:
#             self.fail("Arrays not almost equal\n{}".format(e))
