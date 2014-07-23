import unittest
from numpy import array, float32, int32
import numpy
from hindemith.core import coercer, fuse
from hindemith.operations.dense_linear_algebra import Float32, Int, Scalar, Array
from hindemith.types.stencil import Stencil
from hindemith.utils import unique_name, UnsupportedTypeError

__author__ = 'leonardtruong'


class TestCoercer(unittest.TestCase):
    def _check(self, value, type):
        self.assertIsInstance(coercer(('name', value))[1], type)

    def test_Float32(self):
        self._check(0.0, Float32)

    def test_Int(self):
        self._check(54, Int)

    def test_stencil(self):
        # TODO: This needs to be removed once stencils are reworked
        data = array([1.0 / 12.0, 8.0 / 12.0, -8.0 / 12.0, -1.0 / 12.0], dtype=float32)
        offx = array([-2, -1, 1, 2], dtype=int32)
        offy = array([0, 0, 0, 0], dtype=int32)
        stencil = Stencil(data, offx, offy)
        self._check(stencil, Stencil)

    def test_Array(self):
        array = Array(unique_name(), numpy.ones((10, 10)))
        self._check(array, Array)

    def test_Scalar(self):
        scalar = Scalar(unique_name(), 3)
        self._check(scalar, Scalar)

    def test_ndarray(self):
        array = numpy.ones((20, 20))
        self._check(array, Array)

    def test_not_support(self):
        self.assertRaises(UnsupportedTypeError,  coercer, ('name', 'string'))


class TestFusion(unittest.TestCase):
    def test_dec(self):
        @fuse
        def test_func(arg=None):
            return arg

        a = test_func(arg=1)
        self.assertEqual(a.value, 1)

    def test_no_fusion(self):
        @fuse
        def test_func(A=None, B=None):
            D = A * B
            return D

        A = Array('A', numpy.random.rand(200, 200).astype(numpy.float32))
        B = Array('B', numpy.random.rand(200, 200).astype(numpy.float32))
        actual = test_func(A=A, B=B)
        expected = A.data * B.data
        try:
            numpy.testing.assert_array_almost_equal(actual.data, expected, decimal=3)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_fusion_simple(self):
        @fuse
        def test_func(A=None, B=None, C=None):
            D = A * B
            E = C - D
            return E

        A = Array('A', numpy.random.rand(200, 200).astype(numpy.float32))
        B = Array('B', numpy.random.rand(200, 200).astype(numpy.float32))
        C = Array('C', numpy.random.rand(200, 200).astype(numpy.float32))
        actual = test_func(A=A, B=B, C=C)
        expected = C.data - (A.data * B.data)
        try:
            numpy.testing.assert_array_almost_equal(actual.data, expected, decimal=3)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_fusion_simple2(self):
        @fuse
        def test_func(A=None, B=None, C=None):
            D = A * B
            E = C + D
            return E

        A = Array('A', numpy.random.rand(200, 200).astype(numpy.float32))
        B = Array('B', numpy.random.rand(200, 200).astype(numpy.float32))
        C = Array('C', numpy.random.rand(200, 200).astype(numpy.float32))
        actual = test_func(A=A, B=B, C=C)
        expected = C.data + (A.data * B.data)
        try:
            numpy.testing.assert_array_almost_equal(actual.data, expected, decimal=3)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

