import unittest
import numpy
from hindemith.operations.dense_linear_algebra.core import array_add, \
    array_sub, array_mul, array_div, scalar_array_add, scalar_array_sub, \
    scalar_array_mul, scalar_array_div, array_scalar_add, array_scalar_sub, \
    array_scalar_mul, array_scalar_div


__author__ = 'leonardtruong'


class TestDLA(unittest.TestCase):
    def _check(self, actual, expected):
        try:
            numpy.testing.assert_array_almost_equal(actual, expected)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e)

    def test_array_add(self):
        a = numpy.random.rand(100, 100).astype(numpy.float32)
        b = numpy.random.rand(100, 100).astype(numpy.float32)
        self._check(array_add(a, b), a + b)

    def test_array_sub(self):
        a = numpy.random.rand(100, 100).astype(numpy.float32)
        b = numpy.random.rand(100, 100).astype(numpy.float32)
        self._check(array_sub(a, b), a - b)

    def test_array_mul(self):
        a = numpy.random.rand(100, 100).astype(numpy.float32)
        b = numpy.random.rand(100, 100).astype(numpy.float32)
        self._check(array_mul(a, b), a * b)

    def test_array_div(self):
        a = numpy.random.rand(100, 100).astype(numpy.float32)
        b = numpy.random.rand(100, 100).astype(numpy.float32)
        self._check(array_div(a, b), a / b)

    def test_scalar_array_add(self):
        a = 4
        b = numpy.random.rand(100, 100).astype(numpy.float32)
        self._check(scalar_array_add(a, b), a + b)

    def test_scalar_array_sub(self):
        a = 4
        b = numpy.random.rand(100, 100).astype(numpy.float32)
        self._check(scalar_array_sub(a, b), a - b)

    def test_scalar_array_div(self):
        a = 4
        b = numpy.random.rand(100, 100).astype(numpy.float32)
        self._check(scalar_array_div(a, b), a / b)

    def test_scalar_array_mul(self):
        a = 4
        b = numpy.random.rand(100, 100).astype(numpy.float32)
        self._check(scalar_array_mul(a, b), a * b)

    def test_array_scalar_add(self):
        a = numpy.random.rand(100, 100).astype(numpy.float32)
        b = 4
        self._check(array_scalar_add(a, b), a + b)

    def test_array_scalar_sub(self):
        a = numpy.random.rand(100, 100).astype(numpy.float32)
        b = 4
        self._check(array_scalar_sub(a, b), a - b)

    def test_array_scalar_div(self):
        a = numpy.random.rand(100, 100).astype(numpy.float32)
        b = 4
        self._check(array_scalar_div(a, b), a / b)

    def test_array_scalar_mul(self):
        a = numpy.random.rand(100, 100).astype(numpy.float32)
        b = 4
        self._check(array_scalar_mul(a, b), a * b)
