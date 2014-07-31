import unittest
import numpy
from hindemith.operations.dense_linear_algebra.core import ArrayAdd, ArraySub, \
    ArrayMul, ArrayDiv, ScalarArrayAdd, ScalarArrayDiv, ScalarArrayMul, \
    ScalarArraySub, ArrayScalarAdd, ArrayScalarDiv, ArrayScalarMul, \
    ArrayScalarSub


__author__ = 'leonardtruong'


class TestDLA(unittest.TestCase):
    def test_array_add(self):
        array_add = ArrayAdd()
        a = numpy.random.rand(100, 100).astype(numpy.float32)
        b = numpy.random.rand(100, 100).astype(numpy.float32)
        try:
            numpy.testing.assert_array_almost_equal(array_add(a, b), a + b)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_array_sub(self):
        array_sub = ArraySub()
        a = numpy.random.rand(100, 100).astype(numpy.float32)
        b = numpy.random.rand(100, 100).astype(numpy.float32)
        try:
            numpy.testing.assert_array_almost_equal(array_sub(a, b), a - b)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_array_mul(self):
        array_mul = ArrayMul()
        a = numpy.random.rand(100, 100).astype(numpy.float32)
        b = numpy.random.rand(100, 100).astype(numpy.float32)
        try:
            numpy.testing.assert_array_almost_equal(array_mul(a, b), a * b)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_array_div(self):
        array_div = ArrayDiv()
        a = numpy.random.rand(100, 100).astype(numpy.float32)
        b = numpy.random.rand(100, 100).astype(numpy.float32)
        try:
            numpy.testing.assert_array_almost_equal(array_div(a, b), a / b)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_scalar_array_add(self):
        scalar_array_add = ScalarArrayAdd()
        a = 4
        b = numpy.random.rand(100, 100).astype(numpy.float32)
        try:
            numpy.testing.assert_array_almost_equal(scalar_array_add(a, b),
                                                    a + b)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_scalar_array_sub(self):
        scalar_array_sub = ScalarArraySub()
        a = 4
        b = numpy.random.rand(100, 100).astype(numpy.float32)
        try:
            numpy.testing.assert_array_almost_equal(scalar_array_sub(a, b),
                                                    a - b)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_scalar_array_div(self):
        scalar_array_div = ScalarArrayDiv()
        a = 4
        b = numpy.random.rand(100, 100).astype(numpy.float32)
        try:
            numpy.testing.assert_array_almost_equal(scalar_array_div(a, b),
                                                    a / b)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_scalar_array_mul(self):
        scalar_array_mul = ScalarArrayMul()
        a = 4
        b = numpy.random.rand(100, 100).astype(numpy.float32)
        try:
            numpy.testing.assert_array_almost_equal(scalar_array_mul(a, b),
                                                    a * b)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_array_scalar_add(self):
        array_scalar_add = ArrayScalarAdd()
        a = numpy.random.rand(100, 100).astype(numpy.float32)
        b = 4
        try:
            numpy.testing.assert_array_almost_equal(array_scalar_add(a, b),
                                                    a + b)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_array_scalar_sub(self):
        array_scalar_sub = ArrayScalarSub()
        a = numpy.random.rand(100, 100).astype(numpy.float32)
        b = 4
        try:
            numpy.testing.assert_array_almost_equal(array_scalar_sub(a, b),
                                                    a - b)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_array_scalar_div(self):
        array_scalar_div = ArrayScalarDiv()
        a = numpy.random.rand(100, 100).astype(numpy.float32)
        b = 4
        try:
            numpy.testing.assert_array_almost_equal(array_scalar_div(a, b),
                                                    a / b)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_array_scalar_mul(self):
        array_scalar_mul = ArrayScalarMul()
        a = numpy.random.rand(100, 100).astype(numpy.float32)
        b = 4
        try:
            numpy.testing.assert_array_almost_equal(array_scalar_mul(a, b),
                                                    a * b)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)
