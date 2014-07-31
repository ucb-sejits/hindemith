import unittest
import numpy
from hindemith.operations.dense_linear_algebra.core import ArrayAdd, ArraySub, \
    ArrayMul, ArrayDiv

__author__ = 'leonardtruong'


class TestDLA(unittest.TestCase):
    def test_array_add(self):
        array_add = ArrayAdd()
        a = numpy.random.rand(100, 100)
        b = numpy.random.rand(100, 100)
        try:
            numpy.testing.assert_array_almost_equal(array_add(a, b), a + b)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_array_sub(self):
        array_sub = ArraySub()
        a = numpy.random.rand(100, 100)
        b = numpy.random.rand(100, 100)
        try:
            numpy.testing.assert_array_almost_equal(array_sub(a, b), a - b)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_array_mul(self):
        array_mul = ArrayMul()
        a = numpy.random.rand(100, 100)
        b = numpy.random.rand(100, 100)
        try:
            numpy.testing.assert_array_almost_equal(array_mul(a, b), a * b)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_array_div(self):
        array_div = ArrayDiv()
        a = numpy.random.rand(100, 100)
        b = numpy.random.rand(100, 100)
        try:
            numpy.testing.assert_array_almost_equal(array_div(a, b), a / b)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

