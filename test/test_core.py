import unittest
import numpy
from hindemith.core import coercer
from hindemith.types.common import Float32, Int, Scalar, Array
from hindemith.utils import unique_name, UnsupportedTypeError

__author__ = 'leonardtruong'


class TestCoercer(unittest.TestCase):
    def _check(self, value, type):
        self.assertIsInstance(coercer(('name', value))[1], type)

    def test_Float32(self):
        self._check(0.0, Float32)

    def test_Int(self):
        self._check(54, Int)

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
        self.assertRaises(UnsupportedTypeError, coercer, ('name', 'string'))
