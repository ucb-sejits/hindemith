import unittest
from numpy import array, float32, int32
import numpy
from teller.core import coercer, hm
from teller.types.common import Float32, Int, Array, Scalar
from teller.types.stencil import Stencil
from teller.utils import unique_name

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
        # TODO: This should throw a real Exception with a relevant message
        self.assertRaises(NotImplementedError,  coercer, ('name', 'string'))

class TestDecorator(unittest.TestCase):
    def test_dec(self):
        @hm
        def test_func(arg=None):
            return arg

        a = test_func(arg=1)
        self.assertEqual(a.value, 1)


