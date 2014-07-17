__author__ = 'leonardtruong'

import unittest
import numpy as np

from teller.operations.dense_linear_algebra.elementwise_array_operations import ElementWiseAdd
from teller.types.common import Array


class TestElementWiseArrayOps(unittest.TestCase):
    def test_simple_elementwise_add(self):
        specialized = ElementWiseAdd(backend='ocl')
        python = ElementWiseAdd(backend='python')
        rand1 = Array('testArr1', np.random.rand(60, 80).astype(np.float32) * 100)
        rand2 = Array('testArr2', np.random.rand(60, 80).astype(np.float32) * 100)
        actual = specialized(rand1, rand2)
        expected = python(rand1, rand2)
        try:
            np.testing.assert_array_almost_equal(actual.data, expected.data, decimal=3)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)


