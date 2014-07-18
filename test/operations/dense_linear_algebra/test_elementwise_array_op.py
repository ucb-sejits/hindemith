__author__ = 'leonardtruong'

import unittest
import numpy as np

from teller.operations.dense_linear_algebra.array_op import ArrayAdd
from teller.types.common import Array


class TestArrayOps(unittest.TestCase):
    def test_simple_array_add(self):
        specialized = ArrayAdd(backend='ocl')
        python = ArrayAdd(backend='python')
        rand1 = Array('testArr1', np.random.rand(60, 80).astype(np.float32) * 100)
        rand2 = Array('testArr2', np.random.rand(60, 80).astype(np.float32) * 100)
        actual = specialized(rand1, rand2)
        expected = python(rand1, rand2)
        try:
            np.testing.assert_array_almost_equal(actual.data, expected.data, decimal=3)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)


