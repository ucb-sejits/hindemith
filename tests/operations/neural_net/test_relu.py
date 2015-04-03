import unittest
import numpy as np


from hindemith.operations.neural_net import Relu
from hindemith.types import NDArray
from hindemith.core import hm


class TestRelu(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_simple(self):
        a = NDArray.rand((256, 12, 56, 56), np.float32)

        @hm
        def fn(a):
            return Relu(a)

        actual = fn(a)
        actual.sync()
        expected = np.copy(a)
        expected[expected < 0] = 0
        self._check(actual, expected)
