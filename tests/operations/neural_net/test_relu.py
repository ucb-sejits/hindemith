import unittest
import numpy as np


from hindemith.operations.neural_net import Relu
from hindemith.types import NDArray
from hindemith.core import hm


class TestRelu(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_simple(self):
        bottom = NDArray.rand((256, 12, 56, 56), np.float32)
        top = NDArray.zeros(bottom.shape, bottom.dtype)

        @hm
        def fn(bottom, top):
            top = Relu(bottom)
            return top

        fn(bottom, top)
        top.sync_host()
        expected = np.copy(bottom)
        expected[expected < 0] = 0
        self._check(top, expected)
