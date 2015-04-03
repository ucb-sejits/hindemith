import unittest
import numpy as np


from hindemith.operations.neural_net import Dropout
from hindemith.types import NDArray
from hindemith.core import hm


class TestDropout(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_simple(self):
        a = NDArray.rand((256, 12, 56, 56), np.float32)
        threshold = .5
        mask = NDArray.rand((256, 12, 56, 56), np.float32)

        @hm
        def fn(a, mask):
            return Dropout(a, threshold=0.5, mask=mask)

        actual = fn(a, mask)
        actual.sync()
        expected = np.copy(a)
        scale = 1.0 / (1.0 - threshold)
        expected = expected * mask * scale
        self._check(actual, expected)
