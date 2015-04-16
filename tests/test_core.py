import unittest
from hindemith.types import hmarray
from hindemith.core import compose
from hindemith.operations import Relu
import numpy as np


class TestCore(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_relu(self):
        top = hmarray.zeros((4, 12, 15, 15))
        bottom = hmarray.random((4, 12, 15, 15), _range=(-1, 1))

        @compose
        def fn(bottom, top):
            top = Relu(bottom)
            return top

        fn(bottom, top)
        top.sync_host()

        expected = np.copy(bottom)
        expected[expected < 0] = 0
        self._check(top, expected)
