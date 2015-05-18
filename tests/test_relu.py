import unittest
import hindemith as hm
from hindemith.types import hmarray
from hindemith.core import compose
from hindemith.operations.relu import ReluForward, ReluBackward
import numpy as np
import os


class TestRelu(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_relu(self):
        top = hm.zeros((4, 12, 15, 15))
        bottom = hm.random((4, 12, 15, 15), _range=(-1, 1))

        @compose
        def fn(bottom, top):
            top = ReluForward(bottom)
            return top

        fn(bottom, top)
        top.sync_host()

        expected = np.copy(bottom)
        expected[expected < 0] = 0
        self._check(top, expected)

        top_diff = hm.random(top.shape)
        bottom_diff = hmarray(top.shape)

        @compose
        def fn(top_diff, bottom, bottom_diff):
            bottom_diff = ReluBackward(bottom, top_diff)
            return bottom_diff

        fn(top_diff, bottom, bottom_diff)
        bottom_diff.sync_host()

        expected = np.copy(top_diff)
        expected[bottom < 0] = 0
        self._check(bottom_diff, expected)
