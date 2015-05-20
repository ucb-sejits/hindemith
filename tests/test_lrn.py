import unittest
import numpy as np
from math import pow
import hindemith as hm
from hindemith.operations.lrn import LrnForward
from hindemith.types import hmarray
from hindemith.core import compose

local_size = 5
alpha = 0.0001
beta = 0.75


def reference_lrn(blob):
    output = hmarray(blob.shape)
    shape = blob.shape
    for n in range(shape[0]):
        for c in range(shape[1]):
            for h in range(shape[2]):
                for w in range(shape[3]):
                    c_start = c - (local_size - 1) // 2
                    c_end = min(c_start + local_size, blob.shape[1])
                    c_start = max(c_start, 0)
                    scale = 1.0
                    for i in range(c_start, c_end):
                        value = blob[n, i, h, w]
                        scale += value * value * alpha * local_size
                    output[n, c, h, w] = \
                        blob[n, c, h, w] / pow(scale, beta)
    return output


class TestLrn(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_array_almost_equal(actual, expected, decimal=2)

    def test_simple(self):
        a = hm.random((3, 16, 27, 27))
        scale = hmarray((3, 16, 27, 27))
        actual = hmarray((3, 16, 27, 27))

        @compose
        def fn(bottom, scale, top):
            top, scale = LrnForward(bottom, alpha=alpha, beta=beta,
                                    local_size=local_size, k=1)
            return top, scale

        fn(a, scale, actual)
        actual.sync_host()
        expected = reference_lrn(a)
        self._check(actual, expected)
