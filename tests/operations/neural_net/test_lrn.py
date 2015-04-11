import unittest
import numpy as np
from math import pow


from hindemith.operations.neural_net import LrnForward, LrnBackward
from hindemith.types import NDArray
from hindemith.core import hm

local_size = 5
alpha = 0.0001
beta = 0.75


def reference_lrn(blob):
    output = NDArray.like(blob)
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
        a = NDArray.rand((3, 16, 27, 27), np.float32)
        scale = NDArray((3, 16, 27, 27), np.float32)
        actual = NDArray((3, 16, 27, 27), np.float32)

        @hm
        def fn(bottom, scale, top):
            top = LrnForward(bottom, scale, alpha=alpha, beta=beta,
                             local_size=local_size, k=1)
            return top

        fn(a, scale, actual)
        actual.sync_host()
        expected = reference_lrn(a)
        self._check(actual, expected)

        @hm
        def fn(bottom, top, scale, top_diff, bottom_diff):
            bottom_diff = LrnBackward(bottom, top, top_diff, scale,
                                      alpha=alpha, beta=beta,
                                      local_size=local_size)
            return bottom_diff
        top_diff = NDArray.rand((3, 16, 27, 27), np.float32)
        bottom_diff = NDArray.zeros(a.shape, np.float32)
        fn(a, actual, scale, top_diff, bottom_diff)
