import unittest
from hindemith.types import hmarray
from hindemith.core import compose
from hindemith.operations import Relu, PoolForward
import numpy as np


def reference_pool(data, output, mask, kernel_size, stride, pad):
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = pad
    for n in range(output.shape[0]):
        for c in range(output.shape[1]):
            for y in range(output.shape[2]):
                for x in range(output.shape[3]):
                    y_start = max(y * stride_h - pad_h, 0)
                    x_start = max(x * stride_w - pad_w, 0)
                    y_end = min(y_start + kernel_h, data.shape[2])
                    x_end = min(x_start + kernel_w, data.shape[3])
                    for yy in range(y_start, y_end):
                        for xx in range(x_start, x_end):
                            if data[n, c, yy, xx] > output[n, c, y, x]:
                                output[n, c, y, x] = data[n, c, yy, xx]
                                mask[n, c, y, x] = yy * data.shape[3] + xx


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

    def test_pool(self):
        shape = (3, 16, 24, 24)
        a = hmarray.random(shape, _range=(0, 255))
        actual_mask = hmarray((3, 16, 12, 12))
        actual = hmarray((3, 16, 12, 12))
        expected_mask = hmarray((3, 16, 12, 12))
        expected = hmarray((3, 16, 12, 12))
        expected.fill(float('-inf'))

        @compose
        def fn(bottom, mask, top):
            top, mask = PoolForward(bottom, kernel_size=(2, 2),
                                    padding=(0, 0), stride=(2, 2))
            return top

        fn(a, actual_mask, actual)
        actual.sync_host()
        print(actual)
        actual_mask.sync_host()
        reference_pool(a, expected, expected_mask, (2, 2), (2, 2), (0, 0))
        self._check(actual, expected)
        self._check(actual_mask, expected_mask)
