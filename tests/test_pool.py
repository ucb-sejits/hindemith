import unittest
import numpy as np
import hindemith as hm
from hindemith.types import hmarray
from hindemith.operations.pool import PoolForward, PoolBackward, AvePoolForward
from hindemith.core import compose


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


def reference_ave_pool(data, output, kernel_size, stride, pad):
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
                    ave = 0
                    for yy in range(y_start, y_end):
                        for xx in range(x_start, x_end):
                            ave += data[n, c, yy, xx]
                    output[n, c, y, x] = ave / (kernel_h * kernel_w)


def reference_pool_backward(top_diff, mask, bottom_diff, kernel_size, stride, pad):
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = pad
    for n in range(top_diff.shape[0]):
        for c in range(bottom_diff.shape[1]):
            for ph in range(top_diff.shape[2]):
                for pw in range(top_diff.shape[3]):
                    bottom_index = mask[n, c, ph, pw]
                    bottom_h = bottom_index / bottom_diff.shape[3]
                    bottom_w = bottom_index % bottom_diff.shape[3]
                    bottom_diff[n, c, bottom_h, bottom_w] += top_diff[n, c, ph, pw]


class TestPool(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_array_almost_equal(actual, expected, decimal=4)

    def test_pool(self):
        shape = (3, 16, 24, 24)
        a = hm.random(shape, _range=(0, 255))
        actual_mask = hmarray((3, 16, 12, 12))
        actual = hmarray((3, 16, 12, 12))
        expected_mask = hmarray((3, 16, 12, 12))
        expected = hmarray((3, 16, 12, 12))
        expected.fill(float('-inf'))

        @compose
        def fn(bottom, mask, top):
            top, mask = PoolForward(bottom, kernel_size=(2, 2),
                                    padding=(0, 0), stride=(2, 2))
            return top, mask

        fn(a, actual_mask, actual)
        actual.sync_host()
        actual_mask.sync_host()
        reference_pool(a, expected, expected_mask, (2, 2), (2, 2), (0, 0))
        self._check(actual, expected)
        self._check(actual_mask, expected_mask)
        bottom_diff = hm.zeros(shape)
        expected_bottom_diff = hm.zeros(shape)
        mask = actual_mask
        top_diff = hm.random((3, 16, 12, 12))

        @compose
        def fn(top_diff, mask, bottom_diff):
            bottom_diff = PoolBackward(top_diff, mask,
                                       kernel_size=(2, 2),
                                       padding=(0, 0),
                                       stride=(2, 2))
            return bottom_diff

        fn(top_diff, mask, bottom_diff)
        bottom_diff.sync_host()
        reference_pool_backward(top_diff, mask, expected_bottom_diff,
                                (2, 2), (2, 2), (0, 0))
        self._check(bottom_diff, expected_bottom_diff)

    def test_avg(self):
        shape = (3, 16, 24, 24)
        a = hm.random(shape, _range=(0, 255))
        actual_mask = hmarray((3, 16, 12, 12))
        actual = hmarray((3, 16, 12, 12))
        expected = hmarray((3, 16, 12, 12))
        expected.fill(float('-inf'))

        @compose
        def fn(bottom, mask, top):
            top = AvePoolForward(bottom, kernel_size=(2, 2),
                                 padding=(0, 0), stride=(2, 2))
            return top

        fn(a, actual_mask, actual)
        actual.sync_host()
        reference_ave_pool(a, expected, (2, 2), (2, 2), (0, 0))
        self._check(actual, expected)
