import unittest
import numpy as np


from hindemith.operations.neural_net import PoolForward, PoolBackward
from hindemith.types import NDArray
from hindemith.core import hm

local_size = 5
alpha = 0.0001
beta = 0.75


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
        np.testing.assert_array_almost_equal(actual, expected, decimal=2)

    def test_forward_backward(self):
        shape = (3, 16, 24, 24)
        a = NDArray.rand(shape, np.float32) * 255
        a.ocl_dirty = True
        a.sync()
        actual_mask = NDArray((3, 16, 12, 12), np.float32)
        actual = NDArray((3, 16, 12, 12), np.float32)
        expected_mask = NDArray((3, 16, 12, 12), np.float32)
        expected = NDArray((3, 16, 12, 12), np.float32)
        expected.fill(float('-inf'))

        @hm
        def fn(bottom, mask, top):
            top = PoolForward(bottom, mask, kernel_size=(2, 2),
                              padding=(0, 0), stride=(2, 2))
            return top

        fn(a, actual_mask, actual)
        actual.sync()
        actual_mask.sync()
        reference_pool(a, expected, expected_mask, (2, 2), (2, 2), (0, 0))
        self._check(actual, expected)
        self._check(actual_mask, expected_mask)

        bottom_diff = NDArray.zeros(shape, np.float32)
        expected_bottom_diff = NDArray.zeros(shape, np.float32)
        # mask = NDArray((3, 16, 12, 12), np.int32) 
        mask = actual_mask
        top_diff = NDArray.rand((3, 16, 12, 12), np.float32)
        top_diff.ocl_dirty = True

        @hm
        def fn(top_diff, mask, bottom_diff):
            bottom_diff = PoolBackward(top_diff, mask,
                                       kernel_size=(2, 2),
                                       padding=(0, 0),
                                       stride=(2, 2))
            return bottom_diff

        fn(top_diff, mask, bottom_diff)
        bottom_diff.sync()
        reference_pool_backward(top_diff, mask, expected_bottom_diff,
                                (2, 2), (2, 2), (0, 0))
        self._check(bottom_diff, expected_bottom_diff)
