import unittest
import numpy as np
from hindemith.types import NDArray
from hindemith.core import hm
from hindemith.operations import Im2Col


def reference_im2col(data, kernel_size, stride, padding):
    channels, height, width = data.shape
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    height_col = (height + 2 * pad_h - kernel_h) // stride_h + 1
    width_col = (width + 2 * pad_w - kernel_w) // stride_w + 1
    channels_col = channels * kernel_h * kernel_w
    data_col = NDArray.zeros((channels_col, height_col, width_col), np.float32)
    for c in range(channels_col):
        w_offset = c % kernel_w
        h_offset = (c / kernel_w) % kernel_h
        c_im = c / kernel_h / kernel_w
        for h in range(height_col):
            for w in range(width_col):
                h_pad = h * stride_h - pad_h + h_offset
                w_pad = w * stride_w - pad_w + w_offset
                if (h_pad >= 0 and h_pad < height and w_pad >= 0 and w_pad < width):
                    data_col[c, h, w] = data[c_im, h_pad, w_pad];
                else:
                    data_col[c, h, w] = 0;
    return data_col


class TestIm2Row(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)
        
    def test_simple(self):
        @hm
        def fn(a, col):
            col = Im2Col(a, kernel_size=(11, 11), stride=(4, 4), padding=(0, 0))
            return col

        a = NDArray.rand((3, 227, 227), np.float32) * 255
        a.ocl_dirty = True
        col = NDArray.rand((363, 3025), np.float32)

        expected = reference_im2col(a, (11, 11), (4, 4), (0, 0))
        col = fn(a, col)
        col.sync()
        self._check(col, expected.reshape(363, 3025))
