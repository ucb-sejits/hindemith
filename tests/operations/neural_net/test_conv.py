import unittest
import numpy as np
from hindemith.types import NDArray
from hindemith.core import hm
from hindemith.operations.neural_net import ConvForward

def reference_conv(in_data, weights, out, stride, pad):
    stride_h, stride_w = stride
    pad_h, pad_w = pad
    for n in range(out.shape[0]):
        for out_c in range(weights.shape[0]):
            for in_c in range(weights.shape[1]):
                for y in range(out.shape[2]):
                    for x in range(out.shape[3]):
                        for p in range(weights.shape[2]):
                            for q in range(weights.shape[3]):
                                in_y = y * stride_h - pad_h + p
                                in_x = x * stride_w - pad_w + q
                                if (in_y >= 0 and in_y < in_data.shape[2] and in_x >= 0 and in_x < in_data.shape[3]):
                                    out[n, out_c, y, x] += in_data[n, in_c, in_y, in_x] \
                                        * weights[out_c, in_c, p, q]


class TestConv(unittest.TestCase):
    def test_forward(self):
        @hm
        def fn(a, weights, out):
            out = ConvForward(a, weights, kernel_size=(11, 11), padding=(0, 0), stride=(4, 4))
            return out

        a = NDArray.rand((3, 3, 227, 227), np.float32) * 255
        a.ocl_dirty = True
        weights = NDArray.rand((12, 363), np.float32)
        out = NDArray.zeros((3, 12, 3025), np.float32)

        out = fn(a, weights, out)

        weights = weights.reshape(12, 3, 11, 11)
        expected = np.zeros((3, 12, 55, 55), np.float32)
        reference_conv(a, weights, expected, (4, 4), (0, 0))
        out.sync()
        np.testing.assert_array_almost_equal(out, expected.reshape(3, 12, 3025), decimal=2)


    def test_forward(self):
        n = .1
        @hm
        def fn(top_diff, weights, bottom, bottom_diff):
            bottom_diff = ConvBackward(bottom, top_diff, weights,
                                       learning_rate=n,
                                       kernel_size=(11, 11),
                                       padding=(0, 0),
                                       stride=(4, 4))
            return bottom_diff

        bottom_diff = NDArray.rand((3, 3, 227, 227), np.float32) * 255
        bottom_diff.ocl_dirty = True
        bottom = NDArray.rand((3, 3, 227, 227), np.float32) * 255
        bottom.ocl_dirty = True
        weights = NDArray.rand((12, 363), np.float32)
        out = NDArray.zeros((3, 12, 3025), np.float32)

        out = fn(out, weights, bottom, bottom_diff)

        weights = weights.reshape(12, 3, 11, 11)
        expected = np.zeros((3, 12, 55, 55), np.float32)
        # reference_conv(a, weights, expected, (4, 4), (0, 0))
        out.sync()
        print(out)
        # np.testing.assert_array_almost_equal(out, expected.reshape(3, 12, 3025), decimal=2)
