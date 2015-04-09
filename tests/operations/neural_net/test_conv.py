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


def reference_im2col(data, kernel_size, stride, padding):
    channels, height, width = data.shape
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    height_col = (height + 2 * pad_h - kernel_h) // stride_h + 1
    width_col = (width + 2 * pad_w - kernel_w) // stride_w + 1
    channels_col = channels * kernel_h * kernel_w
    data_col = NDArray.zeros((channels_col, height_col * width_col), np.float32)
    for c in range(channels_col):
        w_offset = c % kernel_w
        h_offset = (c / kernel_w) % kernel_h
        c_im = c / kernel_h / kernel_w
        for h in range(height_col):
            for w in range(width_col):
                h_pad = h * stride_h - pad_h + h_offset
                w_pad = w * stride_w - pad_w + w_offset
                if (h_pad >= 0 and h_pad < height and w_pad >= 0 and w_pad < width):
                    data_col[c, h * width_col] = data[c_im, h_pad, w_pad];
                else:
                    data_col[c, h * width_col + w] = 0;
    return data_col


class TestConv(unittest.TestCase):
    def test_forward(self):
        @hm
        def fn(a, weights, out):
            out = ConvForward(a, weights, kernel_size=(11, 11), padding=(0, 0), stride=(4, 4))
            return out

        a = NDArray.rand((3, 3, 27, 27), np.float32) * 255
        a.ocl_dirty = True
        weights = NDArray.rand((12, 363), np.float32)
        out = NDArray.zeros((3, 12, 25), np.float32)

        out = fn(a, weights, out)

        weights = weights.reshape(12, 3, 11, 11)
        expected = np.zeros((3, 12, 5, 5), np.float32)
        reference_conv(a, weights, expected, (4, 4), (0, 0))
        out.sync()
        np.testing.assert_array_almost_equal(out, expected.reshape(3, 12, 25), decimal=2)


    @unittest.skip("Not working yet")
    def test_backward(self):
        @hm
        def fn(top_diff, weights, bottom, bottom_diff, weights_diff):
            bottom_diff = ConvBackward(bottom, top_diff, weights, weights_diff,
                                       kernel_size=(11, 11),
                                       padding=(0, 0),
                                       stride=(4, 4))
            return bottom_diff

        top_diff = NDArray.rand((3, 12, 3025), np.float32)
        top_diff.ocl_dirty = True
        bottom = NDArray.rand((3, 3, 227, 227), np.float32) * 255
        bottom.ocl_dirty = True
        weights = NDArray.rand((12, 363), np.float32)

        weights_diff = NDArray.zeros((12, 363), np.float32)
        bottom_diff = NDArray.zeros((3, 3, 227, 227), np.float32)

        fn(top_diff, weights, bottom, bottom_diff, weights_diff)

        expected_weights_diff = np.zeros_like(weights)
        expected = np.zeros((3, 12, 55, 55), np.float32)

        expected_bottom_diff = np.zeros_like(bottom_diff)
        for i in range(top_diff.shape[0]):
            col_data = reference_im2col(bottom[i], (11, 11), (4, 4), (0, 0))
            expected_weights_diff += top_diff[i].dot(col_data.T)
        weights_diff.sync()
        np.testing.assert_array_almost_equal(weights_diff, expected_weights_diff, decimal=2)
        # np.testing.assert_array_almost_equal(out, expected.reshape(3, 12, 3025), decimal=2)
