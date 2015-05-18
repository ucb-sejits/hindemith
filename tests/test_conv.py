import unittest
import numpy as np
from hindemith.core import compose
from hindemith.operations.conv import ConvBackward, ConvForward
from hindemith.types import hmarray
import hindemith as hm
import os


def reference_conv(in_data, weights, bias, out, stride, pad):
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
                                if (in_y >= 0 and in_y < in_data.shape[2] and
                                        in_x >= 0 and in_x < in_data.shape[3]):
                                    out[n, out_c, y, x] += \
                                        in_data[n, in_c, in_y, in_x] * \
                                        weights[out_c, in_c, p, q]
    for n in range(out.shape[0]):
        for o in range(out.shape[1]):
            for y in range(out.shape[2]):
                for x in range(out.shape[3]):
                    out[n, o, y, x] += bias[o]


def reference_im2col(data, kernel_size, stride, padding):
    channels, height, width = data.shape
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    height_col = (height + 2 * pad_h - kernel_h) // stride_h + 1
    width_col = (width + 2 * pad_w - kernel_w) // stride_w + 1
    channels_col = channels * kernel_h * kernel_w
    data_col = hmarray.zeros((channels_col, height_col * width_col),
                             np.float32)
    for c in range(channels_col):
        w_offset = c % kernel_w
        h_offset = (c / kernel_w) % kernel_h
        c_im = c / kernel_h / kernel_w
        for h in range(height_col):
            for w in range(width_col):
                h_pad = h * stride_h - pad_h + h_offset
                w_pad = w * stride_w - pad_w + w_offset
                if (h_pad >= 0 and h_pad < height and
                        w_pad >= 0 and w_pad < width):
                    data_col[c, h * width_col + w] = data[c_im, h_pad, w_pad]
                else:
                    data_col[c, h * width_col + w] = 0
    return data_col


def reference_col2im(data_col, kernel_size, stride, padding, shape):
    channels, height, width = shape
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    height_col = (height + 2 * pad_h - kernel_h) // stride_h + 1
    width_col = (width + 2 * pad_w - kernel_w) // stride_w + 1
    channels_col = channels * kernel_h * kernel_w
    data = hmarray.zeros(shape)
    for c in range(channels_col):
        w_offset = c % kernel_w
        h_offset = (c / kernel_w) % kernel_h
        c_im = c / kernel_h / kernel_w
        for h in range(height_col):
            for w in range(width_col):
                h_pad = h * stride_h - pad_h + h_offset
                w_pad = w * stride_w - pad_w + w_offset
                if (h_pad >= 0 and h_pad < height and w_pad >= 0 and w_pad <
                        width):
                    data[c_im, h_pad, w_pad] += data_col[c, h * width_col + w]
    return data


class TestConv(unittest.TestCase):
    def test_conv_forward(self):
        @compose
        def fn(a, weights, out, bias):
            out = ConvForward(a, weights, bias, kernel_size=(11, 11),
                              padding=(0, 0), stride=(4, 4))
            return out

        a = hm.random((3, 3, 27, 27), _range=(0, 255))
        weights = hm.random((12, 363), _range=(-.2, .2))
        out = hm.zeros((3, 12, 5, 5))
        bias = hm.random((12, ))
        fn(a, weights, out, bias)

        weights = weights.reshape(12, 3, 11, 11)
        expected = np.zeros((3, 12, 5, 5), np.float32)
        reference_conv(a, weights, bias, expected, (4, 4), (0, 0))
        out.sync_host()
        np.testing.assert_array_almost_equal(out, expected,
                                             decimal=1)

    # @unittest.skipUnless(os.getenv("HM_BACKEND", "ocl") == "ocl", "Not implemented for openmp")
    @unittest.skip("Broken")
    def test_conv_backward(self):
        # TODO: Check bias diff

        @compose
        def fn(top_diff, weights, bottom, bottom_diff, weights_diff,
               bias_diff):
            bottom_diff, weights_diff, bias_diff = \
                ConvBackward(bottom, top_diff, weights,
                             kernel_size=(11, 11), padding=(0, 0),
                             stride=(4, 4))
            return bottom_diff, weights_diff, bias_diff

        top_diff = hm.random((3, 12, 25), _range=(0, 5))
        bottom = hm.random((3, 3, 27, 27), _range=(0, 255))
        weights = hm.random((12, 363), _range=(-.2, .2))

        weights_diff = hm.zeros((12, 363))
        bias_diff = hm.zeros((12, ))
        bottom_diff = hm.zeros((3, 3, 27, 27))

        fn(top_diff, weights, bottom, bottom_diff, weights_diff, bias_diff)

        expected_weights_diff = np.zeros_like(weights)
        expected_bottom_diff = np.zeros_like(bottom_diff)

        for i in range(top_diff.shape[0]):
            col_data = reference_im2col(bottom[i], (11, 11), (4, 4), (0, 0))
            expected_weights_diff += top_diff[i].dot(col_data.T)
            expected_bottom_diff[i] = reference_col2im(
                weights.T.dot(top_diff[i]), (11, 11), (4, 4), (0, 0),
                expected_bottom_diff[i].shape)
        weights_diff.sync_host()
        np.testing.assert_array_almost_equal(weights_diff,
                                             expected_weights_diff, decimal=2)
        bottom_diff.sync_host()
        np.testing.assert_array_almost_equal(bottom_diff, expected_bottom_diff,
                                             decimal=2)
