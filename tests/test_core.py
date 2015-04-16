import unittest
from hindemith.types import hmarray
from hindemith.core import compose
from hindemith.operations import Relu, PoolForward, ConvForward, ConvBackward, \
    PoolBackward
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
        actual_mask.sync_host()
        reference_pool(a, expected, expected_mask, (2, 2), (2, 2), (0, 0))
        self._check(actual, expected)
        self._check(actual_mask, expected_mask)
        bottom_diff = hmarray.zeros(shape, np.float32)
        expected_bottom_diff = hmarray.zeros(shape)
        mask = actual_mask
        top_diff = hmarray.random((3, 16, 12, 12))

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

    def test_conv_forward(self):
        @compose
        def fn(a, weights, out, bias):
            out = ConvForward(a, weights, bias, kernel_size=(11, 11),
                              padding=(0, 0), stride=(4, 4))
            return out

        a = hmarray.random((3, 3, 27, 27), _range=(0, 255))
        weights = hmarray.random((12, 363))
        out = hmarray.zeros((3, 12, 25))
        bias = hmarray((12, ))
        bias.fill(1)
        bias.sync_ocl()

        fn(a, weights, out, bias)

        weights = weights.reshape(12, 3, 11, 11)
        expected = np.zeros((3, 12, 5, 5), np.float32)
        reference_conv(a, weights, bias, expected, (4, 4), (0, 0))
        out.sync_host()
        np.testing.assert_array_almost_equal(out, expected.reshape(3, 12, 25),
                                             decimal=2)

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

        top_diff = hmarray.random((3, 12, 25), _range=(0, 5))
        bottom = hmarray.random((3, 3, 27, 27), _range=(0, 255))
        weights = hmarray.random((12, 363), _range=(-.2, .2))

        weights_diff = hmarray.zeros((12, 363))
        bias_diff = hmarray.zeros((12, ))
        bottom_diff = hmarray.zeros((3, 3, 27, 27))

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
