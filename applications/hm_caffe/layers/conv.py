from hindemith.operations.conv import ConvForward, ConvBackward
from hindemith.types import hmarray
from hindemith.core import compose
import numpy as np


class ConvLayer(object):
    def __init__(self, layer_param, phase, params=None):
        self.phase = phase
        conv_param = layer_param.convolution_param
        self.num_output = conv_param.num_output
        self.kernel_size = conv_param.kernel_size
        self.stride = conv_param.stride
        self.padding = conv_param.pad
        self.conv_param = conv_param
        if params is not None:
            self.weights = params[0].data.view(hmarray)
            self.weights.sync_ocl()
            self.bias = params[1].data.view(hmarray)
            self.bias.sync_ocl()
        else:
            self.weights = None
            self.bias = None
        # if conv_param.bias_filler.type == 'constant':
        #     self.bias = hmarray.zeros((self.num_output,))
        #     self.bias_diff = hmarray.zeros((self.num_output,))
        #     if conv_param.bias_filler.value != 0:
        #         self.bias.fill(conv_param.bias_filler.value)
        #         self.bias.sync_ocl()

        @compose
        def hm_backward(bottom_diff, bottom, top_diff, weights, weights_diff,
                        bias_diff, kernel_size, padding, stride):
            bottom_diff, weights_diff, bias_diff = \
                ConvBackward(bottom, top_diff, weights,
                             kernel_size=(kernel_size, kernel_size),
                             stride=(stride, stride),
                             padding=(padding, padding))
            return bottom_diff, weights_diff, bias_diff
        self.hm_backward = hm_backward

        @compose
        def hm_forward(top, bottom, weights, bias, kernel_size, padding,
                       stride):
            top = ConvForward(bottom, weights, bias,
                              kernel_size=(kernel_size, kernel_size),
                              padding=(padding, padding),
                              stride=(stride, stride))
            return top
        self.hm_forward = hm_forward

    def set_up(self, bottom, bottom_diff):
        self.bottom = bottom
        self.bottom_diff = bottom_diff
        num, channels, height, width = bottom.shape

        weights_shape = (self.num_output, channels * self.kernel_size *
                         self.kernel_size)
        if self.weights is not None:
            self.weights = self.weights.reshape(weights_shape)
        else:
            n = 1.0 / np.sqrt(self.num_output)
            self.weights = hmarray.random(weights_shape, _range=(-n, n))
        self.weights_diff = hmarray.zeros(weights_shape)

        height_out = (height + 2 * self.padding - self.kernel_size) // \
            self.stride + 1
        width_out = (width + 2 * self.padding - self.kernel_size) // \
            self.stride + 1
        top_shape = (num, self.num_output, height_out, width_out)
        self.top = hmarray.zeros(top_shape)
        self.top_diff = hmarray.zeros(top_shape)
        return [(self.top, self.top_diff)]

    def forward(self):
        self.hm_forward(self.top, self.bottom, self.weights, self.bias,
                        self.kernel_size, self.padding, self.stride)

    def backward(self):
        self.hm_backward(self.bottom_diff, self.bottom, self.top_diff,
                         self.weights, self.weights_diff, self.bias_diff,
                         self.kernel_size, self.padding, self.stride)

    def update_weights(self):
        for buf, diff in [(self.weights, self.weights_diff),
                          (self.bias, self.bias_diff)]:
            diff.sync_host()
            buf -= .01 * diff
            buf.sync_ocl()
