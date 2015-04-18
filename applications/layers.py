from hindemith.types import hmarray
from hindemith.core import compose
from hindemith.operations.conv import ConvForward, ConvBackward
from hindemith.operations.pool import PoolForward, PoolBackward
from hindemith.operations.relu import Relu
from hindemith.operations.lrn import LrnForward
from hindemith.operations.core import SoftmaxWithLossForward, \
    SoftmaxWithLossBackward
from hindemith.operations.softmax import SoftmaxForward
from hindemith.clibs.clblas import sgemm, sgemv
import numpy as np


class ConvLayer(object):
    def __init__(self, layer_param):
        conv_param = layer_param.convolution_param
        self.num_output = conv_param.num_output
        self.kernel_size = conv_param.kernel_size
        self.stride = conv_param.stride
        self.padding = conv_param.pad
        if conv_param.bias_filler == 'constant':
            self.bias = hmarray.zeros((num_output,))
            self.bias_diff = hmarray.zeros((num_output,))
            if conv_param.bias_filler.value != 0:
                self.bias.fill(conv_param.bias_filler.value)
                self.bias.sync_ocl()

        @compose
        def hm_backward(bottom_diff, bottom, top_diff, weights, weights_diff,
                        bias_diff, kernel_size, padding, stride):
            bottom_diff, weights_diff, bias_diff = \
                ConvBackward(bottom, top_diff, weights,
                             kernel_size=(kernel_size, kernel_size),
                             stride=(stride, stride), padding=(padding, padding))
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

    def set_up(self, bottom, bottom_diff=None):
        self.bottom = bottom
        self.bottom_diff = bottom_diff
        num, channels, height, width = bottom.shape

        weights_shape = (self.num_output, channels * self.kernel_size *
                         self.kernel_size)
        scale = 1.0 / np.sqrt(self.num_output)
        self.weights = hmarray.random(weights_shape, _range=(-scale, scale))
        self.weights_diff = hmarray.zeros(weights_shape)

        height_out = (height + 2 * self.padding - self.kernel_size) // \
            self.stride + 1
        width_out = (width + 2 * self.padding - self.kernel_size) // \
            self.stride + 1
        top_shape = (num, self.num_output, height_out, width_out)
        self.top = hmarray.zeros(top_shape)
        self.top_diff = hmarray.zeros(top_shape)
        return self.top, self.top_diff

    def forward(self):
        self.hm_forward(self.top, self.bottom, self.weights, self.bias,
                        self.kernel_size, self.padding, self.stride)

    def backward(self):
        self.hm_backward(self.bottom_diff, self.bottom, self.top_diff,
                         self.weights, self.weights_diff, self.bias_diff,
                         self.kernel_size, self.padding, self.stride)

    def update_weights(self):
        for buf, diff in [(self.weights, self.weights_diff), (self.bias, self.bias_diff)]:
            diff.sync_host()
            buf -= .01 * diff
            buf.sync_ocl()

class PoolingLayer(object):
    def __init__(self, layer_param):
        self.kernel_size = layer_param.pooling_param.kernel_size
        self.stride = layer_param.pooling_param.stride
        self.padding = layer_param.pooling_param.pad

        @compose
        def hm_forward(top, bottom, mask, kernel_size, padding, stride):
            top, mask = PoolForward(bottom,
                                    kernel_size=(kernel_size, kernel_size),
                                    padding=(padding, padding),
                                    stride=(stride, stride))
            return top, mask

        self.hm_forward = hm_forward

        @compose
        def hm_backward(bottom_diff, top_diff, mask, kernel_size, padding, stride):
            bottom_diff = PoolBackward(top_diff, mask,
                                       kernel_size=(kernel_size, kernel_size),
                                       padding=(padding, padding),
                                       stride=(stride, stride))
            return bottom_diff

        self.hm_backward = hm_backward

    def set_up(self, bottom, bottom_diff=None):
        self.bottom = bottom
        self.bottom_diff = bottom_diff
        num, channels, height, width = bottom.shape
        pooled_height = ((height + 2 * self.padding - self.kernel_size) //
                         self.stride) + 1
        pooled_width = ((width + 2 * self.padding - self.kernel_size) //
                        self.stride) + 1
        top_shape = num, channels, pooled_height, pooled_width
        self.mask = hmarray.zeros(top_shape)
        self.top = hmarray.zeros(top_shape)
        self.top_diff = hmarray.zeros(top_shape)
        return self.top, self.top_diff

    def forward(self):
        self.hm_forward(self.top, self.bottom, self.mask, self.kernel_size,
                        self.padding, self.stride)

    def backward(self):
        self.hm_backward(self.bottom_diff, self.top_diff, self.mask,
                         self.kernel_size, self.padding, self.stride)


class InnerProductLayer(object):
    def __init__(self, layer_param):
        self.num_output = layer_param.inner_product_param.num_output

    def set_up(self, bottom, bottom_diff=None):
        self.bottom, self.bottom_diff = bottom, bottom_diff
        N = self.num_output
        K = np.prod(bottom.shape[1:])
        scale = 1.0 / np.sqrt(self.num_output)
        self.weights = hmarray.random((N, K), _range=(-scale, scale))
        self.weights_diff = hmarray.zeros((N, K))
        self.bias = hmarray.zeros((self.num_output, ))
        self.bias_diff = hmarray.zeros((self.num_output, ))
        self.bias_multiplier = hmarray((1, self.bottom.shape[0]))
        self.bias_multiplier.fill(1)
        self.bias_multiplier.sync_ocl()
        top_shape = (bottom.shape[0], N)
        self.top = hmarray.zeros(top_shape)
        self.top_diff = hmarray.zeros(top_shape)
        return self.top, self.top_diff

    def forward(self):
        N = self.num_output
        K = np.prod(self.bottom.shape[1:])
        M = self.bottom.shape[0]
        self.top.fill(0)
        sgemm(False, True, 1.0, self.bottom, 0, K, self.weights, 0, K, 0.0,
              self.top, 0, N, M, N, K)
        sgemm(False, False, 1.0, self.bias_multiplier, 0, 1, self.bias, 0, N,
              1.0, self.top, 0, N, M, N, 1)

    def backward(self):
        N = self.num_output
        K = np.prod(self.bottom.shape[1:])
        M = self.bottom.shape[0]
        sgemm(True, False, 1.0, self.top_diff, 0, N, self.bottom, 0, K, 0.0,
              self.weights_diff, 0, K, N, K, M)
        sgemv(True, M, N, 1.0, self.top_diff, 0, N, self.bias_multiplier, 0, 1,
              0.0, self.bias_diff, 0, 1)
        sgemm(False, False, 1.0, self.top_diff, 0, N, self.weights, 0,
              K, 0.0, self.bottom_diff, 0, K, M, K, N)

    def update_weights(self):
        for buf, diff in [(self.weights, self.weights_diff), (self.bias, self.bias_diff)]:
            diff.sync_host()
            buf -= .01 * diff
            buf.sync_ocl()


class ReluLayer(object):
    def __init__(self, layer_param):
        @compose
        def hm_relu(bottom):
            bottom = Relu(bottom)
            return bottom

        self.hm_relu = hm_relu

    def set_up(self, bottom, bottom_diff=None):
        self.bottom, self.bottom_diff = bottom, bottom_diff
        return bottom, self.bottom_diff

    def forward(self):
        self.hm_relu(self.bottom)

    def backward(self):
        self.hm_relu(self.bottom_diff)


class SoftmaxWithLossLayer(object):
    def __init__(self):
        @compose
        def hm_forward(loss, bottom, label, prob):
            loss = SoftmaxWithLossForward(bottom, label, prob)
            return loss

        self.hm_forward = hm_forward

        @compose
        def hm_backward(bottom_diff, loss, label, prob):
            bottom_diff = SoftmaxWithLossBackward(loss, label, prob)
            return bottom_diff

        self.hm_backward = hm_backward

    def set_up(self, bottom, bottom_diff, label):
        self.bottom, self.bottom_diff, self.label = bottom, bottom_diff, label
        self.top = hmarray.zeros((1, ))
        self.prob = hmarray.zeros(bottom.shape)
        return self.top

    def forward(self):
        self.hm_forward(self.top, self.bottom, self.label, self.prob)

    def backward(self):
        self.hm_backward(self.bottom_diff, self.top, self.label, self.prob)


class SoftmaxLayer(object):
    def __init__(self, layer_param):
        @compose
        def hm_forward(top, bottom, label):
            top = SoftmaxForward(bottom, label)
            return top

        self.hm_forward = hm_forward

    def set_up(self, bottom, label):
        self.bottom, self.label = bottom, label
        self.top = hmarray.zeros(bottom.shape)
        return self.top

    def forward(self):
        self.hm_forward(self.top, self.bottom, self.label)


class LrnLayer(object):
    def __init__(self, layer_param):
        self.alpha = layer_param.lrn_param.alpha
        self.beta = layer_param.lrn_param.beta
        self.local_size = layer_param.lrn_param.local_size

        @compose
        def hm_forward(top, scale, bottom, alph, beta, local_size):
            top, scale = LrnForward(bottom, alpha=alpha, beta=beta,
                                    local_size=local_size, k=1)

        self.hm_forward = hm_forward

    def set_up(self, bottom, bottom_diff=None):
        self.bottom, self.bottom_diff = bottom, bottom_diff
        self.top = hmarray.zeros(bottom.shape)
        self.scale = hmarray.zeros(bottom.shape)
        self.top_diff = hmarray.zeros(bottom.shape)
        return self.top, self.top_diff

    def forward(self):
        self.hm_forward(self.top, self.scale, self.bottom, self.alpha,
                        self.beta, self.local_size)
