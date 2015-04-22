from hindemith.types import hmarray
from hindemith.clibs.clblas import sgemm, sgemv
import numpy as np


class InnerProductLayer(object):
    def __init__(self, layer_param, phase, params=None):
        self.phase = phase
        self.num_output = layer_param.inner_product_param.num_output
        if params is not None:
            self.weights = params[0].data.view(hmarray)
            self.weights.sync_ocl()
            self.bias = params[1].data.view(hmarray)
            self.bias.sync_ocl()
        else:
            self.weights = None
            self.bias = hmarray.zeros((self.num_output, ))

    def set_up(self, bottom, bottom_diff):
        self.bottom, self.bottom_diff = bottom, bottom_diff
        N = self.num_output
        K = np.prod(bottom.shape[1:])
        scale = 1.0 / np.sqrt(self.num_output)
        if self.weights is None:
            self.weights = hmarray.random((N, K), _range=(-scale, scale))
        self.weights_diff = hmarray.zeros((N, K))
        self.bias_diff = hmarray.zeros((self.num_output, ))
        self.bias_multiplier = hmarray((1, self.bottom.shape[0]))
        self.bias_multiplier.fill(1)
        self.bias_multiplier.sync_ocl()
        top_shape = (bottom.shape[0], N)
        self.top = hmarray.zeros(top_shape)
        self.top_diff = hmarray.zeros(top_shape)
        return [(self.top, self.top_diff)]

    def forward(self):
        N = self.num_output
        K = np.prod(self.bottom.shape[1:])
        M = self.bottom.shape[0]
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
        for buf, diff in [(self.weights, self.weights_diff),
                          (self.bias, self.bias_diff)]:
            diff.sync_host()
            buf -= .01 * diff
            buf.sync_ocl()
