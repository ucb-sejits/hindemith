from hmcaffe.layers.base_layer import Layer
from hindemith.operations.lrn import LrnForward, LrnBackward
from hindemith.types import hmarray
from hindemith.core import compose


class LrnLayer(Layer):
    def __init__(self, layer_param, phase):
        self.layer_param = layer_param
        self.phase = phase
        self.alpha = layer_param.lrn_param.alpha
        self.beta = layer_param.lrn_param.beta
        self.local_size = layer_param.lrn_param.local_size

        @compose
        def hm_forward(top, scale, bottom, alpha, beta, local_size):
            top, scale = LrnForward(bottom, alpha=alpha, beta=beta,
                                    local_size=local_size, k=1)

        @compose
        def hm_backward(bottom_diff, bottom, top, scale, top_diff, alpha, beta,
                        local_size):
            bottom_diff = LrnBackward(bottom, top, scale, top_diff,
                                      alpha=alpha, beta=beta,
                                      local_size=local_size)
            return bottom_diff

        self.hm_forward = hm_forward
        self.hm_backward = hm_backward

    def set_up(self, bottom, bottom_diff):
        self.bottom, self.bottom_diff = bottom, bottom_diff
        self.top = hmarray.zeros(bottom.shape)
        self.scale = hmarray.zeros(bottom.shape)
        self.top_diff = hmarray.zeros(bottom.shape)
        return [(self.top, self.top_diff)]

    def forward(self):
        self.hm_forward(self.top, self.scale, self.bottom, self.alpha,
                        self.beta, self.local_size)

    def backward(self):
        self.hm_backward(self.bottom_diff, self.bottom, self.top, self.scale,
                         self.top_diff, self.alpha, self.beta, self.local_size)
