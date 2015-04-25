from hmcaffe.layers.base_layer import Layer
from hindemith.operations.softmax_with_loss import SoftmaxWithLossForward, \
    SoftmaxWithLossBackward
from hindemith.types import hmarray
from hindemith.core import compose


class SoftmaxWithLossLayer(Layer):
    def __init__(self, layer_param, phase):
        self.phase = phase

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

    def set_up(self, bottom, bottom_diff, label, label_diff):
        self.bottom, self.bottom_diff, self.label = bottom, bottom_diff, label
        self.top = hmarray.zeros((1, ))
        self.prob = hmarray.zeros(bottom.shape)
        return [(self.top, None)]

    def forward(self):
        self.hm_forward(self.top, self.bottom, self.label, self.prob)

    def backward(self):
        self.hm_backward(self.bottom_diff, self.top, self.label, self.prob)
