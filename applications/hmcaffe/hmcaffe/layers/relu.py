from hmcaffe.layers.base_layer import Layer
from hindemith.operations.relu import ReluForward, ReluBackward
from hindemith.core import compose


class ReluLayer(Layer):
    def __init__(self, layer_param, phase):
        self.phase = phase
        self.layer_param = layer_param

        @compose
        def hm_forward(bottom):
            bottom = ReluForward(bottom)
            return bottom

        self.hm_forward = hm_forward

        @compose
        def hm_backward(bottom, bottom_diff):
            bottom_diff = ReluBackward(bottom, bottom_diff)
            return bottom_diff

        self.hm_backward = hm_backward

    def set_up(self, bottom, bottom_diff):
        self.bottom, self.bottom_diff = bottom, bottom_diff
        return [(self.bottom, self.bottom_diff)]

    def forward(self):
        self.hm_forward(self.bottom)

    def backward(self):
        self.hm_backward(self.bottom, self.bottom_diff)
