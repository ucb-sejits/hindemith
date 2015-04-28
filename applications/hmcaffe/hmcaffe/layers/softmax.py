from hmcaffe.layers.base_layer import Layer
from hindemith.operations.softmax import SoftmaxForward
from hindemith.types import hmarray
from hindemith.core import compose


class SoftmaxLayer(Layer):
    def __init__(self, layer_param, phase):
        self.layer_param = layer_param
        self.phase = phase

        @compose
        def hm_forward(top, bottom):
            top = SoftmaxForward(bottom)
            return top

        self.hm_forward = hm_forward

    def set_up(self, bottom, bottom_diff):
        self.bottom = bottom
        self.top = hmarray.zeros(bottom.shape)
        return [(self.top, None)]

    def forward(self):
        self.hm_forward(self.top, self.bottom)
