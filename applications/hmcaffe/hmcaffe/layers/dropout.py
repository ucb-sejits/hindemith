from hmcaffe.layers.base_layer import Layer
from hindemith.operations.dropout import Dropout
from hindemith.types import hmarray
from hindemith.core import compose


class DropoutLayer(Layer):
    def __init__(self, layer_param, phase):
        self.phase = phase
        self.threshold = layer_param.dropout_param.dropout_ratio

        @compose
        def hm_dropout(top, bottom, mask, threshold):
            top = Dropout(bottom, mask, threshold=threshold)
            return top

        self.hm_dropout = hm_dropout

    def set_up(self, bottom, bottom_diff, params=None):
        self.bottom = bottom
        self.bottom_diff = bottom_diff
        self.top = hmarray.zeros(bottom.shape)
        self.mask = hmarray.random(bottom.shape)
        self.top_diff = hmarray.zeros(bottom.shape)
        return [(self.top, self.top_diff)]

    def forward(self):
        self.hm_dropout(self.top, self.bottom, self.mask, self.threshold)

    def backward(self):
        self.hm_dropout(self.bottom_diff, self.top_diff, self.mask,
                        self.threshold)
