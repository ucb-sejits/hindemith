from hindemith.operations.relu import Relu
from hindemith.core import compose


class ReluLayer(object):
    def __init__(self, layer_param, phase):
        self.phase = phase

        @compose
        def hm_relu(bottom):
            bottom = Relu(bottom)
            return bottom

        self.hm_relu = hm_relu

    def set_up(self, bottom, bottom_diff):
        self.bottom, self.bottom_diff = bottom, bottom_diff
        return [(self.bottom, self.bottom_diff)]

    def forward(self):
        self.hm_relu(self.bottom)

    def backward(self):
        self.hm_relu(self.bottom_diff)
