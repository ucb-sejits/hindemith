from hmcaffe.layers.base_layer import Layer
from hindemith.types import hmarray
import numpy as np


class ConcatLayer(Layer):
    def __init__(self, layer_param, phase):
        self.phase = phase
        self.concat_axis = layer_param.concat_param.axis

    def set_up(self, *bottoms):
        self.bottoms = bottoms
        top_shape = list(self.bottoms[0].shape)
        # self.num_concats = np.prod(self.bottoms[0].shape[0:self.concat_axis])
        # concat_input_size = np.prod(bottom.shape[self.concat_axis + 1:])
        # bottom_count_sum = np.prod(self.bottoms[0].shape)
        for i in range(2, len(bottoms), 2):
            top_shape[self.concat_axis] += \
                self.bottoms[i].shape[self.concat_axis]

        self.top = hmarray.zeros(tuple(top_shape))

        return [(self.top, None)]

    def forward(self):
        for i in range(self.bottoms):
            bottom = self.bottoms[i]
            bottom.sync_host()
            for n in range(self.num_concats):
                self.top[n, i] = bottom[n]
