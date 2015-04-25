from hmcaffe.layers.base_layer import Layer
from hindemith.types import hmarray
import numpy as np


class AccuracyLayer(Layer):
    def __init__(self, layer_param, phase):
        self.phase = phase
        self.top_k = layer_param.accuracy_param.top_k

    def set_up(self, bottom, bottom_diff, label, _):
        self.bottom = bottom
        self.bottom_diff = bottom_diff
        self.label = label
        self.top = hmarray.zeros((1,))
        return [(self.top, None)]

    def forward(self):
        self.bottom.sync_host()
        accuracy = 0
        for n in range(self.bottom.shape[0]):
            actual_label = self.label[n]
            # Get the index top_k entries, (which are the top_k labels)
            top_k = np.argpartition(self.bottom[n], -self.top_k)[-self.top_k:]
            if actual_label in top_k:
                accuracy += 1
        self.top[0] = accuracy / self.bottom.shape[0]
