from hindemith.types import hmarray


class AccuracyLayer(object):
    def __init__(self, layer_param, phase):
        self.phase = phase

    def set_up(self, bottom, bottom_diff, label, label_diff):
        self.bottom = bottom
        self.bottom_diff = bottom_diff
        self.top = hmarray.zeros(bottom.shape)
        if self.phase == 'TRAIN':
            self.top_diff = hmarray.zeros(bottom.shape)
        else:
            self.top_diff = None
        return [(self.top, self.top_diff)]
