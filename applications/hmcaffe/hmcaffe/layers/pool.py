from hmcaffe.layers.base_layer import Layer
from hindemith.types import hmarray
from hindemith.core import compose
from hindemith.operations.pool import PoolForward, PoolBackward


class PoolingLayer(Layer):
    def __init__(self, layer_param, phase):
        self.layer_param = layer_param
        self.kernel_size = layer_param.pooling_param.kernel_size
        self.stride = layer_param.pooling_param.stride
        self.padding = layer_param.pooling_param.pad
        self.phase = phase

        @compose
        def hm_forward(top, bottom, mask, kernel_size, padding, stride):
            top, mask = PoolForward(bottom,
                                    kernel_size=(kernel_size, kernel_size),
                                    padding=(padding, padding),
                                    stride=(stride, stride))
            return top, mask

        self.hm_forward = hm_forward

        @compose
        def hm_backward(bottom_diff, top_diff, mask, kernel_size, padding,
                        stride):
            bottom_diff = PoolBackward(top_diff, mask,
                                       kernel_size=(kernel_size, kernel_size),
                                       padding=(padding, padding),
                                       stride=(stride, stride))
            return bottom_diff

        self.hm_backward = hm_backward

    def set_up(self, bottom, bottom_diff):
        self.bottom = bottom
        self.bottom_diff = bottom_diff
        num, channels, height, width = bottom.shape
        pooled_height = ((height + 2 * self.padding - self.kernel_size) //
                         self.stride) + 1
        pooled_width = ((width + 2 * self.padding - self.kernel_size) //
                        self.stride) + 1
        top_shape = num, channels, pooled_height, pooled_width
        self.mask = hmarray.zeros(top_shape)
        self.top = hmarray.zeros(top_shape)
        self.top_diff = hmarray.zeros(top_shape)
        return [(self.top, self.top_diff)]

    def forward(self):
        self.hm_forward(self.top, self.bottom, self.mask, self.kernel_size,
                        self.padding, self.stride)

    def backward(self):
        self.hm_backward(self.bottom_diff, self.top_diff, self.mask,
                         self.kernel_size, self.padding, self.stride)
