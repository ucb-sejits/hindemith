import numpy as np
from string import Template


class HMOperation(object):
    pass


class BlockLevel(HMOperation):
    pass


class ElementLevel(HMOperation):
    pass


class Relu(ElementLevel):
    @staticmethod
    def get_launch_parameters(sources, sinks):
        num_work_items = np.prod(sources[0].shape)
        return (num_work_items, )

    @staticmethod
    def emit(sources, sinks, keywords, symbol_table):
        return Template(
            "$target[get_global_id(0)] = $operand[get_global_id(0)] > 0 ? "
            "$operand[get_global_id(0)] : 0;"
        ).substitute(target=sinks[0], operand=sources[0])


class PoolForward(BlockLevel):
    @staticmethod
    def get_launch_parameters(sources, sinks):
        num_work_items = np.prod(sinks[0].shape)
        return (num_work_items, )

    @staticmethod
    def emit(sources, sinks, keywords, symbol_table):
        channels, height, width = symbol_table[sources[0]].shape[1:]
        pad_h, pad_w = keywords['padding']
        stride_h, stride_w = keywords['stride']
        kernel_h, kernel_w = keywords['kernel_size']
        pooled_height = ((height + 2 * pad_h - kernel_h) // stride_h) + 1
        pooled_width = ((width + 2 * pad_w - kernel_w) // stride_w) + 1
        return Template("""
    int index = get_global_id(0);
    int pw = index % $pooled_w;
    int ph = (index / $pooled_w) % $pooled_h;
    int c = (index / $pooled_w / $pooled_h) % $channels;
    int n = index / $pooled_w / $pooled_h / $channels;
    int hstart = ph * $stride - $pad;
    int wstart = pw * $stride - $pad;
    int hend = min(hstart + $kernel_h, $height);
    int wend = min(wstart + $kernel_w, $width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    float maxval = -FLT_MAX;
    int maxidx = -1;
    $bottom += (n * $channels + c) * $height * $width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if ($bottom[h * $width + w] > maxval) {
          maxidx = h * $width + w;
          maxval = $bottom[maxidx];
        }
      }
    }
    $top[index] = maxval;
    $mask[index] = maxidx;
""").substitute(top=sinks[0], mask=sinks[1], bottom=sources[0],
                pooled_h=pooled_height, pooled_w=pooled_width,
                channels=channels, stride=stride_h, pad=pad_h,
                kernel_h=kernel_h, kernel_w=kernel_w,
                height=height, width=width)
