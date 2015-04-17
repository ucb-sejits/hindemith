from hindemith.operations.core import BlockLevel
import numpy as np
from string import Template


class PoolForward(BlockLevel):
    """
    top, mask = PoolForward(bottom)
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        num_work_items = np.prod(sinks[0].shape)
        return (num_work_items, )

    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
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


class PoolBackward(BlockLevel):
    """
    bottom_diff = PoolBackward(top_diff, mask
                               kernel_size=(2, 2),
                               padding=(0, 0),
                               stride=(2, 2))
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        num_work_items = np.prod(sinks[0].shape)
        return (num_work_items, )

    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        channels, height, width = symbol_table[sinks[0]].shape[1:]
        pad_h, pad_w = keywords['padding']
        stride_h, stride_w = keywords['stride']
        kernel_h, kernel_w = keywords['kernel_size']
        pooled_height = ((height + 2 * pad_h - kernel_h) // stride_h) + 1
        pooled_width = ((width + 2 * pad_w - kernel_w) // stride_w) + 1
        return Template("""
    int index = get_global_id(0);
    int w = index % $width;
    int h = (index / $width) % $height;
    int c = (index / $width / $height) % $channels;
    int n = index / $width / $height / $channels;
    int phstart =
        (h + $pad_h < $kernel_h) ? 0 : (h + $pad_h - $kernel_h) / $stride_h + 1;
    int phend = min((h + $pad_h) / $stride_h + 1, $pooled_height);
    int pwstart =
        (w + $pad_w < $kernel_w) ? 0 : (w + $pad_w - $kernel_w) / $stride_w + 1;
    int pwend = min((w + $pad_w) / $stride_w + 1, $pooled_width);
    float gradient = 0;
    int offset = (n * $channels + c) * $pooled_height * $pooled_width;
    $top_diff += offset;
    $mask += offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if ($mask[ph * $pooled_width + pw] == h * $width + w) {
          gradient += $top_diff[ph * $pooled_width + pw];
        }
      }
    }
    $bottom_diff[index] = gradient;        
""").substitute(bottom_diff=sinks[0], mask=sources[1], top_diff=sources[0],
                pooled_height=pooled_height, pooled_width=pooled_width,
                channels=channels, stride_h=stride_h,
                stride_w=stride_w, pad_h=pad_h, pad_w=pad_w,
                kernel_h=kernel_h, kernel_w=kernel_w, height=height,
                width=width)
