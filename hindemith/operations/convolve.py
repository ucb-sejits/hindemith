__author__ = 'leonardtruong'

from hindemith.operations.core import ElementLevel
import numpy as np
from string import Template


class Convolve2D(ElementLevel):
    """
    output = Convolve(input, filter)
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        num_work_items = np.prod(sinks[0].shape)
        return (num_work_items, )

    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        height, width = symbol_table[sources[0]].shape
        kernel_h, kernel_w = symbol_table[sources[1]].shape
        kernel_str = """
        {
        int x = index % $width;
        int y = index / $width;
        float accum = 0.0;
        """
        for i in range(kernel_h):
            for j in range(kernel_w):
                kernel_str += """
                accum += {0}f * $input[min(max(y + {1}, 0), $height - 1) * $width + min(max(x + {2}, 0), $width - 1)];
                """.format(symbol_table[sources[1]][i, j], i - (kernel_h // 2), j - (kernel_w // 2))
        kernel_str += """
            $output[index] = accum;
        }"""
        return Template(
            kernel_str
        ).substitute(output=sinks[0], input=sources[0], filter=sources[1],
                     height=height, width=width, kernel_h=kernel_h, kernel_w=kernel_w)
