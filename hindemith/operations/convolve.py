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
        height, width = symbol_table[sources[0].name].shape
        kernel_h, kernel_w = symbol_table[sources[1].name].shape
        kernel_str = """
        {
        int x = index % $width;
        int y = index / $width;
        float accum = 0.0;
        """
        for i in range(kernel_h):
            y_off = i - (kernel_h // 2)
            if y_off < 0:
                y_index = "max(y + {}, 0)".format(y_off)
            elif y_off > 0:
                y_index = "min(y + {}, $height - 1)".format(y_off)
            else:
                y_index = "y"

            for j in range(kernel_w):
                weight = symbol_table[sources[1].name][i, j]
                if weight == 0:
                    continue
                x_off = j - (kernel_w // 2)
                if x_off < 0:
                    x_index = "max(x + {}, 0)".format(x_off)
                elif x_off > 0:
                    x_index = "min(x + {}, $width - 1)".format(x_off)
                else:
                    x_index = "x"
                kernel_str += """
                accum += {0}f * $input[{1} * $width + {2}];
                """.format(weight, y_index, x_index)
        kernel_str += """
            $output = accum;
        }"""
        return Template(
            kernel_str
        ).substitute(output=sinks[0].get_element(), input=sources[0].name,
                     height=height, width=width, kernel_h=kernel_h, kernel_w=kernel_w)
