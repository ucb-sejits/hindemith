from hindemith.operations.core import ElementLevel
import numpy as np
from string import Template


class ReluForward(ElementLevel):
    """
    top = ReluForward(bottom)
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        num_work_items = np.prod(sources[0].shape)
        return (num_work_items, )

    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        return Template(
            "$target[index] = $operand[index] > 0 ? "
            "$operand[index] : 0;"
        ).substitute(target=sinks[0], operand=sources[0])


class ReluBackward(ElementLevel):
    """
    bottom_diff = ReluBackward(bottom, top_diff)
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        num_work_items = np.prod(sources[0].shape)
        return (num_work_items, )

    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        return Template(
            "$bottom_diff[index] = $top_diff[index] * "
            "(($bottom[index] > 0)"
            " + $negative_slope * ($bottom[index] <= 0));"
        ).substitute(bottom_diff=sinks[0], bottom=sources[0],
                     top_diff=sources[1], negative_slope=0)
