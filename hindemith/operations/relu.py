from hindemith.operations.core import ElementLevel
import numpy as np
from string import Template


class Relu(ElementLevel):
    """
    top = Relu(bottom)
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        num_work_items = np.prod(sources[0].shape)
        return (num_work_items, )

    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        return Template(
            "$target[get_global_id(0)] = $operand[get_global_id(0)] > 0 ? "
            "$operand[get_global_id(0)] : 0;"
        ).substitute(target=sinks[0], operand=sources[0])

