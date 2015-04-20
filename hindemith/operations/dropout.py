from hindemith.operations.core import ElementLevel
import numpy as np
from string import Template


class Dropout(ElementLevel):
    """
    top = Dropout(bottom, mask)
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        num_work_items = np.prod(sources[0].shape)
        return (num_work_items, )

    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        threshold = keywords['threshold']
        scale = 1.0 / (1.0 - threshold)
        return Template(
            "$target[get_global_id(0)] = $operand[get_global_id(0)] *"
            "$mask[get_global_id(0)] * (float) $scale;"
        ).substitute(target=sinks[0], operand=sources[0], mask=sources[1],
                     scale=scale)
