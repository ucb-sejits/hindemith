import numpy as np
from string import Template


class HMOperation(object):
    pass


class ElementLevel(HMOperation):
    pass


class Relu(ElementLevel):
    @staticmethod
    def get_launch_parameters(sources, sinks):
        num_work_items = np.prod(sources[0].shape)
        return (num_work_items, )

    @staticmethod
    def emit(sources, sinks):
        return Template(
            "$target[get_global_id(0)] = $operand[get_global_id(0)] > 0 ? "
            "$operand[get_global_id(0)] : 0;"
        ).substitute(target=sinks[0], operand=sources[0])
