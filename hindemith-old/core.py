from hindemith.types.common import Float32, Int, Scalar, Array
from hindemith.utils import UnsupportedTypeError
from numpy import ndarray

import logging
LOG = logging.getLogger('Hindemith')

__author__ = 'leonardtruong'


def coercer(arg):
    name, value = arg
    if isinstance(value, float):
        return name, Float32(name, value)
    elif isinstance(value, int):
        return name, Int(name, value)
    elif isinstance(value, Array):
        value.name = name
        return name, value
    elif isinstance(value, Scalar):
        value.name = name
        return name, value
    elif isinstance(value, ndarray):
        return name, Array(name, value)
    else:
        raise UnsupportedTypeError(
            "Hindemith found unsupported type: {0}".format(type(value))
        )
