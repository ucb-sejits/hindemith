from _ctypes import sizeof
from teller.types.common import Float32, Int, Scalar
from teller.types.stencil import Stencil
from teller.operations.dense_linear_algebra.array_op import Array
from numpy import ndarray
from teller.utils import UnsupportedTypeError

__author__ = 'leonardtruong'


def coercer(arg):
    name, value = arg
    if isinstance(value, float):
        return name, Float32(name, value)
    elif isinstance(value, int):
        return name, Int(name, value)
    elif isinstance(value, Stencil):
        value.name = name
        return name, value
    elif isinstance(value, Array):
        value.name = name
        return name, value
    elif isinstance(value, Scalar):
        value.name = name
        return name, value
    elif isinstance(value, ndarray):
        return name, Array(name, value)
    else:
        raise UnsupportedTypeError("Teller found unsupported type: {0}".format(type(value)))


def hm(fn):
    def hm_fn(*args, **kwargs):
        coerced_args = {}
        for name, value in map(coercer, kwargs.items()):
            coerced_args[name] = value
        return fn(**coerced_args)
    return hm_fn

