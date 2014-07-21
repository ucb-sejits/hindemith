from _ctypes import sizeof
from ctree import get_ast
from hindemith.operations.dense_linear_algebra import Float32, Int, Scalar, Array
from hindemith.types.stencil import Stencil
from numpy import ndarray
from hindemith.utils import UnsupportedTypeError

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


def fuse(fn):
    def fused_fn(*args, **kwargs):
        coerced_args = {}
        for name, value in map(coercer, kwargs.items()):
            coerced_args[name] = value
        return fn(**coerced_args)
    return fused_fn

