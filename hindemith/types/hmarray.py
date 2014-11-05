import numpy as np
from hindemith.linalg import add, sub, mul, div


def __add__(self, other):
    return add(self, other)


def __sub__(self, other):
    return sub(self, other)


def __mul__(self, other):
    return mul(self, other)


def __div__(self, other):
    return div(self, other)


class hmarray(np.ndarray):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        if isinstance(shape, np.ndarray):
            obj = np.asarray(shape).view(subtype)
        else:
            obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset,
                                     strides, order)
        subtype.__add__ = __add__
        subtype.__sub__ = __sub__
        subtype.__mul__ = __mul__
        subtype.__div__ = __div__
        return obj
