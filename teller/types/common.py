
__author__ = 'leonardtruong'

from ..utils import unique_name


class HMType(object):
    pass


class Scalar(HMType):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __mul__(self, other):
        if isinstance(other, Array):
            return Array(unique_name(), self.value * other.data)
        print(type(other))
        raise NotImplementedError()


class Float32(Scalar):
    pass


class Int(Scalar):
    pass


