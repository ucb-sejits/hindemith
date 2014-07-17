__author__ = 'leonardtruong'

from ..utils import uniqueName

class HMType(object):
    pass


class Scalar(HMType):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __mul__(self, other):
        if isinstance(other, Array):
            return Array(uniqueName(), self.value * other.data)
        print(type(other))
        raise NotImplementedError()


class Float32(Scalar):
    pass


class Int(Scalar):
    pass


class Array(HMType):
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype

    def __mul__(self, other):
        if isinstance(other, Scalar):
            return Array(uniqueName(), self.data * other.value)
        elif isinstance(other, Array):
            return Array(uniqueName(), self.data * other.data)
        print(type(other))
        raise NotImplementedError()

    def __div__(self, other):
        if isinstance(other, Scalar):
            return Array(uniqueName(), self.data / other.value)
        elif isinstance(other, Array):
            return Array(uniqueName(), self.data / other.data)
        print(type(other))
        raise NotImplementedError()

    def __sub__(self, other):
        if isinstance(other, Array):
            return Array(uniqueName(), self.data - other.data)
        print(type(other))
        raise NotImplementedError()

    def __add__(self, other):
        if isinstance(other, Array):
            return Array(uniqueName(), self.data + other.data)
        elif isinstance(other, Scalar):
            return Array(uniqueName(), self.data + other.value)
        print(type(other))
        raise NotImplementedError()

