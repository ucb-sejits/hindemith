from teller.operations.dense_linear_algebra.array_op import ArrayMul, ArraySub, ArrayDiv, ArrayAdd

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


class Array(HMType):
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype

    def __mul__(self, other):
        if isinstance(other, Scalar):
            return Array(unique_name(), self.data * other.value)
        elif isinstance(other, Array):
            return Array(unique_name(), ArrayMul()(self.data, other.data))
        print(type(other))
        raise NotImplementedError()

    def __div__(self, other):
        if isinstance(other, Scalar):
            return Array(unique_name(), self.data / other.value)
        elif isinstance(other, Array):
            return Array(unique_name(), ArrayDiv()(self.data, other.data))
        print(type(other))
        raise NotImplementedError()

    def __sub__(self, other):
        if isinstance(other, Array):
            return Array(unique_name(), ArraySub()(self.data, other.data))
        print(type(other))
        raise NotImplementedError()

    def __add__(self, other):
        if isinstance(other, Array):
            return Array(unique_name(), ArrayAdd()(self.data, other.data))
        elif isinstance(other, Scalar):
            return Array(unique_name(), self.data + other.value)
        print(type(other))
        raise NotImplementedError()

