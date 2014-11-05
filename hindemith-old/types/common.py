from hindemith.operations.dense_linear_algebra.array_op import ArrayAdd, ArraySub, ArrayMul, \
    ArrayDiv
from hindemith.utils import unique_name

__author__ = 'leonardtruong'


class HMType(object):
    pass


class Scalar(HMType):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    # def __mul__(self, other):
    #     if isinstance(other, Array):
    #         return Array(unique_name(), self.value * other.data)
    #     print(type(other))
    #     raise NotImplementedError()

    # def __add__(self, other):
    #     if isinstance(other, Array):
    #         return Array(unique_name(), self.value + other.data)
    #     print(type(other))
    #     raise NotImplementedError()


class Float32(Scalar):
    pass


class Int(Scalar):
    pass


class Array(HMType):
    def __new__(cls, name, data):
        class ArrayInstance(cls):
            def __new__(cls, *args, **kwargs):
                return object.__new__(cls)

            def __init__(self, name, data):
                self.name = name
                self.data = data
                self.shape = data.shape
                self.dtype = data.dtype
        ArrayInstance.__add__ = ArrayAdd(name, data)
        ArrayInstance.__sub__ = ArraySub(name, data)
        ArrayInstance.__mul__ = ArrayMul(name, data)
        ArrayInstance.__div__ = ArrayDiv(name, data)
        return ArrayInstance(name, data)
