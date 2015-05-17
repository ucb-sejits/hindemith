from hindemith.operations.core import ElementLevel, DeviceLevel
import numpy as np
from string import Template


class ElementwiseArrayOp(ElementLevel):
    """
    c = Add(a, b)
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        num_work_items = np.prod(sources[0].shape)
        return (num_work_items, )

    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        return Template(
            "$target[index] = $operand1[index] $op $operand2[index];"
        ).substitute(target=sinks[0], op=cls.op, operand1=sources[0],
                     operand2=sources[1])


class ArrayAdd(ElementwiseArrayOp):
    op = "+"


class ArraySub(ElementwiseArrayOp):
    op = "-"


class ArrayMul(ElementwiseArrayOp):
    op = "*"


class ArrayDiv(ElementwiseArrayOp):
    op = "/"


class ArrayScalarOp(ElementwiseArrayOp):
    """
    b = Add(a, 1)
    """
    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        return Template(
            "$target[index] = $operand1[index] $op $operand2;"
        ).substitute(target=sinks[0], op=cls.op, operand1=sources[0],
                     operand2=symbol_table[sources[1]])


class ArrayScalarAdd(ArrayScalarOp, ArrayAdd):
    pass


class ArrayScalarSub(ArrayScalarOp, ArraySub):
    pass


class ArrayScalarMul(ArrayScalarOp, ArrayMul):
    pass


class ArrayScalarDiv(ArrayScalarOp, ArrayDiv):
    pass


class ElementwiseArrayMap(ElementwiseArrayOp):
    """
    b = Map(fn, a)
    """
    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        return Template(
            "$target[index] = $fn($operand[index]);"
        ).substitute(target=sinks[0], fn=cls.fn, operand=sources[0])


class Sqrt(ElementwiseArrayMap):
    fn = 'sqrt'


class Square(ElementwiseArrayMap):
    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        return Template(
            "$target[index] = pow($operand[index], 2);"
        ).substitute(target=sinks[0], operand=sources[0])
