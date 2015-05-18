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
            "$target = $operand1 $op $operand2;"
        ).substitute(target=sinks[0].get_element(), op=cls.op, operand1=sources[0].get_element(),
                     operand2=sources[1].get_element())


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
            "$target = $operand1 $op $operand2;"
        ).substitute(target=sinks[0].get_element(), op=cls.op, operand1=sources[0].get_element(),
                     operand2=symbol_table[sources[1].name])


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
            "$target = $fn($operand);"
        ).substitute(target=sinks[0].get_element(), fn=cls.fn, operand=sources[0].get_element())


class Sqrt(ElementwiseArrayMap):
    fn = 'sqrt'


class Square(ElementwiseArrayMap):
    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        return Template(
            "$target = pow($operand, 2);"
        ).substitute(target=sinks[0].get_element(), operand=sources[0].get_element())
