from hindemith.operations.core import ElementLevel, register_operation
from hindemith.types import Vector, Matrix
import ast
import numpy as np


class ElementwiseOperation(ElementLevel):
    py_op = None

    def __init__(self, statement, symbol_table):
        self.symbol_table = symbol_table
        self.statement = statement

        self.operand1_name = statement.value.left.id
        self.operand1 = symbol_table[self.operand1_name]
        self.operand2_name = statement.value.right.id
        self.operand2 = symbol_table[self.operand2_name]
        assert (
            type(self.operand1) in {Vector, Matrix} and
            type(self.operand2) in {Vector, Matrix} and
            self.operand1.__class__ is self.operand2.__class__
        )

        if isinstance(self.operand1, Vector):
            symbol_table[statement.targets[0].id] = Vector(self.operand1.size,
                                                           self.operand1.dtype)
        else:
            symbol_table[statement.targets[0].id] = Matrix(self.operand1.shape,
                                                           self.operand1.dtype)
        self.target_name = statement.targets[0].id
        self.target = symbol_table[self.target_name]
        self.sources = [self.operand1_name, self.operand2_name]
        self.sinks = [self.target_name]

    def get_global_size(self):
        return (np.prod(self.operand1.shape), )

    def compile(self):
        return "{} = {} {} {};".format(
            self.target.get_element(self.target_name),
            self.operand1.get_element(self.operand1_name),
            self.op,
            self.operand2.get_element(self.operand2_name)
        )

    @classmethod
    def match(cls, node, symbol_table):
        if not isinstance(node, ast.Assign):
            return False
        node = node.value
        if (isinstance(node, ast.BinOp) and
                isinstance(node.op, cls.py_op) and
                isinstance(node.left, ast.Name) and
                isinstance(node.right, ast.Name)):
            return (
                node.left.id in symbol_table and
                type(symbol_table[node.left.id]) in {Vector, Matrix} and
                node.right.id in symbol_table and
                type(symbol_table[node.right.id]) in {Vector, Matrix} and
                type(symbol_table[node.left.id]) is
                type(symbol_table[node.right.id])
            )
        return False


class ElementwiseAdd(ElementwiseOperation):
    op = "+"
    py_op = ast.Add


class ElementwiseSub(ElementwiseOperation):
    op = "-"
    py_op = ast.Sub


class ElementwiseMul(ElementwiseOperation):
    op = "*"
    py_op = ast.Mult


class ElementwiseDiv(ElementwiseOperation):
    op = "/"
    py_op = ast.Div


for op in [ElementwiseMul, ElementwiseAdd, ElementwiseDiv, ElementwiseSub]:
    register_operation(op)
