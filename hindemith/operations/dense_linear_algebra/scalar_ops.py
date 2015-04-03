from hindemith.operations.core import ElementLevel, register_operation
from hindemith.types import NDArray
import ast
import numpy as np


class ScalarOperation(ElementLevel):
    def __init__(self, statement, symbol_table):
        self.symbol_table = symbol_table
        self.statement = statement

        if isinstance(statement.value.left, ast.Num):
            self.operand1 = statement.value.left.n
        else:
            self.operand1_name = statement.value.left.id
            self.operand1 = symbol_table[self.operand1_name]
        if isinstance(statement.value.right, ast.Num):
            self.operand2 = statement.value.right.n
        else:
            self.operand2_name = statement.value.right.id
            self.operand2 = symbol_table[self.operand2_name]
        if isinstance(self.operand1, NDArray):
            symbol_table[statement.targets[0].id] = NDArray(
                self.operand1.shape, self.operand1.dtype)
            self.sources = [self.operand1_name]
        else:
            symbol_table[statement.targets[0].id] = NDArray(
                self.operand2.shape, self.operand2.dtype)
            self.sources = [self.operand2_name]
        self.target_name = statement.targets[0].id
        self.target = symbol_table[self.target_name]
        self.sinks = [self.target_name]

    def get_global_size(self):
        if isinstance(self.operand1, NDArray):
            return (np.prod(self.operand1.shape), )
        else:
            return (np.prod(self.operand2.shape), )

    def compile(self):
        if isinstance(self.operand1, NDArray):
            op2 = str(self.operand2)
            if type(self.operand2) == float:
                op2 += "f"
            return "{} = {} {} {};".format(
                self.target.get_element(self.target_name),
                self.operand1.get_element(self.operand1_name),
                self.op,
                op2
            )
        else:
            op1 = str(self.operand1)
            if type(self.operand2) == float:
                op1 += "f"
            return "{} = {} {} {};".format(
                self.target.get_element(self.target_name),
                op1,
                self.op,
                self.operand2.get_element(self.operand2_name),
            )

    @classmethod
    def match(cls, node, symbol_table):
        if not isinstance(node, ast.Assign):
            return False
        node = node.value
        if (isinstance(node, ast.BinOp) and
                isinstance(node.op, cls.py_op)):
            if (isinstance(node.left, ast.Name) and
                    isinstance(node.right, ast.Name) and
                    node.left.id in symbol_table and
                    node.right.id in symbol_table):
                return (
                    isinstance(symbol_table[node.left.id], (int, float)) and
                    isinstance(symbol_table[node.right.id], NDArray) or
                    isinstance(symbol_table[node.left.id], NDArray) and
                    isinstance(symbol_table[node.right.id], (int, float))
                )
            elif (isinstance(node.left, ast.Name) and
                  node.left.id in symbol_table and
                  isinstance(node.right, ast.Num)):
                return isinstance(symbol_table[node.left.id], NDArray)
            elif (isinstance(node.right, ast.Name) and
                  node.right.id in symbol_table and
                  isinstance(node.left, ast.Num)):
                return isinstance(symbol_table[node.right.id], NDArray)
        return False


class ScalarAdd(ScalarOperation):
    op = "+"
    py_op = ast.Add


class ScalarSub(ScalarOperation):
    op = "-"
    py_op = ast.Sub


class ScalarMul(ScalarOperation):
    op = "*"
    py_op = ast.Mult


class ScalarDiv(ScalarOperation):
    op = "/"
    py_op = ast.Div


for op in [ScalarMul, ScalarAdd, ScalarDiv, ScalarSub]:
    register_operation(op)
