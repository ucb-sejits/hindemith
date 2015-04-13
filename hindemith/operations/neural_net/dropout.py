from hindemith.operations.core import ElementLevel, register_operation
from hindemith.types import NDArray
import numpy as np
import ast
import sys


class Dropout(ElementLevel):
    def __init__(self, statement, symbol_table):
        self.symbol_table = symbol_table
        self.statement = statement
        self.operand_name = statement.value.args[0].id
        self.operand = self.symbol_table[self.operand_name]
        self.mask_name = None
        for keyword in statement.value.keywords:
            if keyword.arg == 'threshold':
                if isinstance(keyword.value, ast.Name):
                    self.threshold = symbol_table[keyword.value.id]
                else:
                    self.threshold = keyword.value.n
                self.scale = 1.0 / (1.0 - self.threshold)
            elif keyword.arg == 'mask':
                self.mask_name = keyword.value.id
            else:
                raise Exception("Unsupport keyword arg to Dropout",
                                keyword.arg)
        self.sources = [self.operand_name, self.mask_name]

        self.target_name = statement.targets[0].id
        self.target = symbol_table[self.target_name]
        self.sinks = [self.target_name]

    def compile(self):
        body = (
            "{target}[get_global_id(0)] = "
            "{op}[get_global_id(0)] * {mask}[get_global_id(0)] * {scale}f;"
        ).format(target=self.target_name, op=self.operand_name,
                 mask=self.mask_name, scale=self.scale)
        global_size = (np.prod(self.operand.shape), )
        return body, global_size, self.sources, self.sinks

    @classmethod
    def match(cls, node, symbol_table):
        if not isinstance(node, ast.Assign):
            return False
        node = node.value
        if isinstance(node, ast.Call):
            return isinstance(node.func, ast.Name) and \
                node.func.id == 'Dropout'


register_operation(Dropout)
