from hindemith.operations.core import DeviceLevel, register_operation
from hindemith.cl import context, Kernel
import numpy as np
import pycl as cl
import ast


class MMult(DeviceLevel):
    def __init__(self, statement, symbol_table):
        self.symbol_table = symbol_table
        self.statement = statement
        self.operand1_name = statement.value.args[0].id
        self.operand1 = self.symbol_table[self.operand1_name]
        self.operand2_name = statement.value.args[1].id
        self.operand2 = self.symbol_table[self.operand2_name]
        self.sources = [self.operand1_name, self.operand2_name]

        self.target_name = statement.targets[0].id
        self.target = symbol_table[self.target_name]
        self.sinks = [self.target_name]

    def compile(self):
        kernels = """
__kernel void gemm(global const float* A, global const float* B, global float* C) {{
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
 
   float value = 0.0;
   for (int k = 0; k < {width_a}; ++k)
   {{
      float elementA = A[ty * {width_a} + k];
      float elementB = B[k * {width_b} + tx];
      value += elementA * elementB;
   }}
 
   C[ty * {width_b} + tx] = value;
}}
""".format(width_a=self.operand1.shape[1], width_b=self.operand2.shape[1])
        program = cl.clCreateProgramWithSource(
            context, kernels
        ).build()
        kern = program['gemm']
        kern.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_mem)
        global_size = (self.operand2.shape[1], self.operand1.shape[0])
        inputs = [self.operand1_name, self.operand2_name]
        outputs = [self.target_name]
        return [Kernel(kern, inputs, outputs, global_size)]

    @classmethod
    def match(cls, node, symbol_table):
        if not isinstance(node, ast.Assign):
            return False
        node = node.value
        if isinstance(node, ast.Call):
            return isinstance(node.func, ast.Name) and \
                node.func.id == 'MMult'


register_operation(MMult)
