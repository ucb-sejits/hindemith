from hindemith.operations.core import DeviceLevel, register_operation
from hindemith.types import NDArray
from hindemith.cl import context, Kernel
import numpy as np
import pycl as cl
import ast


class Lrn(DeviceLevel):
    def __init__(self, statement, symbol_table):
        self.symbol_table = symbol_table
        self.statement = statement
        self.operand_name = statement.value.args[0].id
        self.operand = self.symbol_table[self.operand_name]
        self.sources = [self.operand_name]
        name, self.scale = NDArray.unique(self.operand.shape,
                                          self.operand.dtype)
        symbol_table[name] = self.scale
        self.scale_name = name
        for keyword in statement.value.keywords:
            if keyword.arg in {'alpha', 'beta', 'local_size', 'k'}:
                if isinstance(keyword.value, ast.Name):
                    value = symbol_table[keyword.value.id]
                else:
                    value = keyword.value.n
                setattr(self, keyword.arg, value)
            else:
                raise Exception("Unsupport keyword arg to Lrn", keyword.arg)

        self.target_name = statement.targets[0].id
        symbol_table[self.target_name] = NDArray.like(self.operand)
        self.target = symbol_table[self.target_name]
        self.sinks = [self.target_name]

    def compile(self):
        shape = self.operand.shape
        kernels = """
__kernel void LRNFillScale(global const float* in, global float* scale) {{
  int index = get_global_id(0);
  int w = index % {width};
  int h = (index / {width}) % {height};
  int n = index / {width} / {height};
  int offset = (n * {channels} * {height} + h) * {width} + w;
  int step = {height} * {width};
  in += offset;
  scale += offset;
  int head = 0;
  int pre_pad = ({local_size} - 1) / 2;
  int post_pad = {local_size} - pre_pad - 1;
  float accum_scale = 0;
  // fill the scale at [n, :, h, w]
  // accumulate values
  while (head < post_pad && head < {channels}) {{
    accum_scale += in[head * step] * in[head * step];
    ++head;
  }}
  // both add and subtract
  while (head < {channels}) {{
    accum_scale += in[head * step] * in[head * step];
    if (head - {local_size} >= 0) {{
      accum_scale -= in[(head - {local_size}) * step] * \
          in[(head - {local_size}) * step];
    }}
    scale[(head - post_pad) * step] = {k} + accum_scale * {alpha_over_size};
    ++head;
  }}
  // subtract only
  while (head < {channels} + post_pad) {{
    if (head - {local_size} >= 0) {{
      accum_scale -= in[(head - {local_size}) * step] * \
          in[(head - {local_size}) * step];
    }}
    scale[(head - post_pad) * step] = {k} + accum_scale * {alpha_over_size};
    ++head;
  }}
}}
__kernel void LRNComputeOutput(global const float* in,
                               global const float* scale,
                               global float* out) {{
  int index = get_global_id(0);
  out[index] = in[index] * pow(scale[index], {negative_beta}f);
}}""".format(width=shape[3],
             height=shape[2],
             channels=shape[1],
             local_size=self.local_size,
             alpha_over_size=self.alpha / self.local_size,
             k=self.k,
             negative_beta=-self.beta)
        program = cl.clCreateProgramWithSource(
            context, kernels
        ).build()
        fill_kern = program['LRNFillScale']
        fill_kern.argtypes = (cl.cl_mem, cl.cl_mem)
        fill_global = (shape[0] * shape[2] * shape[3], )
        fill_inputs = [self.operand_name]
        fill_outputs = [self.scale_name]
        compute_kern = program['LRNComputeOutput']
        compute_kern.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_mem)
        compute_inputs = [self.operand_name, self.scale_name]
        compute_outputs = [self.target_name]
        compute_global = (np.prod(shape), )
        return [Kernel(fill_kern, fill_inputs, fill_outputs, fill_global),
                Kernel(compute_kern, compute_inputs, compute_outputs,
                       compute_global)]

    @classmethod
    def match(cls, node, symbol_table):
        if not isinstance(node, ast.Assign):
            return False
        node = node.value
        if isinstance(node, ast.Call):
            return isinstance(node.func, ast.Name) and \
                node.func.id == 'Lrn'


register_operation(Lrn)
