from hindemith.operations.core import DeviceLevel, register_operation
from hindemith.types import NDArray
from hindemith.cl import context, Kernel
import numpy as np
import pycl as cl
import ast


class LrnForward(DeviceLevel):
    def __init__(self, statement, symbol_table):
        self.symbol_table = symbol_table
        self.statement = statement
        self.operand_name = statement.value.args[0].id
        self.operand = self.symbol_table[self.operand_name]
        self.scale_name = statement.value.args[1].id
        self.sources = [self.operand_name, self.scale_name]
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
        self.target.host_dirty = True
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
                node.func.id == 'LrnForward'


register_operation(LrnForward)


class LrnBackward(DeviceLevel):
    def __init__(self, statement, symbol_table):
        self.symbol_table = symbol_table
        self.statement = statement
        self.bottom_name = statement.value.args[0].id
        self.bottom = self.symbol_table[self.bottom_name]
        self.top_name = statement.value.args[1].id
        self.top = self.symbol_table[self.top_name]
        self.top_diff_name = statement.value.args[2].id
        self.top_diff = self.symbol_table[self.top_diff_name]
        self.scale_name = statement.value.args[3].id
        self.sources = [self.bottom_name, self.top_name, self.top_diff_name, self.scale_name]
        for keyword in statement.value.keywords:
            if keyword.arg in {'alpha', 'beta', 'local_size', 'k'}:
                if isinstance(keyword.value, ast.Name):
                    value = symbol_table[keyword.value.id]
                else:
                    value = keyword.value.n
                setattr(self, keyword.arg, value)
            else:
                raise Exception("Unsupport keyword arg to Lrn", keyword.arg)

        self.bottom_diff_name = statement.targets[0].id
        self.bottom_diff = symbol_table[self.bottom_diff_name]
        self.sinks = [self.bottom_diff_name]

    def compile(self):
        shape = self.bottom_diff.shape
        global_size = (shape[0] * shape[2] * shape[3], )
        kernels = """
__kernel void LrnComputeDiff(global const float* bottom, global const float* scale,
        global const float* top, global const float* top_diff, global float* bottom_diff) {{
  if (get_global_id(0) < {global_size}) {{
    // find out the local offset
    int index = get_global_id(0);
    int w = index % {width};
    int h = (index / {width}) % {height};
    int n = index / {width} / {height};
    int offset = (n * {channels} * {height} + h) * {width} + w;
    int step = {height} * {width};
    bottom += offset;
    top += offset;
    scale += offset;
    top_diff += offset;
    bottom_diff += offset;
    int head = 0;
    int pre_pad = {local_size} - ({local_size} + 1) / 2;
    int post_pad = {local_size} - pre_pad - 1;
    float accum_ratio = 0;
    // accumulate values
    while (head < post_pad && head < {channels}) {{
      accum_ratio += top_diff[head * step] * top[head * step] /
          scale[head * step];
      ++head;
    }}
    // both add and subtract
    while (head < {channels}) {{
      accum_ratio += top_diff[head * step] * top[head * step] /
          scale[head * step];
      if (head - {local_size} >= 0) {{
        accum_ratio -= top_diff[(head - {local_size}) * step] *
            top[(head - {local_size}) * step] / scale[(head - {local_size}) * step];
      }}
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], {negative_beta}) - {cache_ratio} *
          bottom[(head - post_pad) * step] * accum_ratio;
      ++head;
    }}
    // subtract only
    while (head < {channels} + post_pad) {{
      if (head - {local_size} >= 0) {{
        accum_ratio -= top_diff[(head - {local_size}) * step] *
            top[(head - {local_size}) * step] / scale[(head - {local_size}) * step];
      }}
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], {negative_beta}) - {cache_ratio} *
          bottom[(head - post_pad) * step] * accum_ratio;
      ++head;
    }}
  }}
}}
""".format(width=shape[3],
           height=shape[2],
           channels=shape[1],
           local_size=self.local_size,
           negative_beta=-self.beta,
           cache_ratio=2 * self.alpha * self.beta / self.local_size,
           global_size= global_size[0])
        program = cl.clCreateProgramWithSource(
            context, kernels
        ).build()
        kern = program['LrnComputeDiff']
        kern.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem)
        inputs = [self.bottom_name, self.scale_name, self.top_name, self.top_diff_name]
        outputs = [self.bottom_diff_name]

        return [Kernel(kern, inputs, outputs, global_size)]

    @classmethod
    def match(cls, node, symbol_table):
        if not isinstance(node, ast.Assign):
            return False
        node = node.value
        if isinstance(node, ast.Call):
            return isinstance(node.func, ast.Name) and \
                node.func.id == 'LrnBackward'


register_operation(LrnBackward)
