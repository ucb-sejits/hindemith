from hindemith.operations.core import DeviceLevel, register_operation
from hindemith.types import NDArray
from hindemith.cl import queue, context
import numpy as np
import pycl as cl
import ast
from string import Template

kernels = Template("""
@begin=cl@
__kernel void kernel_channel_max(global const float* data,
                                 global float* out) {
  if (get_global_id(0) < $count) {
    int index = get_global_id(0);
    out[index] = data[index];
  }
}
__kernel void kernel_channel_max(global const float* data,
                                 global float* out) {
  if (get_global_id(0) < $outer_inner) {
    int index = get_global_id(0);
    int n = index / $spatial_dim;
    int s = index % $spatial_dim;
    float maxval = -FLT_MAX;
    for (int c = 0; c < $channels; ++c) {
      maxval = max(data[(n * $channels + c) * $spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}
__kernel void kernel_channel_subtract(global const float* channel_max,
                                      global float* data) {
  if (get_global_id(0) < $count) {
    int index = get_global_id(0);
    int n = index / $channels / $spatial_dim;
    int s = index % $spatial_dim;
    data[index] -= channel_max[n * $spatial_dim + s];
  }
}
__kernel void kernel_exp(global const float* data, global float* out) {
  if (get_global_id(0) < $count) {
    int index = get_global_id(0);
    out[index] = exp(data[index]);
  }
}
__kernel void kernel_channel_sum(global const float* data,
                                 global float* channel_sum) {
  if (get_global_id(0) < $outer_inner) {
    int index = get_global_id(0);
    int n = index / $spatial_dim;
    int s = index % $spatial_dim;
    float sum = 0;
    for (int c = 0; c < $channels; ++c) {
      sum += data[(n * $channels + c) * $spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}
__kernel void kernel_channel_div(global const float* channel_sum,
                                 global float* data) {
  if (get_global_id(0) < $count) {
    int index = get_global_id(0);
    int n = index / $channels / $spatial_dim;
    int s = index % $spatial_dim;
    data[index] /= channel_sum[n * $spatial_dim + s];
  }
}
__kernel void kernel_channel_dot(global const float* data_1,
                                 global const float* data_2,
    global float* channel_dot) {
  if (get_global_id(0) < $outer_inner) {
    int index = get_global_id(0);
    int n = index / $spatial_dim;
    int s = index % $spatial_dim;
    float dot = 0;
    for (int c = 0; c < $channels; ++c) {
      dot += (data_1[(n * $channels + c) * $spatial_dim + s]
          * data_2[(n * $channels + c) * $spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}
__kernel void SoftmaxLossForward(global const float* prob_data,
    global const float* label, global float* loss, global float* counts) {
  if (get_global_id(0) < $outer_inner) {
    const int n = index / $spatial_dim;
    const int s = index % $spatial_dim;
    const int label_value = (int) label[n * $spatial_dim + s];
    if ($has_ignore_label && label_value == $ignore_label) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(
          max(prob_data[n * dim + label_value * spatial_dim + s], FLT_MIN));
      counts[index] = 1;
    }
  }
}
@end=cl@
""")


class SoftMaxWithLossForward(DeviceLevel):
    def __init__(self, statement, symbol_table):
        self.symbol_table = symbol_table
        self.statement = statement
        self.bottom_name = statement.value.args[0].id
        self.bottom = self.symbol_table[self.bottom_name]
        self.label_name = statement.value.args[1].id
        self.label = self.symbol_table[self.label_name]
        self.prob_data_name, self.prob_data = \
            NDArray.unique(self.bottom.shape[0], np.float32)
        self.counts_name, self.counts = \
            NDArray.unique(self.bottom.shape[0], np.float32)
        self.loss_name, self.loss = \
            NDArray.unique(self.bottom.shape, np.float32)
        scale_shape = self.bottom.shape
        scale_shape[1] = 1
        self.scale_name, self.scale = NDArray.unique(scale_shape, np.float32)
        self.symbol_table[self.prob_data_name] = self.prob_data
        self.sources = [self.bottom_name, self.label_name]
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
        num = self.top.shape[0]
        channels = self.top.shape[1]
        spatial_dim = np.prod(self.top.shape[2:])
        count = np.prod(self.top.shape)
        outer_inner = num * spatial_dim
        kerns = kernels.substitute(channels=channels, spatial_dim=spatial_dim,
                                   count=count, outer_inner=outer_inner)
        program = cl.clCreateProgramWithSource(context, kerns).build()
        copy_kern = program['kernel_copy']
        copy_kern.argtypes = (cl.cl_mem, cl.cl_mem)
        max_kern = program['kernel_channel_max']
        max_kern.argtypes = (cl.cl_mem, cl.cl_mem)
        sub_kern = program['kernel_channel_subtract']
        sub_kern.argtypes = (cl.cl_mem, cl.cl_mem)
        exp_kern = program['kernel_channel_exp']
        exp_kern.argtypes = (cl.cl_mem, cl.cl_mem)
        sum_kern = program['kernel_channel_sum']
        sum_kern.argtypes = (cl.cl_mem, cl.cl_mem)
        div_kern = program['kernel_channel_div']
        div_kern.argtypes = (cl.cl_mem, cl.cl_mem)
        loss_forward = program['SoftmaxLossForward']
        loss_forward.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem)

        class SoftmaxLauncher(object):
            def __init__(self, op):
                self.op = op

            def launch(self, env):
                copy_kern(self.op.bottom.ocl_buf,
                          self.op.prob_data.ocl_buf).on(queue, (count, ))
                max_kern(self.op.prob_data.ocl_buf,
                         self.op.scale.ocl_buf).on(queue, (outer_inner, ))
                sub_kern(self.op.scale.ocl_buf,
                         self.op.prob_data.ocl_buf).on(queue, (count, ))
                exp_kern(self.op.prob_data.ocl_buf,
                         self.op.prob_data.ocl_buf).on(queue, (count, ))
                sum_kern(self.op.prob_data.ocl_buf,
                         self.op.scale.ocl_buf).on(queue, (outer_inner, ))
                div_kern(self.op.scale.ocl_buf,
                         self.op.prob_data.ocl_buf).on(queue, (count, ))
                self.prob_data.host_dirty = True

        class LossLauncher(object):
            def __init__(self, op):
                self.op = op

            def launch(self, env):
                loss_forward(self.op.prob_data.ocl_buf, self.op.label.ocl_buf,
                             self.op.loss.ocl_buf, self.op.counts.ocl_buf)
                self.op.loss.sync_host(force=True)
                self.op.counts.sync_host(force=True)
                self.op.top[0] = np.sum(self.op.loss) / np.sum(self.op.counts)

        return [SoftmaxLauncher(self), LossLauncher(self)]

    @classmethod
    def match(cls, node, symbol_table):
        if not isinstance(node, ast.Assign):
            return False
        node = node.value
        if isinstance(node, ast.Call):
            return isinstance(node.func, ast.Name) and \
                node.func.id == 'SoftMaxWithLossForward'


register_operation(SoftMaxWithLossForward)
