from hindemith.operations.core import DeviceLevel, register_operation
from hindemith.types import NDArray
from hindemith.cl import queue, context
import numpy as np
import pycl as cl
import ast
from string import Template

kernels = Template("""
// @begin=cl@
__kernel void kernel_copy(global const float* data,
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
    int index = get_global_id(0);
    const int n = index / $spatial_dim;
    const int s = index % $spatial_dim;
    const int label_value = (int) label[n * $spatial_dim + s];
    // if ($has_ignore_label && label_value == $ignore_label) {
    if (false && label_value == $ignore_label) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(
          max(prob_data[n * $channels + label_value * $spatial_dim + s],
              FLT_MIN));
      counts[index] = 1;
    }
  }
}
// @end=cl@
""")


class SoftMaxWithLossForward(DeviceLevel):
    def __init__(self, statement, symbol_table):
        self.symbol_table = symbol_table
        self.statement = statement
        self.bottom_name = statement.value.args[0].id
        self.bottom = self.symbol_table[self.bottom_name]
        self.label_name = statement.value.args[1].id
        self.label = self.symbol_table[self.label_name]
        self.prob_name = statement.value.args[2].id
        self.prob = self.symbol_table[self.prob_name]
        self.counts_name, self.counts = \
            NDArray.unique(self.bottom.shape[0], np.float32)
        self.loss_name, self.loss = \
            NDArray.unique(self.bottom.shape, np.float32)
        scale_shape = list(self.bottom.shape)
        scale_shape[1] = 1
        scale_shape = tuple(scale_shape)
        self.scale_name, self.scale = NDArray.unique(scale_shape, np.float32)
        self.sources = [self.bottom_name, self.label_name]

        self.top_name = statement.targets[0].id
        self.top = symbol_table[self.top_name]
        self.sinks = [self.top_name]

    def compile(self):
        num = self.bottom.shape[0]
        channels = self.bottom.shape[1]
        spatial_dim = np.prod(self.bottom.shape[2:])
        count = np.prod(self.bottom.shape)
        outer_inner = num * spatial_dim
        kerns = kernels.substitute(channels=channels, spatial_dim=spatial_dim,
                                   count=count, outer_inner=outer_inner,
                                   has_ignore_label=False, ignore_label=0)
        program = cl.clCreateProgramWithSource(context, kerns).build()
        copy_kern = program['kernel_copy']
        copy_kern.argtypes = (cl.cl_mem, cl.cl_mem)
        max_kern = program['kernel_channel_max']
        max_kern.argtypes = (cl.cl_mem, cl.cl_mem)
        sub_kern = program['kernel_channel_subtract']
        sub_kern.argtypes = (cl.cl_mem, cl.cl_mem)
        exp_kern = program['kernel_exp']
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
                          self.op.prob.ocl_buf).on(queue, (count, ))
                max_kern(self.op.prob.ocl_buf,
                         self.op.scale.ocl_buf).on(queue, (outer_inner, ))
                sub_kern(self.op.scale.ocl_buf,
                         self.op.prob.ocl_buf).on(queue, (count, ))
                exp_kern(self.op.prob.ocl_buf,
                         self.op.prob.ocl_buf).on(queue, (count, ))
                sum_kern(self.op.prob.ocl_buf,
                         self.op.scale.ocl_buf).on(queue, (outer_inner, ))
                div_kern(self.op.scale.ocl_buf,
                         self.op.prob.ocl_buf).on(queue, (count, ))

        class LossLauncher(object):
            def __init__(self, op):
                self.op = op

            def launch(self, env):
                loss_forward(self.op.prob.ocl_buf, self.op.label.ocl_buf,
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


class SoftMaxWithLossBackward(DeviceLevel):
    def __init__(self, statement, symbol_table):
        self.symbol_table = symbol_table
        self.statement = statement
        self.top_diff_name = statement.value.args[0].id
        self.top_diff = self.symbol_table[self.top_diff_name]
        self.label_name = statement.value.args[1].id
        self.label = self.symbol_table[self.label_name]
        self.prob_name = statement.value.args[2].id
        self.prob = self.symbol_table[self.prob_name]
        self.sources = [self.label_name,
                        self.prob_name,
                        self.top_diff_name]

        self.bottom_diff_name = statement.targets[0].id
        self.bottom_diff = symbol_table[self.bottom_diff_name]
        self.counts_name, self.counts = \
            NDArray.unique(self.bottom_diff.shape[0], np.float32)
        self.sinks = [self.bottom_diff_name]

    def compile(self):
        num = self.bottom_diff.shape[0]
        channels = self.bottom_diff.shape[1]
        spatial_dim = np.prod(self.bottom_diff.shape[2:])
        count = np.prod(self.bottom_diff.shape)
        outer_inner = num * spatial_dim
        global_size = (outer_inner, )
        kerns = Template(
            """
// @begin=cl@
__kernel void kernel_copy(global const float* data,
                                 global float* out) {
  if (get_global_id(0) < $count) {
    int index = get_global_id(0);
    out[index] = data[index];
  }
}
__kernel void kernel_scale(float scale, global float* out) {
  if (get_global_id(0) < $count) {
    int index = get_global_id(0);
    out[index] *= scale;
  }
}
__kernel void SoftmaxLossBackwardGPU(global const float* label,
    global float* bottom_diff, global float* counts) {
  int index = get_global_id(0);
  if (index < $global_size) {
    const int n = index / $spatial_dim;
    const int s = index % $spatial_dim;
    const int label_value = (int)label[n * $spatial_dim + s];

    if (false && label_value == $ignore_label) {
      for (int c = 0; c < $channels; ++c) {
        bottom_diff[n * $dim + c * $spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * $dim + label_value * $spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}
// @end=cl@
            """
        ).substitute(
            global_size=outer_inner,
            spatial_dim=spatial_dim,
            channels=channels,
            dim=np.prod(self.bottom_diff.shape[1:]),
            count=count,
            ignore_label=0
        )

        program = cl.clCreateProgramWithSource(context, kerns).build()
        copy = program['kernel_copy']
        copy.argtypes = (cl.cl_mem, cl.cl_mem)
        scale = program['kernel_scale']
        scale.argtypes = (cl.cl_float, cl.cl_mem)
        backward = program['SoftmaxLossBackwardGPU']
        backward.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_mem)

        class Launcher(object):
            def __init__(self, op):
                self.op = op

            def launch(self, env):
                copy(self.op.prob.ocl_buf, self.op.bottom_diff.ocl_buf).on(
                    queue, (np.prod(self.op.prob.shape),))
                backward(self.op.label.ocl_buf, self.op.bottom_diff.ocl_buf,
                         self.op.counts.ocl_buf).on(queue, global_size)
                self.op.counts.sync_host(True)
                count = np.sum(self.op.counts)
                loss_weight = self.op.top_diff[0]
                scale(loss_weight / count, self.op.bottom_diff.ocl_buf).on(
                    queue, (np.prod(self.op.prob.shape), ))

        return [Launcher(self)]

    @classmethod
    def match(cls, node, symbol_table):
        if not isinstance(node, ast.Assign):
            return False
        node = node.value
        if isinstance(node, ast.Call):
            return isinstance(node.func, ast.Name) and \
                node.func.id == 'SoftMaxWithLossBackward'


register_operation(SoftMaxWithLossBackward)
