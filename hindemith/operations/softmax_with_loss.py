from hindemith.operations.core import DeviceLevel
from hindemith.types import hmarray
from hindemith.cl import queue, context
from string import Template
import numpy as np
import pycl as cl


class SoftmaxWithLossForward(DeviceLevel):
    """
    top = SoftmaxWithLossForward(bottom, label, prob)
    """
    @classmethod
    def get_launcher(cls, sources, sinks, keywords, symbol_table):
        bottom = symbol_table[sources[0]]
        num = bottom.shape[0]
        channels = bottom.shape[1]
        scale_shape = list(bottom.shape)
        scale_shape[1] = 1
        scale = hmarray(tuple(scale_shape))
        loss = hmarray(scale_shape)
        spatial_dim = int(np.prod(bottom.shape[2:]))
        count = np.prod(bottom.shape)

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
  if (get_global_id(0) < $num_times_spatial) {
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
  if (get_global_id(0) < $num_times_spatial) {
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
__kernel void SoftmaxLossForward(global const float* prob_data,
    global const float* label, global float* loss) {
  if (get_global_id(0) < $num_times_spatial) {
    int index = get_global_id(0);
    const int n = index / $spatial_dim;
    const int s = index % $spatial_dim;
    const int label_value = (int) label[n * $spatial_dim + s];
    loss[index] = -log(
        max(prob_data[n * $dim + label_value * $spatial_dim + s],
            FLT_MIN));
  }
}
// @end=cl@
""").substitute(count=count, num_times_spatial=num * spatial_dim,
                channels=channels, spatial_dim=spatial_dim,
                dim=np.prod(bottom.shape[1:]))

        program = cl.clCreateProgramWithSource(context, kernels).build()
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
        loss_forward.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_mem)

        class SoftmaxLauncher(object):
            def compile(self):
                pass

            def launch(self, symbol_table):
                bottom = symbol_table[sources[0]]
                label = symbol_table[sources[1]]
                prob = symbol_table[sources[2]]
                top = symbol_table[sinks[0]]
                copy_kern(bottom.ocl_buf, prob.ocl_buf).on(queue, (count, ))
                max_kern(prob.ocl_buf, scale.ocl_buf).on(queue,
                                                         (num * spatial_dim, ))
                sub_kern(scale.ocl_buf, prob.ocl_buf).on(queue, (count, ))
                exp_kern(prob.ocl_buf, prob.ocl_buf).on(queue, (count, ))
                sum_kern(prob.ocl_buf, scale.ocl_buf).on(queue,
                                                         (num * spatial_dim,))
                div_kern(scale.ocl_buf, prob.ocl_buf).on(queue, (count, ))
                loss_forward(prob.ocl_buf, label.ocl_buf,
                             loss.ocl_buf).on(queue, (num * spatial_dim, ))
                loss.sync_host()
                top[0] = np.sum(loss) / np.float32(num)
                top.sync_ocl()

        return SoftmaxLauncher()


class SoftmaxWithLossBackward(DeviceLevel):
    """
    bottom_diff = SoftMaxWithLossBackward(loss, label, prob)
    """
    @classmethod
    def get_launcher(cls, sources, sinks, keywords, symbol_table):
        bottom_diff = symbol_table[sinks[0]]
        count = np.prod(bottom_diff.shape)
        num = bottom_diff.shape[0]
        spatial_dim = int(np.prod(bottom_diff.shape[2:]))
        kernels = Template("""
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
    global float* bottom_diff) {
  int index = get_global_id(0);
  if (index < $global_size) {
    const int n = index / $spatial_dim;
    const int s = index % $spatial_dim;
    const int label_value = (int)label[n * $spatial_dim + s];
    bottom_diff[n * $dim + label_value * $spatial_dim + s] -= 1;
  }
}""").substitute(count=count, spatial_dim=spatial_dim,
                 dim=np.prod(bottom_diff.shape[1:]),
                 global_size=num*spatial_dim)

        program = cl.clCreateProgramWithSource(context, kernels).build()
        copy = program['kernel_copy']
        copy.argtypes = (cl.cl_mem, cl.cl_mem)
        scale = program['kernel_scale']
        scale.argtypes = (cl.cl_float, cl.cl_mem)
        backward = program['SoftmaxLossBackwardGPU']
        backward.argtypes = (cl.cl_mem, cl.cl_mem)

        class Launcher(object):
            def compile(self):
                pass

            def launch(self, symbol_table):
                bottom_diff = symbol_table[sinks[0]]
                # top_diff = symbol_table[sources[0]]
                label = symbol_table[sources[1]]
                prob = symbol_table[sources[2]]
                copy(prob.ocl_buf, bottom_diff.ocl_buf).on(
                    queue, (np.prod(prob.shape),))
                backward(label.ocl_buf, bottom_diff.ocl_buf).on(
                    queue, (num * spatial_dim), )
                loss_weight = 1.0
                scale(np.float32(loss_weight / float(num)),
                      bottom_diff.ocl_buf).on(queue, (np.prod(prob.shape), ))
        return Launcher()
