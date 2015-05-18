from hindemith.operations.core import DeviceLevel
from hindemith.types import hmarray
import numpy as np
from string import Template
import os
from hindemith.cl import hm_compile_and_load

backend = os.getenv("HM_BACKEND", "ocl")
if backend in {"ocl", "opencl", "OCL"}:
    from hindemith.cl import context, queue
    import pycl as cl


if backend in {"ocl", "opencl", "OCL"}:
    class SoftmaxForward(DeviceLevel):
        """
        top = SoftmaxForward(bottom)
        """
        @classmethod
        def get_launcher(cls, sources, sinks, keywords, symbol_table):
            bottom = symbol_table[sources[0].name]
            num = bottom.shape[0]
            channels = bottom.shape[1]
            scale_shape = list(bottom.shape)
            scale_shape[1] = 1
            scale = hmarray(tuple(scale_shape))
            spatial_dim = int(np.prod(bottom.shape[2:]))
            count = np.prod(bottom.shape)

            kernels = Template("""
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

            class SoftmaxLauncher(object):
                def __init__(self, sources, sinks):
                    self.sources = sources
                    self.sinks = sources

                def compile(self):
                    pass

                def launch(self, symbol_table, wait_for):
                    bottom = symbol_table[sources[0].name]
                    top = symbol_table[sinks[0].name]
                    if count % 16:
                        padded_count = (count + 15) & (~15)
                    else:
                        padded_count = count
                    num_times_spatial = num * spatial_dim
                    if num_times_spatial % 16:
                        padded_num_times_spatial = (num_times_spatial + 15) & (~15)
                    else:
                        padded_num_times_spatial = num_times_spatial
                    evt = copy_kern(bottom.ocl_buf, top.ocl_buf).on(
                        queue, (padded_count,), wait_for=wait_for)
                    evt = max_kern(top.ocl_buf, scale.ocl_buf).on(
                        queue, (padded_num_times_spatial, ), wait_for=evt)
                    evt = sub_kern(scale.ocl_buf, top.ocl_buf).on(
                        queue, (padded_count, ), wait_for=evt)
                    evt = exp_kern(top.ocl_buf, top.ocl_buf).on(
                        queue, (padded_count, ), wait_for=evt)
                    evt = sum_kern(top.ocl_buf, scale.ocl_buf).on(
                        queue, (padded_num_times_spatial, ), wait_for=evt)
                    evt = div_kern(scale.ocl_buf, top.ocl_buf).on(
                        queue, (padded_count, ), wait_for=evt)
                    return [evt]

            return SoftmaxLauncher(sources, sinks)
elif backend in {"omp", "openmp"}:
    class SoftmaxForward(DeviceLevel):
        """
        top = SoftmaxForward(bottom)
        """
        @classmethod
        def get_launcher(cls, sources, sinks, keywords, symbol_table):
            bottom = symbol_table[sources[0].name]
            num = bottom.shape[0]
            channels = bottom.shape[1]
            scale_shape = list(bottom.shape)
            scale_shape[1] = 1
            scale = hmarray(tuple(scale_shape))
            spatial_dim = int(np.prod(bottom.shape[2:]))
            count = np.prod(bottom.shape)

            kernels = Template("""
    #include <float.h>
    #include <math.h>
    void kernel_copy(float* data, float* out) {
      #pragma omp parallel for
      for (int index = 0; index < $count; index++) {
        out[index] = data[index];
      }
    }
    void kernel_channel_max(float* data, float* out) {
      #pragma omp parallel for
      for (int index = 0; index < $num_times_spatial; index++) {
        int n = index / $spatial_dim;
        int s = index % $spatial_dim;
        float maxval = -FLT_MAX;
        for (int c = 0; c < $channels; ++c) {
          maxval = fmax(data[(n * $channels + c) * $spatial_dim + s], maxval);
        }
        out[index] = maxval;
      }
    }
    void kernel_channel_subtract(float* channel_max, float* data) {
      #pragma omp parallel for
      for (int index = 0; index < $count; index++) {
        int n = index / $channels / $spatial_dim;
        int s = index % $spatial_dim;
        data[index] -= channel_max[n * $spatial_dim + s];
      }
    }
    void kernel_exp(float* data, float* out) {
      #pragma omp parallel for
      for (int index = 0; index < $count; index++) {
        out[index] = exp(data[index]);
      }
    }
    void kernel_channel_sum(float* data, float* channel_sum) {
      #pragma omp parallel for
      for (int index = 0; index < $num_times_spatial; index++) {
        int n = index / $spatial_dim;
        int s = index % $spatial_dim;
        float sum = 0;
        for (int c = 0; c < $channels; ++c) {
          sum += data[(n * $channels + c) * $spatial_dim + s];
        }
        channel_sum[index] = sum;
      }
    }
    void kernel_channel_div(float* channel_sum, float* data) {
      #pragma omp parallel for
      for (int index = 0; index < $count; index++) {
        int n = index / $channels / $spatial_dim;
        int s = index % $spatial_dim;
        data[index] /= channel_sum[n * $spatial_dim + s];
      }
    }
    """).substitute(count=count, num_times_spatial=num * spatial_dim,
                    channels=channels, spatial_dim=spatial_dim,
                    dim=np.prod(bottom.shape[1:]))
            lib = hm_compile_and_load(kernels)

            copy_kern = lib.kernel_copy
            max_kern = lib.kernel_channel_max
            sub_kern = lib.kernel_channel_subtract
            exp_kern = lib.kernel_exp
            sum_kern = lib.kernel_channel_sum
            div_kern = lib.kernel_channel_div

            class SoftmaxLauncher(object):
                def __init__(self, sources, sinks):
                    self.sources = sources
                    self.sinks = sinks

                def compile(self):
                    pass

                def launch(self, symbol_table, wait_for):
                    bottom = symbol_table[sources[0].name]
                    top = symbol_table[sinks[0].name]
                    copy_kern.argtypes = tuple(
                        np.ctypeslib.ndpointer(p.dtype, p.ndim, p.shape)
                        for p in [bottom, top])
                    copy_kern(bottom, top)
                    max_kern.argtypes = tuple(
                        np.ctypeslib.ndpointer(p.dtype, p.ndim, p.shape)
                        for p in [top, scale])
                    max_kern(top, scale)
                    sub_kern.argtypes = tuple(
                        np.ctypeslib.ndpointer(p.dtype, p.ndim, p.shape)
                        for p in [scale, top])
                    sub_kern(scale, top)
                    exp_kern.argtypes = tuple(
                        np.ctypeslib.ndpointer(p.dtype, p.ndim, p.shape)
                        for p in [top, top])
                    exp_kern(top, top)
                    sum_kern.argtypes = tuple(
                        np.ctypeslib.ndpointer(p.dtype, p.ndim, p.shape)
                        for p in [top, scale])
                    sum_kern(top, scale)
                    div_kern.argtypes = tuple(
                        np.ctypeslib.ndpointer(p.dtype, p.ndim, p.shape)
                        for p in [scale, top])
                    div_kern(scale, top)

            return SoftmaxLauncher(sources, sinks)
