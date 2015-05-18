from hindemith.operations.core import DeviceLevel, ElementLevel
from string import Template
import numpy as np
import os
import ast
backend = os.getenv("HM_BACKEND", "ocl")
if backend in {"ocl", "opencl", "OCL"}:
    from hindemith.cl import context, queue, hm_compile_and_load
    import pycl as cl
else:
    from hindemith.cl import hm_compile_and_load


if backend in {"ocl", "opencl", "OCL"}:
    class LrnForward(DeviceLevel):
        """
        top, scale = LrnForward(bottom, alpha=alpha, beta=beta,
                                local_size=local_size, k=1)
        """
        @classmethod
        def get_launcher(cls, sources, sinks, keywords, symbol_table):
            num, channels, height, width = symbol_table[sources[0].name].shape
            local_size = keywords['local_size']
            alpha = keywords['alpha']
            k = keywords['k']
            beta = keywords['beta']
            compute_global = (num * channels * height * width, )
            fill_global = (num * height * width, )
            kernel = Template("""
    // @begin=cl@
    __kernel void LRNFillScale(global const float* in, global float* scale) {
      if (get_global_id(0) < $fill_global) {
        int index = get_global_id(0);
        int w = index % $width;
        int h = (index / $width) % $height;
        int n = index / $width / $height;
        int offset = (n * $channels * $height + h) * $width + w;
        int step = $height * $width;
        in += offset;
        scale += offset;
        int head = 0;
        int pre_pad = ($local_size - 1) / 2;
        int post_pad = $local_size - pre_pad - 1;
        float accum_scale = 0;
        // fill the scale at [n, :, h, w]
        // accumulate values
        while (head < post_pad && head < $channels) {
          accum_scale += in[head * step] * in[head * step];
          ++head;
        }
        // both add and subtract
        while (head < $channels) {
          accum_scale += in[head * step] * in[head * step];
          if (head - $local_size >= 0) {
            accum_scale -= in[(head - $local_size) * step] * \
                in[(head - $local_size) * step];
          }
          scale[(head - post_pad) * step] = $k + accum_scale * $alpha_over_size;
          ++head;
        }
        // subtract only
        while (head < $channels + post_pad) {
          if (head - $local_size >= 0) {
            accum_scale -= in[(head - $local_size) * step] * \
                in[(head - $local_size) * step];
          }
          scale[(head - post_pad) * step] = $k + accum_scale * $alpha_over_size;
          ++head;
        }
      }
    }
    __kernel void LRNComputeOutput(global const float* in,
                                   global const float* scale,
                                   global float* out) {
      if (get_global_id(0) < $compute_global) {
        int index = get_global_id(0);
        out[index] = in[index] * pow(scale[index], (float)$negative_beta);
      }
    }
    // @end=cl@
    """).substitute(width=width, height=height, channels=channels,
                    local_size=local_size,
                    alpha_over_size=float(alpha) / local_size,
                    k=k, negative_beta=-beta, fill_global=fill_global[0],
                    compute_global=compute_global[0])
            program = cl.clCreateProgramWithSource(
                context, kernel
            ).build()
            fill_kern = program['LRNFillScale']
            fill_kern.argtypes = (cl.cl_mem, cl.cl_mem)

            compute_kern = program['LRNComputeOutput']
            compute_kern.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_mem)

            class LrnLauncher(object):
                def __init__(self, sources, sinks):
                    self.sources = sources
                    self.sinks = sinks

                def compile(self):
                    pass

                def launch(self, symbol_table, wait_for):
                    bottom = symbol_table[sources[0].name]
                    top = symbol_table[sinks[0].name]
                    scale = symbol_table[sinks[1].name]
                    if fill_global[0] % 16:
                        padded = (fill_global[0] + 15) & (~15)
                    else:
                        padded = fill_global[0]
                    evt = fill_kern(bottom.ocl_buf, scale.ocl_buf).on(queue, (padded,), wait_for=wait_for)
                    if compute_global[0] % 16:
                        padded = (compute_global[0] + 15) & (~15)
                    else:
                        padded = compute_global[0]
                    evt = compute_kern(bottom.ocl_buf, scale.ocl_buf,
                                       top.ocl_buf).on(queue, (padded,), wait_for=evt)
                    return [evt]
            return LrnLauncher(sources, sinks)
elif backend in {"omp", "openmp"}:
    class LrnForward(DeviceLevel):
        """
        top, scale = LrnForward(bottom, alpha=alpha, beta=beta,
                                local_size=local_size, k=1)
        """
        @classmethod
        def get_launcher(cls, sources, sinks, keywords, symbol_table):
            num, channels, height, width = symbol_table[sources[0].name].shape
            local_size = keywords['local_size']
            alpha = keywords['alpha']
            k = keywords['k']
            beta = keywords['beta']
            compute_global = (num * channels * height * width, )
            fill_global = (num * height * width, )
            kernel = Template("""
    #include <math.h>
    void LRNFillScale(float* in_global, float* scale_global) {
      for (int index = 0; index < $fill_global; index++) {
        int w = index % $width;
        int h = (index / $width) % $height;
        int n = index / $width / $height;
        int offset = (n * $channels * $height + h) * $width + w;
        int step = $height * $width;
        float* in = in_global + offset;
        float* scale = scale_global + offset;
        int head = 0;
        int pre_pad = ($local_size - 1) / 2;
        int post_pad = $local_size - pre_pad - 1;
        float accum_scale = 0;
        // fill the scale at [n, :, h, w]
        // accumulate values
        while (head < post_pad && head < $channels) {
          accum_scale += in[head * step] * in[head * step];
          ++head;
        }
        // both add and subtract
        while (head < $channels) {
          accum_scale += in[head * step] * in[head * step];
          if (head - $local_size >= 0) {
            accum_scale -= in[(head - $local_size) * step] * \
                in[(head - $local_size) * step];
          }
          scale[(head - post_pad) * step] = $k + accum_scale * $alpha_over_size;
          ++head;
        }
        // subtract only
        while (head < $channels + post_pad) {
          if (head - $local_size >= 0) {
            accum_scale -= in[(head - $local_size) * step] * \
                in[(head - $local_size) * step];
          }
          scale[(head - post_pad) * step] = $k + accum_scale * $alpha_over_size;
          ++head;
        }
      }
    }
    void LRNComputeOutput(float* in, float* scale, float* out) {
      for (int index = 0; index < $compute_global; index++) {
        out[index] = in[index] * pow(scale[index], (float)$negative_beta);
      }
    }
    """).substitute(width=width, height=height, channels=channels,
                    local_size=local_size,
                    alpha_over_size=float(alpha) / local_size,
                    k=k, negative_beta=-beta, fill_global=fill_global[0],
                    compute_global=compute_global[0])
            lib = hm_compile_and_load(kernel)
            fill_kern = lib.LRNFillScale

            compute_kern = lib.LRNComputeOutput

            class LrnLauncher(object):
                def __init__(self, sources, sinks):
                    self.sources = sources
                    self.sinks = sinks

                def compile(self):
                    pass

                def launch(self, symbol_table, wait_for):
                    bottom = symbol_table[sources[0].name]
                    top = symbol_table[sinks[0].name]
                    scale = symbol_table[sinks[1].name]
                    fill_kern.argtypes = tuple(
                        np.ctypeslib.ndpointer(p.dtype, p.ndim, p.shape)
                        for p in [bottom, scale])
                    compute_kern.argtypes = tuple(
                        np.ctypeslib.ndpointer(p.dtype, p.ndim, p.shape)
                        for p in [bottom, scale, top])
                    fill_kern(bottom, scale)
                    compute_kern(bottom, scale, top)
            return LrnLauncher(sources, sinks)


class LrnBackward(ElementLevel):
    """
    bottom_diff = LrnBackward(bottom, top, scale, top_diff,
                              alpha=alpha, beta=beta,
                              local_size=local_size)
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        num, _, height, width = sinks[0].shape
        return (num * height * width, )

    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        _, channels, height, width = symbol_table[sinks[0]].shape
        local_size = keywords['local_size']
        alpha = keywords['alpha']
        beta = keywords['beta']
        return Template("""
// @begin=cl@
    int index = get_global_id(0);
    // find out the local offset
    int w = index % $width;
    int h = (index / $width) % $height;
    int n = index / $width / $height;
    int offset = (n * $channels * $height + h) * $width + w;
    int step = $height * $width;
    $bottom_data += offset;
    $top_data += offset;
    $scale += offset;
    $top_diff += offset;
    $bottom_diff += offset;
    int head = 0;
    int pre_pad = $size - ($size + 1) / 2;
    int post_pad = $size - pre_pad - 1;
    float accum_ratio = 0;
    // accumulate values
    while (head < post_pad && head < $channels) {
      accum_ratio += $top_diff[head * step] * $top_data[head * step] /
          $scale[head * step];
      ++head;
    }
    // both add and subtract
    while (head < $channels) {
      accum_ratio += $top_diff[head * step] * $top_data[head * step] /
          $scale[head * step];
      if (head - $size >= 0) {
        accum_ratio -= $top_diff[(head - $size) * step] *
            $top_data[(head - $size) * step] / $scale[(head - $size) * step];
      }
      $bottom_diff[(head - post_pad) * step] = $top_diff[(head - post_pad) * step]
          * pow($scale[(head - post_pad) * step], (float)$negative_beta) - (float)$cache_ratio *
          $bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < $channels + post_pad) {
      if (head - $size >= 0) {
        accum_ratio -= $top_diff[(head - $size) * step] *
            $top_data[(head - $size) * step] / $scale[(head - $size) * step];
      }
      $bottom_diff[(head - post_pad) * step] = $top_diff[(head - post_pad) * step]
          * pow($scale[(head - post_pad) * step], (float)$negative_beta) - (float)$cache_ratio *
          $bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
// @end=cl@
""").substitute(channels=channels, height=height, width=width,
                cache_ratio=2.0 * alpha * beta / local_size,
                negative_beta=-beta, size=local_size,
                bottom_data=sources[0], top_data=sources[1], scale=sources[2],
                top_diff=sources[3], bottom_diff=sinks[0])
