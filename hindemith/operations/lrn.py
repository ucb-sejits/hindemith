from hindemith.operations.core import DeviceLevel
from hindemith.types import hmarray
from hindemith.cl import context, queue
import numpy as np
import pycl as cl
from string import Template


class LrnForward(DeviceLevel):
    """
    top, scale = LrnForward(bottom, alpha=alpha, beta=beta,
                            local_size=local_size, k=1)
    """
    @classmethod
    def get_launcher(cls, sources, sinks, keywords, symbol_table):
        num, channels, height, width = symbol_table[sources[0]].shape
        local_size = keywords['local_size']
        alpha = keywords['alpha']
        k = keywords['k']
        beta = keywords['beta']
        kernel = Template("""
__kernel void LRNFillScale(global const float* in, global float* scale) {
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
__kernel void LRNComputeOutput(global const float* in,
                               global const float* scale,
                               global float* out) {
  int index = get_global_id(0);
  out[index] = in[index] * pow(scale[index], (float)$negative_beta);
}
""").substitute(width=width, height=height, channels=channels,
                local_size=local_size, alpha_over_size=float(alpha) / local_size,
                k=k, negative_beta=-beta)
        program = cl.clCreateProgramWithSource(
            context, kernel
        ).build()
        fill_kern = program['LRNFillScale']
        fill_kern.argtypes = (cl.cl_mem, cl.cl_mem)

        fill_global = (num * height * width, )

        compute_kern = program['LRNComputeOutput']
        compute_kern.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_mem)

        compute_global = (num * channels * height * width, )

        class LrnLauncher(object):
            def compile(self):
                pass

            def launch(self, symbol_table):
                bottom = symbol_table[sources[0]]
                top = symbol_table[sinks[0]]
                scale = symbol_table[sinks[1]]
                fill_kern(bottom.ocl_buf, scale.ocl_buf).on(queue, fill_global)
                compute_kern(bottom.ocl_buf, scale.ocl_buf, top.ocl_buf).on(queue, compute_global)
        return LrnLauncher()


