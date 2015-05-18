from hindemith.operations.core import DeviceLevel, ElementLevel
from hindemith.types import hmarray
import numpy as np
import pycl as cl
from string import Template
import os
import ctypes as ct
import ast
backend = os.getenv("HM_BACKEND", "ocl")
if backend in {"ocl", "opencl", "OCL"}:
    from hindemith.clibs.clblas import sgemm, sgemv
    from hindemith.cl import context, queues, hm_compile_and_load
else:
    from hindemith.cl import hm_compile_and_load


# class ConvForward(ElementLevel):
#     """
#     top = ConvForward(bottom, weights, bias, kernel_size=(11, 11),
#                       stride=(1, 1), padding=(0, 0))
#     """
#     @classmethod
#     def get_launch_parameters(cls, sources, sinks):
#         num_work_items = np.prod(sinks[0].shape)
#         return (num_work_items, True)
#
#     @classmethod
#     def emit(cls, sources, sinks, keywords, symbol_table):
#         kernel_h, kernel_w = keywords['kernel_size']
#         pad_h, pad_w = keywords['padding']
#         stride_h, stride_w = keywords['stride']
#         num, in_channels, in_height, in_width = symbol_table[sources[0]].shape
#         out_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
#         out_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1
#         out_channels = symbol_table[sinks[0]].shape[1]
#         num_work_items = num * out_channels * out_height * out_width
#         return Template("""
#       int out_x = index % $width_out;
#       int out_y = (index / $width_out) % $height_out;
#       int out_c = (index / $width_out / $height_out) % $channels_out;
#       int n = index / $width_out / $height_out / $channels_out;
#       float tmp = 0.0f;
#       for (int in_c = 0; in_c < $channels_in; in_c++) {
#         #pragma unroll
#         for (int i = 0; i < $kernel_h; i++) {
#           int in_y = out_y * $stride_h - $pad_h + i;
#           #pragma unroll
#           for (int j = 0; j < $kernel_w; j++) {
#             int in_x = out_x * $stride_w - $pad_w + j;
#             if (in_y >= 0 && in_y < $height_in && in_x >= 0 && in_x < $width_in)
#               tmp += $in_data[((n * $channels_in + in_c) * $height_in + in_y) * $width_in + in_x] * $weights[((out_c * $channels_in + in_c) * $kernel_h + i) * $kernel_w + j];
#           }
#         }
#       }
#       $out[index] = tmp + $bias[out_c];
# """).substitute(kernel_h=kernel_h, kernel_w=kernel_w,
#                 pad_h=pad_h, pad_w=pad_w,
#                 stride_h=stride_h, stride_w=stride_w,
#                 channels_in=in_channels, height_in=in_height,
#                 width_in=in_width, channels_out=out_channels,
#                 height_out=out_height, width_out=out_width,
#                 out=sinks[0], in_data=sources[0], weights=sources[1],
#                 bias=sources[2], global_size=num_work_items)


class ConvForward(DeviceLevel):
    """
    top = ConvForward(bottom, weights, bias, kernel_size=(11, 11),
                      stride=(1, 1), padding=(0, 0))
    """
    if backend in {"ocl", "opencl", "OCL"}:
        @classmethod
        def get_launcher(cls, sources, sinks, keywords, symbol_table):
            kernel_h, kernel_w = keywords['kernel_size']
            pad_h, pad_w = keywords['padding']
            stride_h, stride_w = keywords['stride']
            num, channels, height, width = symbol_table[sources[0].name].shape
            channels_col = channels * kernel_h * kernel_w
            # height_col = (height + 2 * pad_h - kernel_h) // stride_h + 1
            # width_col = (width + 2 * pad_w - kernel_w) // stride_w + 1
            out_channels, height_col, width_col = symbol_table[sinks[0].name].shape[1:]
            is_1x1 = kernel_w == 1 and kernel_h == 1 and stride_h == 1 and \
                     stride_w == 1 and pad_h == 0 and pad_w == 0
            if not is_1x1:
                col_datas = [hmarray((channels_col, height_col * width_col))
                            for _ in range(len(queues))]
            bias_multiplier = hmarray(
                (1, np.prod(symbol_table[sinks[0].name].shape[2:])))
            bias_multiplier.fill(1.0)
            bias_multiplier.sync_ocl()

            im2col_global_size = channels * height_col * width_col

            im2col = Template("""
    __kernel void im2col(global const float* data_im, global float* data_col,
                         int bot_offset) {
      if (get_global_id(0) < $global_size) {
        int index = get_global_id(0);
        int h_index = index / $width_col;
        int w_out = index - h_index * $width_col;
        int channel_in = h_index / $height_col;
        int h_out = h_index - channel_in * $height_col;
        int channel_out = channel_in * $kernel_h * $kernel_w;
        int h_in = h_out * $stride_h - $pad_h;
        int w_in = w_out * $stride_w - $pad_w;
        global float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * $height_col + h_out) * $width_col + w_out;
        global const float* data_im_ptr = data_im + bot_offset;
        data_im_ptr += (channel_in * $height + h_in) * $width + w_in;
        #pragma unroll
        for (int i = 0; i < $kernel_h; ++i) {
          #pragma unroll
          for (int j = 0; j < $kernel_w; ++j) {
            int h = h_in + i;
            int w = w_in + j;
            *data_col_ptr = (h >= 0 && w >= 0 && h < $height && w < $width) ?
                data_im_ptr[i * $width + j] : 0;
            data_col_ptr += $height_col * $width_col;
          }
        }
      }
    }
    """).substitute(global_size=im2col_global_size, stride_h=stride_h,
                    stride_w=stride_w, pad_h=pad_h, pad_w=pad_w,
                    kernel_h=kernel_h, kernel_w=kernel_w, width=width,
                    height=height, height_col=height_col,
                    width_col=width_col)

            im2col = cl.clCreateProgramWithSource(
                context, im2col
            ).build()['im2col']
            im2col.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_int)

            if im2col_global_size % 64:
                padded = (im2col_global_size + 63) & (~63)
            else:
                padded = im2col_global_size

            class ConvLauncher(object):
                def __init__(self, sources, sinks):
                    self.sources = sources
                    self.sinks = sinks

                def compile(self):
                    pass

                def launch(self, symbol_table, wait_for):
                    bottom = symbol_table[sources[0].name]
                    bot_offset = np.prod(bottom.shape[1:])
                    weights = symbol_table[sources[1].name]
                    bias = symbol_table[sources[2].name]
                    top = symbol_table[sinks[0].name]
                    top_offset = np.prod(top.shape[1:])
                    m = weights.shape[0]
                    n = np.prod(top.shape[2:])
                    k = np.prod(weights.shape[1:])
                    # cl.clFinish(queues[0])
                    evts = []
                    if is_1x1:
                        for i in range(bottom.shape[0]):
                            evt = sgemm(False, False, 1.0, weights, 0, k,
                                        bottom, i * bot_offset, n, 0.0,
                                        top, i * top_offset, n, m, n,
                                        k, queues[i % len(queues)], wait_for=wait_for)
                            evt = sgemm(False, False, 1.0, bias, 0, 1,
                                        bias_multiplier, 0, n, 1.0, top, i *
                                        top_offset, n, m, n, 1, queues[i % len(queues)], wait_for=evt)
                            evts.append(evt)
                    else:
                        for i in range(bottom.shape[0]):
                            evt = im2col(bottom.ocl_buf,
                                        col_datas[i % len(queues)].ocl_buf,
                                        i * bot_offset
                                        ).on(queues[i % len(queues)], (padded, ),
                                            wait_for=wait_for)
                            evt = sgemm(False, False, 1.0, weights, 0, k,
                                        col_datas[i % len(queues)],
                                        0, n, 0.0, top, i * top_offset, n, m, n,
                                        k, queues[i % len(queues)], wait_for=evt)
                            evt = sgemm(False, False, 1.0, bias, 0, 1,
                                        bias_multiplier, 0, n, 1.0, top, i *
                                        top_offset, n, m, n, 1, queues[i % len(queues)], wait_for=evt)
                            evts.append(evt)
                    return evts
                    # for q in queues:
                    #     cl.clFinish(q)
            return ConvLauncher(sources, sinks)
    elif backend in {"omp", "openmp"}:
        @classmethod
        def get_launcher(cls, sources, sinks, keywords, symbol_table):
            kernel_h, kernel_w = keywords['kernel_size']
            pad_h, pad_w = keywords['padding']
            stride_h, stride_w = keywords['stride']
            num, channels, height, width = symbol_table[sources[0].name].shape
            channels_col = channels * kernel_h * kernel_w
            height_col = (height + 2 * pad_h - kernel_h) // stride_h + 1
            width_col = (width + 2 * pad_w - kernel_w) // stride_w + 1
            out_channels, height_col, width_col = symbol_table[sinks[0].name].shape[1:]
            col_data = hmarray((channels_col, height_col * width_col))
            bias_multiplier = hmarray(
                (1, np.prod(symbol_table[sinks[0].name].shape[2:])))
            bias_multiplier.fill(1.0)
            bias_multiplier.sync_ocl()

            im2col_global_size = channels * height_col * width_col

            im2col = Template("""
    void im2col(float* data_im, float* data_col, int bot_offset) {
      #pragma omp parallel for
      for (int index = 0; index < $global_size; index++) {
        int h_index = index / $width_col;
        int w_out = index - h_index * $width_col;
        int channel_in = h_index / $height_col;
        int h_out = h_index - channel_in * $height_col;
        int channel_out = channel_in * $kernel_h * $kernel_w;
        int h_in = h_out * $stride_h - $pad_h;
        int w_in = w_out * $stride_w - $pad_w;
        float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * $height_col + h_out) * $width_col + w_out;
        const float* data_im_ptr = data_im + bot_offset;
        data_im_ptr += (channel_in * $height + h_in) * $width + w_in;
        for (int i = 0; i < $kernel_h; ++i) {
          for (int j = 0; j < $kernel_w; ++j) {
            int h = h_in + i;
            int w = w_in + j;
            *data_col_ptr = (h >= 0 && w >= 0 && h < $height && w < $width) ?
                data_im_ptr[i * $width + j] : 0;
            data_col_ptr += $height_col * $width_col;
          }
        }
      }
    }
    """).substitute(global_size=im2col_global_size, stride_h=stride_h,
                    stride_w=stride_w, pad_h=pad_h, pad_w=pad_w,
                    kernel_h=kernel_h, kernel_w=kernel_w, width=width,
                    height=height, height_col=height_col,
                    width_col=width_col)
            lib = hm_compile_and_load(im2col)
            im2col = lib.im2col


            class ConvLauncher(object):
                def __init__(self, sources, sinks):
                    self.sources = sources
                    self.sinks = sinks

                def compile(self):
                    pass

                def launch(self, symbol_table, wait_for):
                    bottom = symbol_table[sources[0].name]
                    bot_offset = np.prod(bottom.shape[1:])
                    weights = symbol_table[sources[1].name]
                    bias = symbol_table[sources[2].name]
                    top = symbol_table[sinks[0].name]
                    im2col.argtypes = tuple(
                        np.ctypeslib.ndpointer(p.dtype, p.ndim, p.shape) for p in
                        [bottom, col_data]) + (ct.c_int, )
                    if len(weights.shape) > 2:
                        weights = weights.reshape(weights.shape[0], np.prod(weights.shape[1:]))
                    for i in range(bottom.shape[0]):
                        im2col(bottom, col_data, i * bot_offset)
                        top[i] = weights.dot(col_data).reshape(weights.shape[0], height_col, width_col)
                        top[i] += bias[:, np.newaxis, np.newaxis]
            return ConvLauncher(sources, sinks)


class ConvBackward(DeviceLevel):
    """
    bottom_diff, weights_diff, bias_diff = \
        ConvBackward(bottom, top_diff, weights,
                     kernel_size=(11, 11), stride=(1, 1), padding=(0, 0))
    """
    @classmethod
    def get_launcher(cls, sources, sinks, keywords, symbol_table):
        kernel_h, kernel_w = keywords['kernel_size']
        pad_h, pad_w = keywords['padding']
        stride_h, stride_w = keywords['stride']
        num, channels, height, width = symbol_table[sources[0]].shape
        channels_col = channels * kernel_h * kernel_w
        height_col = (height + 2 * pad_h - kernel_h) // stride_h + 1
        width_col = (width + 2 * pad_w - kernel_w) // stride_w + 1
        col_data = hmarray((channels_col, height_col * width_col))
        bias_multiplier = hmarray(
            (1, np.prod(symbol_table[sinks[0]].shape[2:])))
        bias_multiplier.fill(1.0)
        bias_multiplier.sync_ocl()

        im2col_global_size = channels * height_col * width_col
        col2im_global_size = channels * height * width

        kernels = Template("""
__kernel void col2im(global float* data_col, global float* data_im,
                     int im_offset) {
  if (get_global_id(0) < $col2im_global_size) {
    int index = get_global_id(0);
    float val = 0;
    int w = index % $width + $pad_w;
    int h = (index / $width) % $height + $pad_h;
    int c = index / ($width * $height);
    // compute the start and end of the output
    int w_col_start = (w < $kernel_w) ? 0 : (w - $kernel_w) / $stride_w + 1;
    int w_col_end = min(w / $stride_w + 1, $width_col);
    int h_col_start = (h < $kernel_h) ? 0 : (h - $kernel_h) / $stride_h + 1;
    int h_col_end = min(h / $stride_h + 1, $height_col);
    // equivalent implementation
    int offset = (c * $kernel_h * $kernel_w + h * $kernel_w + w) * \
          $height_col * $width_col;
    int coeff_h_col = (1 - $stride_h * $kernel_w * $height_col) * \
          $width_col;
    int coeff_w_col = (1 - $stride_w * $height_col * $width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
          val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[im_offset + index] = val;
  }
}
__kernel void im2col(global const float* data_im, global float* data_col,
                     int bot_offset) {
  if (get_global_id(0) < $global_size) {
    int index = get_global_id(0);
    int w_out = index % $width_col;
    int h_index = index / $width_col;
    int h_out = h_index % $height_col;
    int channel_in = h_index / $height_col;
    int channel_out = channel_in * $kernel_h * $kernel_w;
    int h_in = h_out * $stride_h - $pad_h;
    int w_in = w_out * $stride_w - $pad_w;
    global float* data_col_ptr = data_col;
    data_col_ptr += (channel_out * $height_col + h_out) * $width_col + w_out;
    global const float* data_im_ptr = data_im + bot_offset;
    data_im_ptr += (channel_in * $height + h_in) * $width + w_in;
    for (int i = 0; i < $kernel_h; ++i) {
      for (int j = 0; j < $kernel_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col_ptr = (h >= 0 && w >= 0 && h < $height && w < $width) ?
            data_im_ptr[i * $width + j] : 0;
        data_col_ptr += $height_col * $width_col;
      }
    }
  }
}
""").substitute(global_size=im2col_global_size, stride_h=stride_h,
                stride_w=stride_w, pad_h=pad_h, pad_w=pad_w,
                kernel_h=kernel_h, kernel_w=kernel_w, width=width,
                height=height, height_col=height_col,
                width_col=width_col, col2im_global_size=col2im_global_size)

        program = cl.clCreateProgramWithSource(context, kernels).build()
        im2col = program['im2col']
        im2col.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_int)
        col2im = program['col2im']
        col2im.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_int)

        class ConvLauncher(object):
            def __init__(self, sources, sinks):
                self.sources = [ast.Name(s, ast.Load()) for s in sources]
                self.sinks = [ast.Name(s, ast.Load()) for s in sinks]

            def compile(self):
                pass

            def launch(self, symbol_table, wait_for):
                queue = queues[0]
                bottom = symbol_table[sources[0]]
                bot_offset = np.prod(bottom.shape[1:])
                top_diff = symbol_table[sources[1]]
                top_offset = np.prod(top_diff.shape[1:])
                weights = symbol_table[sources[2]]
                bottom_diff = symbol_table[sinks[0]]
                bottom_diff.fill(0)
                bottom_diff.sync_ocl()
                weights_diff = symbol_table[sinks[1]]
                weights_diff.fill(0)
                weights_diff.sync_ocl()
                bias_diff = symbol_table[sinks[2]]
                bias_diff.fill(0)
                bias_diff.sync_ocl()
                for i in range(bottom.shape[0]):
                    n = np.prod(top_diff.shape[2:])
                    sgemv(False, top_diff.shape[1],
                          n, 1.0, top_diff, i *
                          top_offset, n, bias_multiplier, 0, 1, 1.0,
                          bias_diff, 0, 1)
                    im2col(bottom.ocl_buf, col_data.ocl_buf, i
                           * bot_offset).on(queue, im2col_global_size)
                    m = top_diff.shape[1]
                    n = col_data.shape[0]
                    k = col_data.shape[1]

                    sgemm(False, True, 1.0, top_diff, i *
                          top_offset, k, col_data, 0, k, 1.0,
                          weights_diff, 0, n, m, n, k)

                    m = weights.shape[1]
                    n = col_data.shape[1]
                    k = weights.shape[0]

                    sgemm(True, False, 1.0, weights, 0, m,
                          top_diff, i * top_offset, n, 0.0,
                          col_data, 0, n,
                          m, n, k)
                    col2im(col_data.ocl_buf,
                           bottom_diff.ocl_buf, i *
                           bot_offset).on(queue, col2im_global_size)

        return ConvLauncher(sources, sinks)
