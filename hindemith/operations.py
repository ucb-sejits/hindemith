import numpy as np
from string import Template
from hindemith.types import hmarray
from hindemith.cl import context, queue
from hindemith.clibs.clblas import sgemm
import pycl as cl


class HMUndefinedMethodError(NotImplementedError):
    def __init__(self, cls, method):
        message = "{} operation must implement a {} method".format(cls, method)
        super(HMUndefinedMethodError, self).__init__(message)


class HMOperation(object):
    pass


class DeviceLevel(HMOperation):
    """
    An operation that can run multiple OpenCL kernels without interference.
    """
    @classmethod
    def get_launcher(cls, sources, sinks, symbol_table):
        raise HMUndefinedMethodError(cls, "get_launcher")


class BlockLevel(HMOperation):
    """
    An OpenCL Kernel
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        """
        Return a tuple of parameters used to launch the kernel

        :return: (num_work_items, )
        :rtype: tuple(int)
        """
        raise HMUndefinedMethodError(cls, "get_launch_parameters")

    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        """
        Emit the code to be the body of the OpenCL Kernel.  A
        BlockLevel operation can use any valid OpenCL api calls and
        constructs.

        :param list sources: List of sources as strings
        :param list sinks: List of sinks as strings
        :param dict symbol_table: The current symbol_table

        :returns: String to be appended to kernel body
        :rtype: str
        """
        raise HMUndefinedMethodError(cls, "emit")


class ElementLevel(HMOperation):
    """
    An operation that does not communicate across work items.
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        """
        Return a tuple of parameters used to launch the kernel

        :return: (num_work_items, )
        :rtype: tuple(int)
        """
        raise HMUndefinedMethodError(cls, "get_launch_parameters")

    @classmethod
    def emit(cls, sources, sinks, symbol_table):
        """
        Emit the code to be inserted into the body of the Kernel,
        ElementLevel operations are not allowed to communicate across
        work items (i.e. using barriers or local memory).  If done,
        behavior is undefined.

        :param list sources: List of sources as strings
        :param list sinks: List of sinks as strings
        :param dict symbol_table: The current symbol_table

        :returns: String to be appended to kernel body
        :rtype: str
        """
        raise HMUndefinedMethodError(cls, "emit")


class Relu(ElementLevel):
    """
    top = Relu(bottom)
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        num_work_items = np.prod(sources[0].shape)
        return (num_work_items, )

    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        return Template(
            "$target[get_global_id(0)] = $operand[get_global_id(0)] > 0 ? "
            "$operand[get_global_id(0)] : 0;"
        ).substitute(target=sinks[0], operand=sources[0])


class PoolForward(BlockLevel):
    """
    top, mask = PoolForward(bottom)
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        num_work_items = np.prod(sinks[0].shape)
        return (num_work_items, )

    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        channels, height, width = symbol_table[sources[0]].shape[1:]
        pad_h, pad_w = keywords['padding']
        stride_h, stride_w = keywords['stride']
        kernel_h, kernel_w = keywords['kernel_size']
        pooled_height = ((height + 2 * pad_h - kernel_h) // stride_h) + 1
        pooled_width = ((width + 2 * pad_w - kernel_w) // stride_w) + 1
        return Template("""
    int index = get_global_id(0);
    int pw = index % $pooled_w;
    int ph = (index / $pooled_w) % $pooled_h;
    int c = (index / $pooled_w / $pooled_h) % $channels;
    int n = index / $pooled_w / $pooled_h / $channels;
    int hstart = ph * $stride - $pad;
    int wstart = pw * $stride - $pad;
    int hend = min(hstart + $kernel_h, $height);
    int wend = min(wstart + $kernel_w, $width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    float maxval = -FLT_MAX;
    int maxidx = -1;
    $bottom += (n * $channels + c) * $height * $width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if ($bottom[h * $width + w] > maxval) {
          maxidx = h * $width + w;
          maxval = $bottom[maxidx];
        }
      }
    }
    $top[index] = maxval;
    $mask[index] = maxidx;
""").substitute(top=sinks[0], mask=sinks[1], bottom=sources[0],
                pooled_h=pooled_height, pooled_w=pooled_width,
                channels=channels, stride=stride_h, pad=pad_h,
                kernel_h=kernel_h, kernel_w=kernel_w,
                height=height, width=width)


class ConvForward(DeviceLevel):
    """
    top = ConvForward(bottom, weights, bias, kernel_size=(11, 11),
                      stride=(1, 1), padding=(0, 0))
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

        weights = symbol_table[sources[1]]
        im2col_global_size = weights.shape[1] * height_col * width_col

        im2col = Template("""
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
                width_col=width_col)

        im2col = cl.clCreateProgramWithSource(
            context, im2col
        ).build()['im2col']
        im2col.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_int)

        class ConvLauncher(object):
            def compile(self):
                pass

            def launch(self, symbol_table):
                bottom = symbol_table[sources[0]]
                bot_offset = np.prod(bottom.shape[1:])
                weights = symbol_table[sources[1]]
                bias = symbol_table[sources[2]]
                top = symbol_table[sinks[0]]
                top_offset = np.prod(top.shape[1:])
                for i in range(bottom.shape[0]):
                    im2col(bottom.ocl_buf, col_data.ocl_buf,
                           i * bot_offset).on(queue, im2col_global_size)
                    m = weights.shape[0]
                    n = np.prod(top.shape[2:])
                    k = weights.shape[1]
                    sgemm(False, False, 1.0, weights, 0, k, col_data,
                          0, n, 0.0, top, i * top_offset, n, m, n, k)
                    sgemm(False, False, 1.0, bias, 0, 1,
                          bias_multiplier, 0, n, 1.0, top, i *
                          top_offset, n, m, n, 1)
        return ConvLauncher()
