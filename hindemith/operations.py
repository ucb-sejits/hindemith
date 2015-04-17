import numpy as np
from string import Template
from hindemith.types import hmarray
from hindemith.cl import context, queue
from hindemith.clibs.clblas import sgemm, sgemv
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


class PoolBackward(BlockLevel):
    """
    bottom_diff = PoolBackward(top_diff, mask
                               kernel_size=(2, 2),
                               padding=(0, 0),
                               stride=(2, 2))
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        num_work_items = np.prod(sinks[0].shape)
        return (num_work_items, )

    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        channels, height, width = symbol_table[sinks[0]].shape[1:]
        pad_h, pad_w = keywords['padding']
        stride_h, stride_w = keywords['stride']
        kernel_h, kernel_w = keywords['kernel_size']
        pooled_height = ((height + 2 * pad_h - kernel_h) // stride_h) + 1
        pooled_width = ((width + 2 * pad_w - kernel_w) // stride_w) + 1
        return Template("""
    int index = get_global_id(0);
    int w = index % $width;
    int h = (index / $width) % $height;
    int c = (index / $width / $height) % $channels;
    int n = index / $width / $height / $channels;
    int phstart =
        (h + $pad_h < $kernel_h) ? 0 : (h + $pad_h - $kernel_h) / $stride_h + 1;
    int phend = min((h + $pad_h) / $stride_h + 1, $pooled_height);
    int pwstart =
        (w + $pad_w < $kernel_w) ? 0 : (w + $pad_w - $kernel_w) / $stride_w + 1;
    int pwend = min((w + $pad_w) / $stride_w + 1, $pooled_width);
    float gradient = 0;
    int offset = (n * $channels + c) * $pooled_height * $pooled_width;
    $top_diff += offset;
    $mask += offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if ($mask[ph * $pooled_width + pw] == h * $width + w) {
          gradient += $top_diff[ph * $pooled_width + pw];
        }
      }
    }
    $bottom_diff[index] = gradient;        
""").substitute(bottom_diff=sinks[0], mask=sources[1], top_diff=sources[0],
                pooled_height=pooled_height, pooled_width=pooled_width,
                channels=channels, stride_h=stride_h,
                stride_w=stride_w, pad_h=pad_h, pad_w=pad_w,
                kernel_h=kernel_h, kernel_w=kernel_w, height=height,
                width=width)


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

        im2col_global_size = channels * height_col * width_col

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
                           i * bot_offset).on(queue, (im2col_global_size, ))
                    m = weights.shape[0]
                    n = np.prod(top.shape[2:])
                    k = weights.shape[1]
                    sgemm(False, False, 1.0, weights, 0, k, col_data,
                          0, n, 0.0, top, i * top_offset, n, m, n, k)
                    sgemm(False, False, 1.0, bias, 0, 1,
                          bias_multiplier, 0, n, 1.0, top, i *
                          top_offset, n, m, n, 1)
        return ConvLauncher()


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
            def compile(self):
                pass

            def launch(self, symbol_table):
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
        return ConvLauncher()


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
                max_kern(prob.ocl_buf, scale.ocl_buf).on(queue, (num * spatial_dim, ))
                sub_kern(scale.ocl_buf, prob.ocl_buf).on(queue, (count, ))
                exp_kern(prob.ocl_buf, prob.ocl_buf).on(queue, (count, ))
                sum_kern(prob.ocl_buf, scale.ocl_buf).on(queue, (num * spatial_dim, ))
                div_kern(scale.ocl_buf, prob.ocl_buf).on(queue, (count, ))
                loss_forward(prob.ocl_buf, label.ocl_buf, loss.ocl_buf).on(queue, (num * spatial_dim, ))
                loss.sync_host()
                top[0] = np.sum(loss) / np.float32(num)

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
                top_diff = symbol_table[sources[0]]
                label = symbol_table[sources[1]]
                prob = symbol_table[sources[2]]
                copy(prob.ocl_buf, bottom_diff.ocl_buf).on(
                    queue, (np.prod(prob.shape),))
                backward(label.ocl_buf, bottom_diff.ocl_buf).on(
                    queue, (num * spatial_dim), )
                loss_weight = top_diff[0]
                scale(np.float32(loss_weight / float(count)), bottom_diff.ocl_buf).on(
                    queue, (np.prod(prob.shape), ))
        return Launcher()
