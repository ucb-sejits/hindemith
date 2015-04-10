from hindemith.operations.core import DeviceLevel, register_operation
from hindemith.types import NDArray
from hindemith.cl import context, queue
from string import Template
import numpy as np
import pycl as cl
import ctypes as ct
import ast
try:
    clBLAS_lib = ct.util.find_library("clBLAS")
    try:
        _clblaslib = ct.cdll.LoadLibrary(clBLAS_lib)
    except OSError:
        _clblaslib = ct.cdll.LoadLibrary("/usr/local/lib64/" + clBLAS_lib)
except OSError:
    raise Exception("Could not find clBLAS, please install it and add"
                    "it to your LD_LIBRARY_PATH or DYLD_LIBRARY_PATH")

err = _clblaslib.clblasSetup()


def sgemm(transA, transB, alpha, A, A_offset, B, B_offset, beta, C, C_offset,
          m, n, k):
    cblas_row_major = ct.c_int(0)
    transA = ct.c_int(1 if transA else 0)
    transB = ct.c_int(1 if transB else 0)
    m = ct.c_size_t(int(m))
    n = ct.c_size_t(int(n))
    k = ct.c_size_t(int(k))
    alpha = ct.c_float(alpha)
    beta = ct.c_float(beta)
    err = _clblaslib.clblasSgemm(cblas_row_major, transA, transB, m, n, k,
                                 alpha, A.ocl_buf, ct.c_size_t(A_offset), k,
                                 B.ocl_buf, ct.c_size_t(B_offset), n, beta,
                                 C.ocl_buf, ct.c_size_t(C_offset), n,
                                 ct.c_size_t(1), ct.byref(queue),
                                 ct.c_size_t(0), None, None)
    if err:
        raise Exception("clBLAS sgemm returned error code {}".format(err))


class ConvForward(DeviceLevel):
    def __init__(self, statement, symbol_table):
        self.symbol_table = symbol_table
        self.statement = statement
        self.operand_name = statement.value.args[0].id
        self.operand = self.symbol_table[self.operand_name]
        self.weights_name = statement.value.args[1].id
        self.weights = self.symbol_table[self.weights_name]
        self.sources = [self.operand_name, self.weights_name]
        for keyword in statement.value.keywords:
            if keyword.arg == 'kernel_size':
                self.kernel_h, self.kernel_w = tuple(
                    elt.n for elt in keyword.value.elts)
            elif keyword.arg == 'padding':
                self.pad_h, self.pad_w = tuple(
                    elt.n for elt in keyword.value.elts)
            elif keyword.arg == 'stride':
                self.stride_h, self.stride_w = tuple(
                    elt.n for elt in keyword.value.elts)
            else:
                raise Exception("Unsupport keyword arg to Conv", keyword.arg)
        num, channels, height, width = self.operand.shape
        self.channels_col = channels * self.kernel_h * self.kernel_w
        self.height_col = (height + 2 * self.pad_h - self.kernel_h) // \
            self.stride_h + 1
        self.width_col = (width + 2 * self.pad_w - self.kernel_w) // \
            self.stride_w + 1
        self.col_data_name, self.col_data = \
            NDArray.unique((self.channels_col, self.height_col *
                            self.width_col), np.float32)
        self.symbol_table[self.col_data_name] = self.col_data

        self.target_name = statement.targets[0].id
        self.target = symbol_table[self.target_name]
        self.sinks = [self.target_name, self.col_data_name]

    def compile(self):
        im2col_global_size = (self.operand.shape[1] * self.height_col *
                              self.width_col, )
        im2col_kernel = Template("""
// @begin=cl@
__kernel void im2col(global const float* $data_im, global float* $data_col,
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
    global float* data_col_ptr = $data_col;
    data_col_ptr += (channel_out * $height_col + h_out) * $width_col + w_out;
    global const float* data_im_ptr = $data_im + bot_offset;
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
// @end=cl@
""").substitute(data_im=self.operand_name, data_col=self.target_name,
                height_col=self.height_col, width_col=self.width_col,
                channels_col=self.channels_col, height=self.operand.shape[2],
                width=self.operand.shape[3], kernel_h=self.kernel_h,
                kernel_w=self.kernel_w, stride_h=self.stride_h,
                stride_w=self.stride_w, pad_h=self.pad_h, pad_w=self.pad_w,
                global_size=im2col_global_size[0])
        im2col = cl.clCreateProgramWithSource(
            context, im2col_kernel).build()['im2col']
        im2col.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_int)

        class ConvLauncher(object):
            def __init__(self, op):
                self.op = op

            def launch(self, env):
                bottom = env[self.op.operand_name]
                bottom.sync_ocl()
                self.op.weights.sync_ocl()
                top = env[self.op.target_name]
                top_offset = np.prod(top.shape[1:])
                bot_offset = np.prod(bottom.shape[1:])
                for i in range(bottom.shape[0]):
                    im2col(bottom.ocl_buf, self.op.col_data.ocl_buf, i
                           * bot_offset).on(queue, im2col_global_size)
                    sgemm(False, False,
                          1.0, self.op.weights, 0, self.op.col_data, 0,
                          0.0, top, i * top_offset,
                          self.op.weights.shape[0],
                          np.prod(top.shape[2:]),
                          self.op.weights.shape[1])
                top.host_dirty = True
        return [ConvLauncher(self)]

        # return body, global_size, self.sources, self.sinks

    @classmethod
    def match(cls, node, symbol_table):
        if not isinstance(node, ast.Assign):
            return False
        node = node.value
        if isinstance(node, ast.Call):
            return isinstance(node.func, ast.Name) and \
                node.func.id == 'ConvForward'


register_operation(ConvForward)


class ConvBackward(DeviceLevel):
    def __init__(self, statement, symbol_table):
        """
        ConvBackward(bottom, top_diff, weights, learning_rate, )
        """
        self.symbol_table = symbol_table
        self.statement = statement
        self.bottom_name = statement.value.args[0].id
        self.bottom = self.symbol_table[self.bottom_name]
        self.top_diff_name = statement.value.args[1].id
        self.top_diff = self.symbol_table[self.top_diff_name]
        self.weights_name = statement.value.args[2].id
        self.weights = self.symbol_table[self.weights_name]
        self.weights_diff_name = statement.value.args[3].id
        self.weights_diff = self.symbol_table[self.weights_diff_name]

        self.sources = [self.bottom_name, self.top_diff_name,
                        self.weights_name]

        for keyword in statement.value.keywords:
            if keyword.arg == 'kernel_size':
                self.kernel_h, self.kernel_w = tuple(
                    elt.n for elt in keyword.value.elts)
            elif keyword.arg == 'padding':
                self.pad_h, self.pad_w = tuple(
                    elt.n for elt in keyword.value.elts)
            elif keyword.arg == 'stride':
                self.stride_h, self.stride_w = tuple(
                    elt.n for elt in keyword.value.elts)
            # elif keyword.arg == 'learning_rate':
            #     # TODO: Support numbers and symbols
            #     self.learning_rate = self.symbol_table[keyword.value.id]
            else:
                raise Exception("Unsupport keyword arg to Conv", keyword.arg)

        num, channels, height, width = self.bottom.shape
        self.channels_col = channels * self.kernel_h * self.kernel_w
        self.height_col = (height + 2 * self.pad_h - self.kernel_h) // \
            self.stride_h + 1
        self.width_col = (width + 2 * self.pad_w - self.kernel_w) // \
            self.stride_w + 1
        self.col_data_name, self.col_data = \
            NDArray.unique((self.channels_col, self.height_col *
                            self.width_col), np.float32)

        self.bottom_diff_name = statement.targets[0].id
        self.bottom_diff = symbol_table[self.bottom_diff_name]
        self.sinks = [self.bottom_diff_name, self.weights_diff_name]

    def compile(self):
        im2col_global_size = (self.bottom.shape[1] * self.height_col *
                              self.width_col, )
        col2im_global_size = (np.prod(self.bottom.shape[1:]), )
# s/{\([^}]*\)}/$\1/g
        kernels = Template("""
// @begin=cl@
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
  if (get_global_id(0) < $im2col_global_size) {
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
// @end=cl@
""").substitute(height_col=self.height_col, width_col=self.width_col,
                channels_col=self.channels_col, height=self.bottom.shape[2],
                width=self.bottom.shape[3], kernel_h=self.kernel_h,
                kernel_w=self.kernel_w, stride_h=self.stride_h,
                stride_w=self.stride_w, pad_h=self.pad_h, pad_w=self.pad_w,
                im2col_global_size=im2col_global_size[0],
                col2im_global_size=col2im_global_size[0])
        program = cl.clCreateProgramWithSource(context, kernels).build()
        col2im = program['col2im']
        col2im.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_int)
        im2col = program['im2col']
        im2col.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_int)

        class ConvLauncher(object):
            def __init__(self, op):
                self.op = op

            def launch(self, env):
                bottom = env[self.op.bottom_name]
                bottom.sync_ocl()
                bot_offset = np.prod(bottom.shape[1:])
                self.op.top_diff.sync_ocl()
                top_offset = np.prod(self.op.top_diff.shape[1:])
                for i in range(self.op.top_diff.shape[0]):
                    im2col(bottom.ocl_buf, self.op.col_data.ocl_buf, i
                           * bot_offset).on(queue, im2col_global_size)
                    # FIXME: Passing transpose to sgemm causes error,
                    # have to do it here first for now
                    self.op.col_data.host_dirty = True
                    self.op.col_data.sync_host()
                    buf = np.copy(self.op.col_data).T.view(NDArray)
                    sgemm(False, False, 1.0, self.op.top_diff, i * top_offset,
                          buf, 0, 1.0,
                          self.op.weights_diff, 0,
                          self.op.top_diff.shape[1],
                          buf.shape[1], buf.shape[0])
                    weight_trans = np.copy(self.op.weights).T.view(NDArray)
                    sgemm(False, False, 1.0, weight_trans, 0,
                          self.op.top_diff, i * top_offset, 0.0,
                          self.op.col_data, 0,
                          weight_trans.shape[0],
                          self.op.top_diff.shape[2],
                          weight_trans.shape[1])
                    col2im(self.op.col_data.ocl_buf,
                           self.op.bottom_diff.ocl_buf, i *
                           bot_offset).on(queue, col2im_global_size)
                self.op.weights_diff.host_dirty = True
                self.op.bottom_diff.host_dirty = True
        return [ConvLauncher(self)]

    @classmethod
    def match(cls, node, symbol_table):
        if not isinstance(node, ast.Assign):
            return False
        node = node.value
        if isinstance(node, ast.Call):
            return isinstance(node.func, ast.Name) and \
                node.func.id == 'ConvBackward'

register_operation(ConvBackward)
