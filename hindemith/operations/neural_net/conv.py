from hindemith.operations.core import DeviceLevel, register_operation
from hindemith.types import NDArray
from hindemith.cl import context, queue
import numpy as np
import pycl as cl
import ctypes as ct
import ast
_clblaslib = ct.cdll.LoadLibrary(ct.util.find_library("clBLAS"))
err = _clblaslib.clblasSetup()

def sgemm(A, A_offset, B, B_offset, C, C_offset, m, n, k):
    cblas_row_major = ct.c_int(0)
    no_trans = ct.c_int(0)
    m = ct.c_size_t(int(m))
    n = ct.c_size_t(int(n))
    k = ct.c_size_t(int(k))
    one = ct.c_float(1.0)
    zero = ct.c_float(0.0)
    _clblaslib.clblasSgemm(cblas_row_major, no_trans, no_trans, m, n, k,
                           one, A.ocl_buf, ct.c_size_t(A_offset), k, B.ocl_buf,
                           ct.c_size_t(B_offset), n, zero, C.ocl_buf,
                           ct.c_size_t(C_offset), n,
                           ct.c_size_t(1), ct.byref(queue), ct.c_size_t(0),
                           None, None)


class Conv(DeviceLevel):
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
        self.height_col = (height + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
        self.width_col = (width + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1
        self.col_data_name, self.col_data = \
            NDArray.unique((self.channels_col, self.height_col * self.width_col), np.float32)
        self.symbol_table[self.col_data_name] = self.col_data

        self.target_name = statement.targets[0].id
        self.target = symbol_table[self.target_name]
        self.sinks = [self.target_name, self.col_data_name]

    def compile(self):
        im2col_global_size = (self.operand.shape[1] * self.height_col * self.width_col, )
        im2col_kernel = """
__kernel void im2col(global const float* {data_im}, global float* {data_col}, int bot_offset) {{
  if (get_global_id(0) < {global_size}) {{
    int index = get_global_id(0);
    int w_out = index % {width_col};
    int h_index = index / {width_col};
    int h_out = h_index % {height_col};
    int channel_in = h_index / {height_col};
    int channel_out = channel_in * {kernel_h} * {kernel_w};
    int h_in = h_out * {stride_h} - {pad_h};
    int w_in = w_out * {stride_w} - {pad_w};
    global float* data_col_ptr = {data_col};
    data_col_ptr += (channel_out * {height_col} + h_out) * {width_col} + w_out;
    global const float* data_im_ptr = {data_im} + bot_offset;
    data_im_ptr += (channel_in * {height} + h_in) * {width} + w_in;
    for (int i = 0; i < {kernel_h}; ++i) {{
      for (int j = 0; j < {kernel_w}; ++j) {{
        int h = h_in + i;
        int w = w_in + j;
        *data_col_ptr = (h >= 0 && w >= 0 && h < {height} && w < {width}) ?
            data_im_ptr[i * {width} + j] : 0;
        data_col_ptr += {height_col} * {width_col};
      }}
    }}
  }}
}}
""".format(data_im=self.operand_name, 
           data_col=self.target_name, height_col=self.height_col,
           width_col=self.width_col, channels_col=self.channels_col,
           height=self.operand.shape[2], width=self.operand.shape[3],
           kernel_h=self.kernel_h, kernel_w=self.kernel_w,
           stride_h=self.stride_h, stride_w=self.stride_w,
           pad_h=self.pad_h, pad_w=self.pad_w,
           global_size=im2col_global_size[0])
        im2col = cl.clCreateProgramWithSource(context, im2col_kernel).build()['im2col']
        im2col.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_int)

        class ConvLauncher(object):
            def __init__(self, op):
                self.op = op
            def launch(self, env):
                bottom = env[self.op.operand_name]
                bottom.sync()
                top = env[self.op.target_name]
                top_offset = np.prod(top.shape[1:])
                bot_offset = np.prod(bottom.shape[1:])
                top.host_dirty = True
                for i in range(bottom.shape[0]):
                    im2col(bottom.ocl_buf, self.op.col_data.ocl_buf, i
                    * bot_offset).on(queue, im2col_global_size)
                    sgemm(self.op.weights, 0, self.op.col_data, 0,
                             top, i * top_offset,
                             self.op.weights.shape[0],
                             self.op.col_data.shape[1],
                             self.op.weights.shape[1])
        return [ConvLauncher(self)]

        # return body, global_size, self.sources, self.sinks

    @classmethod
    def match(cls, node, symbol_table):
        if not isinstance(node, ast.Assign):
            return False
        node = node.value
        if isinstance(node, ast.Call):
            return isinstance(node.func, ast.Name) and \
                node.func.id == 'Conv'


register_operation(Conv)
