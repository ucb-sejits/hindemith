from hindemith.operations.core import ElementLevel, register_operation
import numpy as np
import ast


class Im2Col(ElementLevel):
    def __init__(self, statement, symbol_table):
        self.symbol_table = symbol_table
        self.statement = statement
        self.operand_name = statement.value.args[0].id
        self.operand = self.symbol_table[self.operand_name]
        self.sources = [self.operand_name]
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
                raise Exception("Unsupport keyword arg to Pool", keyword.arg)
        channels, height, width = self.operand.shape
        self.channels_col = channels * self.kernel_h * self.kernel_w
        self.height_col = (height + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
        self.width_col = (width + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1

        self.target_name = statement.targets[0].id
        self.target = symbol_table[self.target_name]
        self.sinks = [self.target_name]

    def compile(self):
        body = """
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
    global const float* data_im_ptr = {data_im};
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
""".format(data_im=self.operand_name, 
           data_col=self.target_name, height_col=self.height_col,
           width_col=self.width_col, channels_col=self.channels_col,
           height=self.operand.shape[1], width=self.operand.shape[2],
           kernel_h=self.kernel_h, kernel_w=self.kernel_w,
           stride_h=self.stride_h, stride_w=self.stride_w,
           pad_h=self.pad_h, pad_w=self.pad_w)
        global_size = (self.operand.shape[0] * self.height_col * self.width_col, )
        return body, global_size, self.sources, self.sinks

    @classmethod
    def match(cls, node, symbol_table):
        if not isinstance(node, ast.Assign):
            return False
        node = node.value
        if isinstance(node, ast.Call):
            return isinstance(node.func, ast.Name) and \
                node.func.id == 'Im2Col'


register_operation(Im2Col)
