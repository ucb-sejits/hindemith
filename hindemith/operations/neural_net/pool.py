from hindemith.operations.core import ElementLevel, register_operation
from hindemith.types import NDArray
import numpy as np
import ast


class Pool(ElementLevel):
    def __init__(self, statement, symbol_table):
        self.symbol_table = symbol_table
        self.statement = statement
        self.operand_name = statement.value.args[0].id
        self.operand = self.symbol_table[self.operand_name]
        self.mask_name = statement.value.args[1].id
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
        height, width = self.operand.shape[2:]
        self.pooled_height = ((height + 2 * self.pad_h - self.kernel_h) //
                              self.stride_h) + 1
        self.pooled_width = ((width + 2 * self.pad_w - self.kernel_w) //
                             self.stride_w) + 1

        self.target_name = statement.targets[0].id
        self.target = symbol_table[self.target_name]
        self.sinks = [self.target_name, self.mask_name]

    def compile(self):
        body = """
    int index = get_global_id(0);
    int pw = index % {pooled_width};
    int ph = (index / {pooled_width}) % {pooled_height};
    int c = (index / {pooled_width} / {pooled_height}) % {channels};
    int n = index / {pooled_width} / {pooled_height} / {channels};
    int hstart = ph * {stride_h} - {pad_h};
    int wstart = pw * {stride_w} - {pad_w};
    int hend = min(hstart + {kernel_h}, {height});
    int wend = min(wstart + {kernel_w}, {width});
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    float maxval = -FLT_MAX;
    int maxidx = -1;
    {bottom} += (n * {channels} + c) * {height} * {width};
    for (int h = hstart; h < hend; ++h) {{
      for (int w = wstart; w < wend; ++w) {{
        if ({bottom}[h * {width} + w] > maxval) {{
          maxidx = h * {width} + w;
          maxval = {bottom}[maxidx];
        }}
      }}
    }}
    {top}[index] = maxval;
    {mask}[index] = maxidx;
""".format(bottom=self.operand_name, mask=self.mask_name,
           top=self.target_name, pooled_width=self.pooled_width,
           pooled_height=self.pooled_height, channels=self.operand.shape[1],
           height=self.operand.shape[2], width=self.operand.shape[3],
           kernel_h=self.kernel_h, kernel_w=self.kernel_w,
           stride_h=self.stride_h, stride_w=self.stride_w,
           pad_h=self.pad_h, pad_w=self.pad_w)
        global_size = (np.prod(self.target.shape), )
        return body, global_size, self.sources, self.sinks

    @classmethod
    def match(cls, node, symbol_table):
        if not isinstance(node, ast.Assign):
            return False
        node = node.value
        if isinstance(node, ast.Call):
            return isinstance(node.func, ast.Name) and \
                node.func.id == 'Pool'


register_operation(Pool)
