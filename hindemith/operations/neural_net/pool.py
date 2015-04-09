from hindemith.operations.core import ElementLevel, register_operation
import numpy as np
import ast


class PoolForward(ElementLevel):
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
                node.func.id == 'PoolForward'


register_operation(PoolForward)


class PoolBackward(ElementLevel):
    def __init__(self, statement, symbol_table):
        self.symbol_table = symbol_table
        self.statement = statement
        self.top_diff_name = statement.value.args[0].id
        self.top_diff = self.symbol_table[self.top_diff_name]
        self.mask_name = statement.value.args[1].id
        self.sources = [self.top_diff_name, self.mask_name]
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

        self.bottom_diff_name = statement.targets[0].id
        self.bottom_diff = symbol_table[self.bottom_diff_name]
        height, width = self.bottom_diff.shape[2:]
        self.pooled_height = ((height + 2 * self.pad_h - self.kernel_h) //
                              self.stride_h) + 1
        self.pooled_width = ((width + 2 * self.pad_w - self.kernel_w) //
                             self.stride_w) + 1
        self.sinks = [self.bottom_diff_name]

    def compile(self):
        body = """
    int index = get_global_id(0);
    int w = index % {width};
    int h = (index / {width}) % {height};
    int c = (index / {width} / {height}) % {channels};
    int n = index / {width} / {height} / {channels};
    int phstart =
        (h + {pad_h} < {kernel_h}) ? 0 : (h + {pad_h} - {kernel_h}) / {stride_h} + 1;
    int phend = min((h + {pad_h}) / {stride_h} + 1, {pooled_height});
    int pwstart =
        (w + {pad_w} < {kernel_w}) ? 0 : (w + {pad_w} - {kernel_w}) / {stride_w} + 1;
    int pwend = min((w + {pad_w}) / {stride_w} + 1, {pooled_width});
    float gradient = 0;
    int offset = (n * {channels} + c) * {pooled_height} * {pooled_width};
    {top_diff} += offset;
    mask += offset;
    for (int ph = phstart; ph < phend; ++ph) {{
      for (int pw = pwstart; pw < pwend; ++pw) {{
        if (mask[ph * {pooled_width} + pw] == h * {width} + w) {{
          gradient += {top_diff}[ph * {pooled_width} + pw];
        }}
      }}
    }}
    {bottom_diff}[index] = gradient;
""".format(bottom_diff=self.bottom_diff_name, mask=self.mask_name,
           top_diff=self.top_diff_name, pooled_width=self.pooled_width,
           pooled_height=self.pooled_height, channels=self.bottom_diff.shape[1],
           height=self.bottom_diff.shape[2], width=self.bottom_diff.shape[3],
           kernel_h=self.kernel_h, kernel_w=self.kernel_w,
           stride_h=self.stride_h, stride_w=self.stride_w,
           pad_h=self.pad_h, pad_w=self.pad_w)
        global_size = (np.prod(self.bottom_diff.shape), )
        return body, global_size, self.sources, self.sinks

    @classmethod
    def match(cls, node, symbol_table):
        if not isinstance(node, ast.Assign):
            return False
        node = node.value
        if isinstance(node, ast.Call):
            return isinstance(node.func, ast.Name) and \
                node.func.id == 'PoolBackward'

register_operation(PoolBackward)
