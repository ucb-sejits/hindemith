from ctypes import c_float, c_int
from _ctypes import sizeof, POINTER
from ctree.ocl.macros import get_global_id
from numpy import zeros_like
from pycl import clGetDeviceIDs, clCreateContext, clCreateCommandQueue, \
    buffer_from_ndarray, clEnqueueNDRangeKernel, buffer_to_ndarray, cl_mem, \
    clCreateProgramWithSource, clCreateBuffer
from ctree.c.nodes import SymbolRef, Assign, FunctionDecl
from ctree.ocl.nodes import OclFile
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from hindemith.utils import unique_name, clamp, unique_kernel_name
from hindemith.types.common import Array
from ctree.transformations import PyBasicConversions
import ast
__author__ = 'leonardtruong'


class OclFunc2(ConcreteSpecializedFunction):
    def __init__(self):
        self.device = clGetDeviceIDs()[-1]
        self.context = clCreateContext([self.device])
        self.queue = clCreateCommandQueue(self.context)

    def finalize(self, kernel, global_size):
        self.kernel = kernel
        self.kernel.argtypes = (cl_mem, cl_mem)
        self.global_size = global_size
        return self

    def __call__(self, im):
        output = zeros_like(im.data)
        in_buf, evt = buffer_from_ndarray(self.queue, im.data, blocking=False)
        evt.wait()
        self.kernel.setarg(0, in_buf, sizeof(cl_mem))

        out_buf = clCreateBuffer(self.context, output.nbytes)
        self.kernel.setarg(1, out_buf, sizeof(cl_mem))
        evt = clEnqueueNDRangeKernel(self.queue, self.kernel, self.global_size)
        evt.wait()
        _, evt = buffer_to_ndarray(self.queue, out_buf, output)
        evt.wait()
        del in_buf
        del out_buf
        return Array(unique_name(), output)

    def __del__(self):
        del self.context
        del self.queue


class PyrUpLazy(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        return tuple((arg.dtype, arg.shape) for arg in args)

    def transform(self, tree, program_config):
        arg_cfg = program_config[0]
        self.entry_point = unique_kernel_name()
        ctypeObject = c_float()
        ctype = c_float
        len_x = arg_cfg[0][1][0]
        len_y = arg_cfg[0][1][1]
        output = unique_name()
        params = [
            SymbolRef("input", POINTER(ctype)(), _global=True, _const=True),
            SymbolRef(output, POINTER(ctype)(), _global=True)
        ]
        defn = []
        defn.extend([
            Assign(SymbolRef('x', c_int()), get_global_id(0)),
            Assign(SymbolRef('y', c_int()), get_global_id(1)),
            Assign(SymbolRef('temp', ctypeObject), 0),
        ])
        body = \
            """
temp = .5 * input[clamp(x/2, 0, (len_x / 2) - 1) * len_y +
                  clamp(y/2, 0, (len_y / 2) - 1)]
if (x & 0x1):
    temp += .25 * input[clamp(x/2 + 1, 0, (len_x / 2) - 1) * len_y +
                        clamp(y/2, 0, (len_y /  2) - 1)]
else:
    temp += .25 * input[clamp(x/2 - 1, 0, (len_x / 2) - 1) * len_y +
                        clamp(y/2, 0, (len_y / 2) - 1)]
if (y & 0x1):
    temp += .25 * input[clamp(x/2, 0, (len_x / 2) - 1) * len_y +
                        clamp(y/2 + 1, 0, (len_y / 2) - 1)]
else:
    temp += .25 * input[clamp(x/2, 0, (len_x / 2) - 1) *len_y +
                        clamp(y/2 - 1, 0, (len_y / 2) - 1)]
output[x * len_y + y] = temp
"""
        body = ast.parse(body).body
        name_dict = {
            'output': output
        }
        const_dict = {
            'len_x': len_x,
            'len_y': len_y,
        }
        transformation = PyBasicConversions(name_dict, const_dict)
        defn.extend(body)
        tree = FunctionDecl(None, self.entry_point, params, defn)
        tree.set_kernel()
        kernel = OclFile("kernel", [tree])
        kernel = transformation.visit(kernel)
        return kernel

    def finalize(self, tree, program_config):
        arg_cfg, tune_cfg = program_config
        len_x = arg_cfg[0][1][0]
        len_y = arg_cfg[0][1][1]
        fn = OclFunc2()
        program = clCreateProgramWithSource(fn.context, tree.codegen()).build()
        ptr = program[self.entry_point]
        return fn.finalize(ptr, (len_x, len_y))


class PyrUp(object):
    def __new__(cls, pure_python=False):
        if pure_python:
            cls.__call__ = cls.pure_python
            return object.__new__(cls)
        else:
            return PyrUpLazy(None)

    def pure_python(self, im):
        im = im.data
        output = zeros_like(im)
        for x in range(im.shape[0]):
            for y in range(im.shape[1]):
                output[(x, y)] = .5 * im[(
                    clamp(int(x/2), 0, im.shape[0] / 2 - 1),
                    clamp(int((y/2)), 0, im.shape[1] / 2 - 1)
                )]
                if x & 0x1:  # if x is odd
                    output[(x, y)] += .25 * im[(
                        clamp(int(x/2 + 1), 0, im.shape[0] / 2 - 1),
                        clamp(int((y/2)), 0, im.shape[1] / 2 - 1)
                    )]
                else:
                    output[(x, y)] += .25 * im[(
                        clamp(int(x/2 - 1), 0, im.shape[0] / 2 - 1),
                        clamp(int((y/2)), 0, im.shape[1] / 2 - 1)
                    )]
                if y & 0x1:  # if y is odd
                    output[(x, y)] += .25 * im[(
                        clamp(int(x/2), 0, im.shape[0] / 2 - 1),
                        clamp(int((y/2 + 1)), 0, im.shape[1] / 2 - 1)
                    )]
                else:
                    output[(x, y)] += .25 * im[(
                        clamp(int(x/2), 0, im.shape[0] / 2 - 1),
                        clamp(int((y/2 - 1)), 0, im.shape[1] / 2 - 1)
                    )]
        return Array(unique_name(), output)

pyr_up = PyrUp()
