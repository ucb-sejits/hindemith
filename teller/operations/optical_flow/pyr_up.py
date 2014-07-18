from _ctypes import sizeof
from numpy import zeros, zeros_like, array
from pycl import clGetDeviceIDs, clCreateContext, clCreateCommandQueue, cl_mem, buffer_from_ndarray, \
    clEnqueueNDRangeKernel, buffer_to_ndarray, clCreateProgramWithSource
from ctree.c.nodes import SymbolRef, Constant
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from teller.utils import unique_name, clamp
from teller.operations.dense_linear_algebra.array_op import Array

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

        out_buf, evt = buffer_from_ndarray(self.queue, output, blocking=False)
        evt.wait()
        self.kernel.setarg(1, out_buf, sizeof(cl_mem))
        evt = clEnqueueNDRangeKernel(self.queue, self.kernel, self.global_size)
        evt.wait()
        _, evt = buffer_to_ndarray(self.queue, out_buf, output)
        evt.wait()
        return Array(unique_name(), output)


class PyrUpLazy(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        return tuple((arg.dtype, arg.shape) for arg in args)

    def transform(self, tree, program_config):
        arg_cfg = program_config[0]

        body = StringTemplate("""
            void __kernel pyr_up(__global const $type* input, __global $type* output) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                output[y * $len_x + x] = .5 * input[
                        clamp(x/2, 0, ($len_x / 2) - 1) +
                        clamp(y/2, 0, ($len_y / 2) - 1) * $len_x];
                if (x & 0x1) {
                    output[y * $len_x + x] += .25 * input[
                        clamp(x/2 + 1, 0, ($len_x / 2) - 1) +
                        clamp(y/2, 0, ($len_y /  2) - 1) * $len_x];
                } else {
                    output[y * $len_x + x] += .25 * input[
                        clamp(x/2 - 1, 0, ($len_x / 2) - 1) +
                        clamp(y/2, 0, ($len_y / 2) - 1) * $len_x];
                }

                if (y & 0x1) {
                    output[y * $len_x + x] += .25 * input[
                        clamp(x/2, 0, ($len_x / 2) - 1) +
                        clamp(y/2 + 1, 0, ($len_y / 2) - 1) * $len_x];
                } else {
                    output[y * $len_x + x] += .25 * input[
                        clamp(x/2, 0, ($len_x / 2) - 1) +
                        clamp(y/2 - 1, 0, ($len_y / 2) - 1) * $len_x];
                }
            }
        """,
                              {
                                  'type': SymbolRef('float'),
                                  'len_x': Constant(arg_cfg[0][1][1]),
                                  'len_y': Constant(arg_cfg[0][1][0])
                              }
        )
        fn = OclFunc2()
        kernel = OclFile("kernel", [body])
        program = clCreateProgramWithSource(fn.context, kernel.codegen()).build()
        ptr = program['pyr_up']
        return fn.finalize(ptr, (arg_cfg[0][1][1], arg_cfg[0][1][0]))


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
                output[(x, y)] = .5 * im[(clamp(int(x/2),0, im.shape[0] / 2 - 1),
                                          clamp(int((y/2)), 0, im.shape[1] / 2 - 1))]
                if x & 0x1:  # if x is odd
                    output[(x, y)] += .25 * im[(clamp(int(x/2 + 1),0, im.shape[0] / 2 - 1),
                                                clamp(int((y/2)), 0, im.shape[1] / 2 - 1))]
                else:
                    output[(x, y)] += .25 * im[(clamp(int(x/2 - 1),0, im.shape[0] / 2 - 1),
                                                clamp(int((y/2)), 0, im.shape[1] / 2 - 1))]
                if y & 0x1:  # if y is odd
                    output[(x, y)] += .25 * im[(clamp(int(x/2),0, im.shape[0] / 2 - 1),
                                                clamp(int((y/2 + 1)), 0, im.shape[1] / 2 - 1))]
                else:
                    output[(x, y)] += .25 * im[(clamp(int(x/2),0, im.shape[0] / 2 - 1),
                                                clamp(int((y/2 - 1)), 0, im.shape[1] / 2 - 1))]
        return Array(unique_name(), output)

pyr_up = PyrUp()

