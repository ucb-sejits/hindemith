from _ctypes import sizeof
from numpy import zeros, array, zeros_like
from pycl import clGetDeviceIDs, clCreateContext, clCreateCommandQueue, cl_mem, buffer_from_ndarray, \
    clEnqueueNDRangeKernel, buffer_to_ndarray, clCreateProgramWithSource, clWaitForEvents
from ctree.c.nodes import SymbolRef, Constant
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from teller.core import hm
from teller.utils import unique_name
from teller.operations.dense_linear_algebra import Array

__author__ = 'leonardtruong'


class OclFunc(ConcreteSpecializedFunction):
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
        events = []
        in_buf, in_evt = buffer_from_ndarray(self.queue, im.data, blocking=False)
        events.append(in_evt)
        self.kernel.setarg(0, in_buf, sizeof(cl_mem))

        out_buf, out_evt = buffer_from_ndarray(self.queue, output, blocking=False)
        events.append(out_evt)
        self.kernel.setarg(1, out_buf, sizeof(cl_mem))
        clWaitForEvents(*events)
        evt = clEnqueueNDRangeKernel(self.queue, self.kernel, self.global_size)
        evt.wait()
        _, evt = buffer_to_ndarray(self.queue, out_buf, output)
        evt.wait()
        return Array(unique_name(), output)


class PyrDownLazy(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        return tuple((arg.dtype, arg.shape) for arg in args)

    def transform(self, tree, program_config):
        #TODO: Have to flip indices, figure out why
        arg_cfg = program_config[0]

        body = StringTemplate("""
            void __kernel pyr_down(__global const $type* input, __global $type* output) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                output[y * $len_x + x] = (
                    input[(y * 2) * $len_x + (x * 2)] +
                    input[(y * 2) * $len_x + (x * 2 + 1)] +
                    input[(y * 2 + 1) * $len_x + (x * 2 + 1)] +
                    input[(y * 2 + 1) * $len_x + (x * 2)]
                ) / 4.0;
            }
        """,
                              {
                                  'type': SymbolRef('float'),
                                  'len_x': Constant(arg_cfg[0][1][1])
                              }
        )
        fn = OclFunc()
        kernel = OclFile("kernel", [body])
        program = clCreateProgramWithSource(fn.context, kernel.codegen()).build()
        ptr = program['pyr_down']
        return fn.finalize(ptr, (arg_cfg[0][1][1] / 2, arg_cfg[0][1][0] / 2))


class PyrDown(object):
    def __new__(cls, pure_python=False):
        if pure_python:
            cls.__call__ = cls.pure_python
            return object.__new__(cls)
        else:
            return PyrDownLazy(None)

    def pure_python(self, im):
        im = im.data
        retval = zeros_like(im)
        for x in range(im.shape[0] / 2):
            for y in range(im.shape[1] / 2):
                retval[(x, y)] = (
                    im[(2 * x, 2 * y)] +
                    im[(2 * x + 1, 2 * y)] +
                    im[(2 * x + 1, 2 * y + 1)] +
                    im[(2 * x, 2 * y + 1)]
                )/4.0
        return Array(unique_name(), retval)

pyr_down = PyrDown()

@hm
def pyr_down_fn(im):
    return pyr_down(im)
