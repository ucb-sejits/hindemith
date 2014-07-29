__author__ = 'chick'

from _ctypes import sizeof
from numpy import zeros, zeros_like, array
from pycl import clGetDeviceIDs, clCreateContext, clCreateCommandQueue, cl_mem, buffer_from_ndarray, \
    clEnqueueNDRangeKernel, buffer_to_ndarray, clCreateProgramWithSource, clCreateBuffer, cl_int
from ctree.c.nodes import SymbolRef, Constant
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from hindemith.utils import unique_name, clamp
from hindemith.types.common import Array

from collections import namedtuple

CallArgs = namedtuple('CallArgs', ['base_shape', 'num_powers', 'border'])


class OclMatrixPowers(ConcreteSpecializedFunction):
    def __init__(self):
        self.device = clGetDeviceIDs()[-1]
        self.context = clCreateContext([self.device])
        self.queue = clCreateCommandQueue(self.context)

    def finalize(self, kernel, kernel2, global_size):
        self.kernel = kernel
        self.kernel2 = kernel2
        self.kernel.argtypes = (cl_mem, cl_mem)
        self.global_size = global_size
        return self

    def __call__(self, im, num_powers, border):
        out_shape = [num_powers] + list(im.shape)
        output = np.zeros(out_shape, dtype=np.float32)

        in_buf, evt = buffer_from_ndarray(self.queue, im.data, blocking=False)
        evt.wait()
        self.kernel.setarg(0, in_buf, sizeof(cl_mem))

        out_buf = clCreateBuffer(self.queue.context, output.nbytes)
        self.kernel.setarg(1, out_buf, sizeof(cl_mem))

        evt = clEnqueueNDRangeKernel(self.queue, self.kernel, self.global_size)
        evt.wait()

        self.kernel2.setarg(0, out_buf, sizeof(cl_mem))

        for power in range(num_powers):
            self.kernel2.setarg(1, power, sizeof(cl_int))
            evt = clEnqueueNDRangeKernel(self.queue, self.kernel2, self.global_size)
            evt.wait()

        _, evt = buffer_to_ndarray(self.queue, out_buf, output)
        evt.wait()
        return Array(unique_name(), output)


class MatrixPowersLazy(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        return CallArgs(args[0].shape, args[1], args[2])

    def transform(self, tree, program_config):
        call_args = program_config[0]

        base_size = call_args.base_shape[0] * call_args.base_shape[1]
        border = call_args.border

        body = StringTemplate("""
            void __kernel matrix_powers_copy_base_layer(__global const $type* input, __global $type* output) {
                int x = get_global_id(0);
                int y = get_global_id(1);

                output[y * $len_x + x] = input[y * $len_x + x];
            }
            void __kernel matrix_powers_compute_next_step(__global $type* matrix, const int power) {
                int x = get_global_id(0);
                int y = get_global_id(1);

                matrix[(power+1) * $base_size + y * $len_x + x] =
                    0.1f * matrix[
                        power * $base_size + clamp(y-1, $border, $len_y-$border-1) * $len_x +  clamp(x, $border, $len_x-$border-1)
                    ] +
                    0.1f * matrix[
                        power * $base_size + clamp(y+1, $border, $len_y-$border-1) * $len_x +  clamp(x, $border, $len_x-$border-1)
                    ] +
                    0.4f * matrix[
                        power * $base_size + clamp(y, $border, $len_y-$border-1) * $len_x +  clamp(x-1, $border, $len_x-$border-1)
                    ] +
                    0.4f * matrix[
                        power * $base_size + clamp(y, $border, $len_y-$border-1) * $len_x +  clamp(x+1, $border, $len_x-$border-1)
                    ] +
                    1.0f * matrix[
                        power * $base_size + clamp(y, $border, $len_y-$border-1) * $len_x +  clamp(x, $border, $len_x-$border-1)
                    ];
            }
        """, {
            'type': SymbolRef('float'),
            'len_x': Constant(call_args.base_shape[1]),
            'len_y': Constant(call_args.base_shape[0]),
            'base_size': Constant(base_size),
            'border': Constant(border),
        })

        fn = OclMatrixPowers()
        kernel = OclFile("kernel", [body])
        print(kernel.codegen())
        program = clCreateProgramWithSource(fn.context, kernel.codegen()).build()
        ptr = program['matrix_powers_copy_base_layer']
        ptr2 = program['matrix_powers_compute_next_step']
        return fn.finalize(ptr, ptr2, (call_args.base_shape[1], call_args.base_shape[0]))


class MatrixPowers(object):
    def __new__(cls, pure_python=False):
        if pure_python:
            cls.__call__ = cls.pure_python
            return object.__new__(cls)
        else:
            return MatrixPowersLazy(None)

    def pure_python(self, source, destination, border):
        depth, height, width = destination.shape
        wb = width - border - 1
        hb = height - border - 1

        for out_x in range(source.shape[0]):
            for out_y in range(source.shape[1]):
                destination[0, out_x, out_y] = source[
                    clamp(out_x, 0, source.shape[1]-1),
                    clamp(out_y, 0, source.shape[0]-1)
                ]

        for step in range(depth-1):
            for out_x in range(width):
                for out_y in range(height):
                    destination[step + 1, out_x, out_y] = \
                        0.4 * destination[step, clamp(out_x-1, border, wb), clamp(out_y, border, hb)] + \
                        0.1 * destination[step, clamp(out_x, border, wb), clamp(out_y-1, border, hb)] + \
                        0.1 * destination[step, clamp(out_x, border, wb), clamp(out_y+1, border, hb)] + \
                        0.4 * destination[step, clamp(out_x+1, border, wb), clamp(out_y, border, hb)] + \
                        1.0 * destination[step, clamp(out_x, border, wb), clamp(out_y-1, border, hb)]


if __name__ == '__main__':
    import numpy as np

    x_size = 5
    a = np.ones([x_size, 5], dtype=np.float32)
    mp = MatrixPowers()
    c = mp(a, 10, 0)
    print c.data

    # mp = MatrixPowers(pure_python=True)
    # c = mp(a, b)
    # print b

