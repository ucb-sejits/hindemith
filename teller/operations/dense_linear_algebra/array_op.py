from _ctypes import sizeof
from numpy import zeros, array, zeros_like
from pycl import clGetDeviceIDs, clCreateContext, clCreateCommandQueue, cl_mem, buffer_from_ndarray, \
    clEnqueueNDRangeKernel, buffer_to_ndarray, clCreateProgramWithSource, clWaitForEvents
from ctree.c.nodes import SymbolRef, Constant
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction

__author__ = 'leonardtruong'


class ArrayOpConcrete(ConcreteSpecializedFunction):
    def __init__(self):
        self.device = clGetDeviceIDs()[-1]
        self.context = clCreateContext([self.device])
        self.queue = clCreateCommandQueue(self.context)

    def finalize(self, kernel, global_size):
        self.kernel = kernel
        self.kernel.argtypes = (cl_mem, cl_mem, cl_mem)
        self.global_size = global_size
        return self

    def __call__(self, input1, input2):
        output = zeros_like(input1.data)
        events = []
        in_buf1, in_evt = buffer_from_ndarray(self.queue, input1.data, blocking=False)
        events.append(in_evt)
        self.kernel.setarg(0, in_buf1, sizeof(cl_mem))

        in_buf2, in_evt = buffer_from_ndarray(self.queue, input2.data, blocking=False)
        events.append(in_evt)
        self.kernel.setarg(1, in_buf2, sizeof(cl_mem))

        out_buf, out_evt = buffer_from_ndarray(self.queue, output, blocking=False)
        events.append(out_evt)
        self.kernel.setarg(2, out_buf, sizeof(cl_mem))
        clWaitForEvents(*events)
        evt = clEnqueueNDRangeKernel(self.queue, self.kernel, self.global_size)
        evt.wait()
        _, evt = buffer_to_ndarray(self.queue, out_buf, output)
        evt.wait()
        return output


class ArrayOpLazy(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        return tuple((arg.dtype, arg.shape) for arg in args)

    def transform(self, tree, program_config):
        #TODO: Have to flip indices, figure out why
        arg_cfg = program_config[0]

        body = StringTemplate("""
            void __kernel elementwise_op(__global const $type* input1,
                                   __global const $type* input2,
                                   __global $type* output) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                output[y * $len_x + x] = input1[y * $len_x + x] $op input2[y * $len_x + x];
            }
        """,
                              {
                                  'type': SymbolRef('float'),
                                  'len_x': Constant(arg_cfg[0][1][1]),
                                  'op': StringTemplate(tree)
                              }
        )
        fn = ArrayOpConcrete()
        kernel = OclFile("kernel", [body])
        program = clCreateProgramWithSource(fn.context, kernel.codegen()).build()
        ptr = program['elementwise_op']
        return fn.finalize(ptr, (arg_cfg[0][1][1], arg_cfg[0][1][0]))


class ArrayAdd(object):
    def __new__(cls, backend='ocl'):
        if backend == 'python':
            cls.__call__ = cls.pure_python
            return object.__new__(cls)
        # elif backend == 'c':
        #     return WarpImg2DLazyC(None)
        elif backend == 'ocl':
            return ArrayOpLazy('+')
        # TODO: Create HMException
        raise Exception("Unsupported backend: {0}".format(backend))

    def pure_python(self, input1, input2):
        return input1 + input2


class ArraySub(object):
    def __new__(cls, backend='ocl'):
        if backend == 'python':
            cls.__call__ = cls.pure_python
            return object.__new__(cls)
        # elif backend == 'c':
        #     return WarpImg2DLazyC(None)
        elif backend == 'ocl':
            return ArrayOpLazy('-')
        # TODO: Create HMException
        raise Exception("Unsupported backend: {0}".format(backend))

    def pure_python(self, input1, input2):
        return input1 - input2


class ArrayMul(object):
    def __new__(cls, backend='ocl'):
        if backend == 'python':
            cls.__call__ = cls.pure_python
            return object.__new__(cls)
        # elif backend == 'c':
        #     return WarpImg2DLazyC(None)
        elif backend == 'ocl':
            return ArrayOpLazy('*')
        # TODO: Create HMException
        raise Exception("Unsupported backend: {0}".format(backend))

    def pure_python(self, input1, input2):
        return input1 * input2


class ArrayDiv(object):
    def __new__(cls, backend='ocl'):
        if backend == 'python':
            cls.__call__ = cls.pure_python
            return object.__new__(cls)
        # elif backend == 'c':
        #     return WarpImg2DLazyC(None)
        elif backend == 'ocl':
            return ArrayOpLazy('/')
        # TODO: Create HMException
        raise Exception("Unsupported backend: {0}".format(backend))

    def pure_python(self, input1, input2):
        return input1 / input2

