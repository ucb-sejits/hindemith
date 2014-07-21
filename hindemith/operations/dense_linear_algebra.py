from ctypes import c_int, c_float
from ctree.ocl.macros import get_global_id
from docutils.parsers.rst.directives import body

__author__ = 'leonardtruong'

from _ctypes import sizeof, POINTER
from numpy import zeros_like
from numpy.ctypeslib import ndpointer
from pycl import clGetDeviceIDs, clCreateContext, clCreateCommandQueue, cl_mem, buffer_from_ndarray, \
    clEnqueueNDRangeKernel, buffer_to_ndarray, clCreateProgramWithSource, clWaitForEvents
from ctree.c.nodes import SymbolRef, Constant, FunctionDecl, Assign, ArrayRef, Add, Sub, Mul, Div
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from hindemith.types.common import HMType
from hindemith.utils import unique_name, UnsupportedBackendError, unique_kernel_name


class Scalar(HMType):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __mul__(self, other):
        if isinstance(other, Array):
            return Array(unique_name(), self.value * other.data)
        print(type(other))
        raise NotImplementedError()


class Float32(Scalar):
    pass


class Int(Scalar):
    pass


class Array(HMType):
    def __new__(cls, name, data):
        class ArrayInstance(cls):
            def __new__(cls, *args, **kwargs):
                return object.__new__(cls)

            def __init__(self, name, data):
                self.name = name
                self.data = data
                self.shape = data.shape
                self.dtype = data.dtype
        ArrayInstance.__add__ = ArrayAdd(name, data)
        ArrayInstance.__sub__ = ArraySub(name, data)
        ArrayInstance.__mul__ = ArrayMul(name, data)
        ArrayInstance.__div__ = ArrayDiv(name, data)
        return ArrayInstance(name, data)


class ArrayOpConcrete(ConcreteSpecializedFunction):
    def __init__(self, array, output_name):
        self.device = clGetDeviceIDs()[-1]
        self.context = clCreateContext([self.device])
        self.queue = clCreateCommandQueue(self.context)
        self.array = array
        self.output_name = output_name

    def finalize(self, kernel, global_size):
        self.kernel = kernel
        self.kernel.argtypes = (cl_mem, cl_mem, cl_mem)
        self.global_size = global_size
        return self

    def __call__(self, input2):
        output = zeros_like(self.array)
        events = []
        in_buf1, in_evt = buffer_from_ndarray(self.queue, self.array, blocking=False)
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
        return Array(self.output_name, output)


class ArrayOpLazy(LazySpecializedFunction):
    def __init__(self, tree, name, array=None):
        super(ArrayOpLazy, self).__init__(tree)
        self.array = array
        self.array_name = name

    def args_to_subconfig(self, args):
        def process_arg(arg):
            if isinstance(arg, Array):
                return arg.name, ndpointer(arg.dtype, arg.data.ndim, arg.shape), arg.shape
            elif isinstance(arg, Scalar):
                return arg.name, type(arg.value)
        return tuple(map(process_arg, args))

    def transform(self, tree, program_config):
        #TODO: Have to flip indices, figure out why
        arg_cfg, tune_cfg = program_config
        output_name = unique_name()
        params = [
            SymbolRef(self.array_name, POINTER(c_float)(), _global=True, _const=True),
            SymbolRef(arg_cfg[0][0], POINTER(c_float)(), _global=True, _const=True),
            SymbolRef(output_name, POINTER(c_float)(), _global=True)
        ]
        defn = []
        defn.extend([
            Assign(SymbolRef('element_id%d' % d, c_int()), get_global_id(d))
            for d in range(len(arg_cfg[0][2]))
        ])
        index = StringTemplate('element_id1 * $len_x + element_id0', {'len_x': Constant(
            arg_cfg[0][2][1])})
        defn.append(
            Assign(
                ArrayRef(SymbolRef(params[-1].name), index),
                tree(
                    ArrayRef(SymbolRef(params[0].name), index),
                    ArrayRef(SymbolRef(params[1].name), index),
                )
            )
        )
        entry_point = unique_kernel_name()
        tree = FunctionDecl(None, entry_point, params, defn)
        tree.set_kernel()
        fn = ArrayOpConcrete(self.array, output_name)
        kernel = OclFile("kernel", [tree])
        program = clCreateProgramWithSource(fn.context, kernel.codegen()).build()
        ptr = program[entry_point]
        return fn.finalize(ptr, (arg_cfg[0][2][1], arg_cfg[0][2][0]))


class ArrayOp(object):
    def __new__(cls, name, array, backend):
        if backend == 'python':
            cls.__call__ = cls.pure_python
            return super(ArrayOp, cls).__new__(cls, name, array, backend)
        raise UnsupportedBackendError(
            "Teller found an unsupported backend: {0}".format(backend)
        )

    def __init__(self, name, array, backend):
        self.array = array
        self.array_name = name

    def pure_python(self, input2):
        raise NotImplementedError()


class ArrayAdd(ArrayOp):
    def __new__(cls, name, array, backend='ocl'):
        if backend == 'ocl':
            return ArrayOpLazy(Add, name, array)
        return super(ArrayAdd, cls).__new__(cls, name, array, backend)

    def pure_python(self, input2):
        return Array(unique_name(), self.array + input2.data)


class ArraySub(ArrayOp):
    def __new__(cls, name, array, backend='ocl'):
        if backend == 'ocl':
            return ArrayOpLazy(Sub, name, array)
        return super(ArraySub, cls).__new__(cls, name, array, backend)

    def pure_python(self, input2):
        return Array(unique_name(), self.array - input2.data)


class ArrayMul(ArrayOp):
    def __new__(cls, name, array, backend='ocl'):
        if backend == 'ocl':
            return ArrayOpLazy(Mul, name, array)
        return super(ArrayMul, cls).__new__(cls, name, array, backend)

    def pure_python(self, input2):
        return Array(unique_name(), self.array * input2.data)


class ArrayDiv(ArrayOp):
    def __new__(cls, name, array, backend='ocl'):
        if backend == 'ocl':
            return ArrayOpLazy(Div, name, array)
        return super(ArrayDiv, cls).__new__(cls, name, array, backend)

    def pure_python(self, input2):
        return Array(unique_name(), self.array / input2.data)

