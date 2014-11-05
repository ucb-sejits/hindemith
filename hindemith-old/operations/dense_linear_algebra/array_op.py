from _ctypes import sizeof, POINTER
from ctypes import c_float, c_int
from ctree.c.nodes import SymbolRef, Assign, Constant, ArrayRef, FunctionDecl, \
    Add, Sub, Mul, Div
from ctree.jit import ConcreteSpecializedFunction, LazySpecializedFunction
from ctree.ocl.macros import get_global_id
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
from numpy import zeros_like
from numpy.ctypeslib import ndpointer
import hindemith.types as types
from hindemith.utils import unique_name, unique_kernel_name, \
    UnsupportedBackendError
from pycl import clGetDeviceIDs, clCreateContext, clCreateCommandQueue, \
    cl_mem, buffer_from_ndarray, clWaitForEvents, buffer_to_ndarray, \
    clEnqueueNDRangeKernel, clCreateProgramWithSource

__author__ = 'leonardtruong'


class ArrayOpConcrete(ConcreteSpecializedFunction):
    def __init__(self, array, output):
        self.device = clGetDeviceIDs()[-1]
        self.context = clCreateContext([self.device])
        self.queue = clCreateCommandQueue(self.context)
        self.array = array
        self.output = output

    def finalize(self, kernel, global_size):
        self.kernel = kernel
        self.global_size = global_size
        return self

    def process_inputs(self, *args):
        events = []
        processed = []
        self.kernel.argtypes = tuple(cl_mem for _ in args)
        for index, arg in enumerate(args):
            if isinstance(arg, types.common.Array):
                arg = arg.data
            buf, evt = buffer_from_ndarray(self.queue, arg, blocking=False)
            processed.append(buf)
            events.append(evt)
            self.kernel.setarg(index, buf, sizeof(cl_mem))
        clWaitForEvents(*events)
        return processed

    def process_output(self, out_buf, output):
        _, evt = buffer_to_ndarray(self.queue, out_buf, output.data)
        evt.wait()
        return output

    def __call__(self, *args):
        args = (self.array,) + args + (self.output,)
        bufs = self.process_inputs(*args)

        evt = clEnqueueNDRangeKernel(self.queue, self.kernel, self.global_size)
        evt.wait()
        result = self.process_output(bufs[-1], args[-1])
        for buf in bufs:
            del buf
        return result

    def __del__(self):
        del self.context
        del self.queue


class ArrayOpLazy(LazySpecializedFunction):
    def __init__(self, tree, name, array=None):
        super(ArrayOpLazy, self).__init__(tree)
        self.array = array
        self.array_name = name
        self.fusable_nodes = []

    def args_to_subconfig(self, args):
        def process_arg(arg):
            if isinstance(arg, types.common.Array):
                return (
                    arg.name,
                    ndpointer(arg.dtype, arg.data.ndim, arg.shape),
                    arg.shape
                )
            elif isinstance(arg, types.common.Scalar):
                return arg.name, type(arg.value)
        return tuple(map(process_arg, args))

    def transform(self, tree, program_config):
        # TODO: Have to flip indices, figure out why
        arg_cfg, tune_cfg = program_config
        output_name = unique_name()
        params = [
            SymbolRef(self.array_name, POINTER(c_float)(), _global=True,
                      _const=True),
            SymbolRef(arg_cfg[0][0], POINTER(c_float)(), _global=True,
                      _const=True),
            SymbolRef(output_name, POINTER(c_float)(), _global=True)
        ]
        defn = []
        defn.extend([
            Assign(SymbolRef('element_id%d' % d, c_int()), get_global_id(d))
            for d in range(len(arg_cfg[0][2]))
        ])
        index = StringTemplate('element_id1 * $len_x + element_id0',
                               {'len_x': Constant(arg_cfg[0][2][1])})
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
        fn = ArrayOpConcrete(self.array, self.generate_output(output_name))
        kernel = OclFile("kernel", [tree])
        program = clCreateProgramWithSource(
            fn.context, kernel.codegen()
        ).build()
        ptr = program[entry_point]
        return fn.finalize(ptr, (arg_cfg[0][2][1], arg_cfg[0][2][0]))

    def get_semantic_tree(self, arg, output_name):
        params = [
            SymbolRef(self.array_name, POINTER(c_float)(), _global=True,
                      _const=True),
            SymbolRef(arg.name, POINTER(c_float)(), _global=True, _const=True),
            SymbolRef(output_name, POINTER(c_float)(), _global=True)
        ]
        defn = []
        defn.extend([
            Assign(SymbolRef('element_id%d' % d, c_int()), get_global_id(d))
            for d in range(len(arg.data.shape))
        ])
        index = StringTemplate('element_id1 * $len_x + element_id0',
                               {'len_x': Constant(arg.data.shape[1])})
        defn.append(
            Assign(
                ArrayRef(SymbolRef(params[-1].name), index),
                self.original_tree(
                    ArrayRef(SymbolRef(params[0].name), index),
                    ArrayRef(SymbolRef(params[1].name), index),
                    )
            )
        )
        entry_point = unique_kernel_name()
        tree = FunctionDecl(None, entry_point, params, defn)
        tree.set_kernel()
        kernel = OclFile("kernel", [tree])
        return kernel

    def get_fusable_nodes(self, arg, output_name):
        return [self.get_semantic_tree(arg, output_name)]

    def generate_output(self, name):
        self.output = types.common.Array(name, zeros_like(self.array))
        return self.output


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
        return types.common.Array(unique_name(), self.array + input2.data)


class ArraySub(ArrayOp):
    def __new__(cls, name, array, backend='ocl'):
        if backend == 'ocl':
            return ArrayOpLazy(Sub, name, array)
        return super(ArraySub, cls).__new__(cls, name, array, backend)

    def pure_python(self, input2):
        return types.common.Array(unique_name(), self.array - input2.data)


class ArrayMul(ArrayOp):
    def __new__(cls, name, array, backend='ocl'):
        if backend == 'ocl':
            return ArrayOpLazy(Mul, name, array)
        return super(ArrayMul, cls).__new__(cls, name, array, backend)

    def pure_python(self, input2):
        return types.common.Array(unique_name(), self.array * input2.data)


class ArrayDiv(ArrayOp):
    def __new__(cls, name, array, backend='ocl'):
        if backend == 'ocl':
            return ArrayOpLazy(Div, name, array)
        return super(ArrayDiv, cls).__new__(cls, name, array, backend)

    def pure_python(self, input2):
        return types.common.Array(unique_name(), self.array / input2.data)


def square(input):
    return types.common.Array(unique_name(), input.data * input.data)
