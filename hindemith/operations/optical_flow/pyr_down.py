from _ctypes import sizeof, POINTER
from ctypes import c_float, c_int
from ctree.ocl import get_context_from_device
from ctree.ocl.macros import get_global_id
from numpy import zeros_like
from pycl import clGetDeviceIDs, clCreateCommandQueue, \
    buffer_from_ndarray, clEnqueueNDRangeKernel, buffer_to_ndarray, \
    clCreateProgramWithSource, clWaitForEvents, cl_mem
from ctree.c.nodes import SymbolRef, Constant, Assign, ArrayRef, Add, Div, \
    FunctionDecl
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from hindemith.utils import unique_name, unique_kernel_name
from hindemith.types.common import Array

__author__ = 'leonardtruong'


class OclFunc(ConcreteSpecializedFunction):
    def __init__(self):
        self.device = clGetDeviceIDs()[-1]
        self.context = get_context_from_device(self.device)
        self.queue = clCreateCommandQueue(self.context)

    def finalize(self, kernel, global_size, output_name):
        self.kernel = kernel
        self.kernel.argtypes = (cl_mem, cl_mem)
        self.global_size = global_size
        self.output_name = output_name
        return self

    def __call__(self, im):
        output = zeros_like(im.data)
        events = []
        in_buf, in_evt = buffer_from_ndarray(
            self.queue, im.data, blocking=False)
        events.append(in_evt)
        self.kernel.setarg(0, in_buf, sizeof(cl_mem))

        out_buf, out_evt = buffer_from_ndarray(
            self.queue, output, blocking=False)
        events.append(out_evt)
        self.kernel.setarg(1, out_buf, sizeof(cl_mem))
        clWaitForEvents(*events)
        evt = clEnqueueNDRangeKernel(self.queue, self.kernel, self.global_size)
        evt.wait()
        _, evt = buffer_to_ndarray(self.queue, out_buf, output)
        evt.wait()
        return Array(self.output_name, output)

    def __del__(self):
        del self.context
        del self.queue


class PyrDownLazy(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        return tuple((arg.name, arg.dtype, arg.shape) for arg in args)

    def transform(self, tree, program_config):
        # TODO: Have to flip indices, figure out why
        arg_cfg = program_config[0]

        input_name = arg_cfg[0][0]
        self.output_name = unique_name()
        params = [
            SymbolRef(
                input_name, POINTER(c_float)(), _global=True, _const=True),
            SymbolRef(self.output_name, POINTER(c_float)(), _global=True)
        ]
        defn = []
        defn.extend([
            Assign(SymbolRef('element_id%d' % d, c_int()), get_global_id(d))
            for d in range(len(arg_cfg[0][2]))
        ])
        out_index = StringTemplate('element_id1 * $len_x + element_id0',
                                   {'len_x': Constant(arg_cfg[0][2][1])})
        defn.append(Assign(
            ArrayRef(SymbolRef(self.output_name), out_index),
            Div(
                Add(
                    ArrayRef(
                        SymbolRef(input_name),
                        StringTemplate(
                            '(element_id1 * 2) * $len_x + (element_id0 * 2)',
                            {'len_x': Constant(arg_cfg[0][2][1])})
                    ),
                    Add(
                        ArrayRef(
                            SymbolRef(input_name),
                            StringTemplate(
                                '(element_id1 * 2) * $len_x + \
                                 (element_id0 * 2 + 1)',
                                {'len_x': Constant(arg_cfg[0][2][1])})
                        ),
                        Add(
                            ArrayRef(
                                SymbolRef(input_name),
                                StringTemplate(
                                    '(element_id1 * 2 + 1) * $len_x + \
                                     (element_id0 * 2 + 1)',
                                    {'len_x': Constant(arg_cfg[0][2][1])})
                            ),
                            Add(
                                ArrayRef(
                                    SymbolRef(input_name),
                                    StringTemplate(
                                        '(element_id1 * 2 + 1) * $len_x + \
                                         (element_id0 * 2)',
                                        {'len_x': Constant(arg_cfg[0][2][1])})
                                ),
                            )
                        )
                    )
                ),
                Constant(4.0)
            )
        ))

        self.entry_point = unique_kernel_name()
        tree = FunctionDecl(None, self.entry_point, params, defn)
        tree.set_kernel()
        kernel = OclFile("kernel", [tree])
        return kernel

    def finalize(self, tree, program_config):
        arg_cfg, tune_cfg = program_config
        fn = OclFunc()
        program = clCreateProgramWithSource(
            fn.context, tree.codegen()).build()
        ptr = program[self.entry_point]
        return fn.finalize(
            ptr, (int(arg_cfg[0][2][1] / 2), int(arg_cfg[0][2][0] / 2)), self.output_name
        )


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
        for x in range(int(im.shape[0] / 2)):
            for y in range(int(im.shape[1] / 2)):
                retval[(x, y)] = (
                    im[(2 * x, 2 * y)] +
                    im[(2 * x + 1, 2 * y)] +
                    im[(2 * x + 1, 2 * y + 1)] +
                    im[(2 * x, 2 * y + 1)]
                )/4.0
        return Array(unique_name(), retval)

pyr_down = PyrDown()
pyr_down_fn = pyr_down
