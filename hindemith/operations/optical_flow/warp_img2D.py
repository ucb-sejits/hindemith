from _ctypes import sizeof
from ctypes import CFUNCTYPE, c_void_p
from ctree.ocl.macros import get_global_id
from numpy import zeros_like
import numpy as np
from ctree.nodes import Project
from hindemith.utils import unique_name, unique_kernel_name, \
    UnsupportedBackendError
from hindemith.types.common import Array
from pycl import clCreateCommandQueue, cl_mem, clGetDeviceIDs, \
    clCreateContext, buffer_from_ndarray, clEnqueueNDRangeKernel, \
    buffer_to_ndarray, clCreateProgramWithSource, clWaitForEvents
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction

from ctree.c.nodes import SymbolRef, Constant, CFile, FunctionDecl, FunctionCall, \
    Add, Sub, Mul, PostDec, PostInc, Assign, AddAssign, For, If, And, Lt, GtE, \
    Cast, ArrayRef, Return
from ctypes import c_float, c_int, POINTER

__author__ = 'leonardtruong'


class WarpImg2DConcreteOcl(ConcreteSpecializedFunction):
    def __init__(self):
        self.device = clGetDeviceIDs()[-1]
        self.context = clCreateContext([self.device])
        self.queue = clCreateCommandQueue(self.context)

    def finalize(self, kernel, global_size):
        self.kernel = kernel
        self.kernel.argtypes = (cl_mem, cl_mem, cl_mem, cl_mem)
        self.global_size = global_size
        return self

    def __call__(self, input, u, v):
        output = zeros_like(input.data)
        events = []
        in_buf, in_evt = buffer_from_ndarray(self.queue, input.data,
                                             blocking=False)
        events.append(in_evt)
        self.kernel.setarg(0, in_buf, sizeof(cl_mem))

        u_buf, u_evt = buffer_from_ndarray(self.queue, u.data, blocking=False)
        events.append(u_evt)
        self.kernel.setarg(1, u_buf, sizeof(cl_mem))

        v_buf, v_evt = buffer_from_ndarray(self.queue, v.data, blocking=False)
        events.append(v_evt)
        self.kernel.setarg(2, v_buf, sizeof(cl_mem))

        out_buf, out_evt = buffer_from_ndarray(self.queue, output,
                                               blocking=False)
        events.append(out_evt)
        self.kernel.setarg(3, out_buf, sizeof(cl_mem))
        clWaitForEvents(*events)
        evt = clEnqueueNDRangeKernel(self.queue, self.kernel, self.global_size)
        evt.wait()
        _, evt = buffer_to_ndarray(self.queue, out_buf, output)
        evt.wait()
        return Array(unique_name(), output)


class WarpImg2DLazyOcl(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        return tuple((arg.dtype, arg.shape) for arg in args)

    def transform(self, tree, program_config):
        arg_cfg = program_config[0]

        len_x = Constant(arg_cfg[0][1][0])
        len_y = Constant(arg_cfg[0][1][1])

        self.entry_point = unique_kernel_name()

        body = FunctionDecl(
            None,
            self.entry_point,
            [
                SymbolRef('input', POINTER(c_float)(), _global=True,
                          _const=True),
                SymbolRef('u', POINTER(c_float)(), _global=True, _const=True),
                SymbolRef('v', POINTER(c_float)(), _global=True, _const=True),
                SymbolRef('output', POINTER(c_float)(), _global=True)
            ],
            [
                Assign(SymbolRef('x', c_int()), get_global_id(0)),
                Assign(SymbolRef('y', c_int()), get_global_id(1)),
                Assign(
                    SymbolRef('my_x', c_int()),
                    Cast(c_int(),
                         ArrayRef(SymbolRef('u'),
                                  Add(SymbolRef('x'),
                                      Mul(SymbolRef('y'), len_x))))),
                Assign(
                    SymbolRef('my_y', c_int()),
                    Cast(c_int(),
                         ArrayRef(SymbolRef('v'),
                                  Add(SymbolRef('x'),
                                      Mul(SymbolRef('y'), len_x))))),
                Assign(
                    SymbolRef('xfrac', c_float()),
                    Sub(ArrayRef(SymbolRef('u'),
                                 Add(SymbolRef('x'),
                                     Mul(len_x, SymbolRef('y')))),
                        SymbolRef('my_x'))),
                Assign(
                    SymbolRef('yfrac', c_float()),
                    Sub(ArrayRef(SymbolRef('v'),
                                 Add(SymbolRef('x'),
                                     Mul(len_x, SymbolRef('y')))),
                        SymbolRef('my_y'))),
                If(Lt(
                    ArrayRef(SymbolRef('u'),
                             Add(SymbolRef('x'),
                                 Mul(len_x, SymbolRef('y')))),
                    Constant(0.0)),
                    [
                        PostDec('my_x'),
                        Assign(SymbolRef('xfrac'),
                               Add(Constant(1.0),
                                   SymbolRef('xfrac')))
                    ]),
                If(Lt(
                    ArrayRef(SymbolRef('v'),
                             Add(SymbolRef('x'),
                                 Mul(len_x, SymbolRef('y')))),
                    Constant(0.0)),
                    [
                        PostDec('my_y'),
                        Assign(SymbolRef('yfrac'),
                               Add(Constant(1.0),
                                   SymbolRef('yfrac')))
                    ]),
                Assign(SymbolRef('tmp', c_float()), Constant(0.0)),
                If(
                    And(
                        And(
                            GtE(Add(SymbolRef('x'),
                                    SymbolRef('my_x')), Constant(0)),
                            Lt(Add(SymbolRef('x'),
                                   Add(SymbolRef('my_x'), Constant(1))), len_x)
                        ),
                        And(
                            GtE(Add(SymbolRef('y'),
                                    SymbolRef('my_y')), Constant(0)),
                            Lt(Add(SymbolRef('y'),
                                   Add(SymbolRef('my_y'), Constant(1))), len_y)
                        )
                    ),
                    [
                        AddAssign(
                            SymbolRef('tmp'),
                            Mul(
                                Mul(
                                    ArrayRef(
                                        SymbolRef('input'),
                                        Add(
                                            Add(
                                                SymbolRef('x'),
                                                SymbolRef('my_x')),
                                            Mul(
                                                len_x,
                                                Add(SymbolRef('y'),
                                                    SymbolRef('my_y'))))),
                                    Sub(Constant(1.0), SymbolRef('xfrac'))),
                                Sub(Constant(1.0), SymbolRef('yfrac')))),
                        AddAssign(
                            SymbolRef('tmp'),
                            Mul(
                                Mul(
                                    ArrayRef(
                                        SymbolRef('input'),
                                        Add(
                                            Add(
                                                Add(SymbolRef('x'),
                                                    SymbolRef('my_x')),
                                                Constant(1)),
                                            Mul(
                                                len_x,
                                                Add(SymbolRef('y'),
                                                    SymbolRef('my_y'))))),
                                    SymbolRef('xfrac')),
                                Sub(Constant(1.0), SymbolRef('yfrac')))),
                        AddAssign(
                            SymbolRef('tmp'),
                            Mul(
                                Mul(
                                    ArrayRef(
                                        SymbolRef('input'),
                                        Add(
                                            Add(
                                                SymbolRef('x'),
                                                SymbolRef('my_x')),
                                            Mul(
                                                len_x,
                                                Add(Add(SymbolRef('y'),
                                                        SymbolRef('my_y')),
                                                    Constant(1))))),
                                    Sub(Constant(1.0), SymbolRef('xfrac'))),
                                SymbolRef('yfrac'))),
                        AddAssign(
                            SymbolRef('tmp'),
                            Mul(
                                Mul(ArrayRef(
                                    SymbolRef('input'),
                                    Add(
                                        Add(
                                            Add(SymbolRef('x'),
                                                SymbolRef('my_x')),
                                            Constant(1)),
                                        Mul(
                                            len_x,
                                            Add(Add(SymbolRef('y'),
                                                    SymbolRef('my_y')),
                                                Constant(1))))),
                                    SymbolRef('xfrac')),
                                SymbolRef('yfrac'))),
                        ],
                    Assign(
                        SymbolRef('tmp'),
                        ArrayRef(
                            SymbolRef('input'),
                            Add(
                                FunctionCall(
                                    SymbolRef('clamp'),
                                    [
                                        Add(SymbolRef('x'), SymbolRef('my_x')),
                                        Constant(0),
                                        Sub(len_x, Constant(1))
                                    ]
                                ),
                                Mul(
                                    len_x,
                                    FunctionCall(SymbolRef('clamp'), [
                                        Add(SymbolRef('y'), SymbolRef('my_y')),
                                        Constant(0),
                                        Sub(len_y, Constant(1))
                                    ]
                                    ),
                                    )
                            )
                        )
                    )

                ),
                Assign(
                    ArrayRef(SymbolRef('output'),
                             Add(SymbolRef('x'),
                                 Mul(len_x, SymbolRef('y')))),
                    SymbolRef('tmp')
                )
            ]
        )

        body.set_kernel()
        kernel = OclFile("kernel", [body])
        return kernel

    def finalize(self, tree, program_config):
        arg_cfg = program_config[0]
        fn = WarpImg2DConcreteOcl()
        program = clCreateProgramWithSource(fn.context,
                                            tree.codegen()).build()
        ptr = program[self.entry_point]
        return fn.finalize(ptr, arg_cfg[0][1])


class WarpImg2DConcreteC(ConcreteSpecializedFunction):
    def finalize(self, entry_name, tree, entry_type):
        self._c_function = self._compile(entry_name, tree, entry_type)
        return self

    def __call__(self, input, u, v):
        output = zeros_like(input.data)
        self._c_function(input.data, u.data, v.data, output)
        return Array(unique_name(), output)


class WarpImg2DLazyC(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        return tuple((arg.data.dtype, arg.data.shape, arg.data.ndim)
                     for arg in args)

    def transform(self, tree, program_config):
        arg_cfg = program_config[0]

        len_x = Constant(arg_cfg[0][1][0])
        len_y = Constant(arg_cfg[0][1][1])

        body = [
            StringTemplate('#include <math.h>'),
            FunctionDecl(
                c_int(),
                'clamp',
                [
                    SymbolRef('x', c_float()),
                    SymbolRef('minval', c_float()),
                    SymbolRef('maxval', c_float())
                ],
                Return(
                    FunctionCall(
                        'fmin',
                        [
                            FunctionCall(
                                'fmax',
                                [SymbolRef('x'), SymbolRef('minval')]
                            ),
                            SymbolRef('maxval')
                        ]
                    )
                )
            ),
            FunctionDecl(
                None,
                'warp_img_2D',
                [
                    SymbolRef('input', POINTER(c_float)()),
                    SymbolRef('u', POINTER(c_float)()),
                    SymbolRef('v', POINTER(c_float)()),
                    SymbolRef('output', POINTER(c_float)())
                ],
                For(
                    Assign(SymbolRef('x', c_int()), Constant(0)),
                    Lt(SymbolRef('x'), len_x),
                    PostInc(SymbolRef('x')),
                    [
                        For(
                            Assign(SymbolRef('y', c_int()), Constant(0)),
                            Lt(SymbolRef('y'), len_y),
                            PostInc(SymbolRef('y')),
                            [
                                Assign(
                                    SymbolRef('my_x', c_int()),
                                    Cast(c_int(),
                                         ArrayRef(SymbolRef('u'),
                                                  Add(SymbolRef('x'),
                                                      Mul(SymbolRef('y'),
                                                          len_x))))),
                                Assign(
                                    SymbolRef('my_y', c_int()),
                                    Cast(c_int(),
                                         ArrayRef(SymbolRef('v'),
                                                  Add(SymbolRef('x'),
                                                      Mul(SymbolRef('y'),
                                                          len_x))))),
                                Assign(
                                    SymbolRef('xfrac', c_float()),
                                    Sub(ArrayRef(SymbolRef('u'),
                                                 Add(SymbolRef('x'),
                                                     Mul(len_x,
                                                         SymbolRef('y')))),
                                        SymbolRef('my_x'))),
                                Assign(
                                    SymbolRef('yfrac', c_float()),
                                    Sub(ArrayRef(SymbolRef('v'),
                                                 Add(SymbolRef('x'),
                                                     Mul(len_x,
                                                         SymbolRef('y')))),
                                        SymbolRef('my_y'))),
                                If(Lt(
                                    ArrayRef(SymbolRef('u'),
                                             Add(SymbolRef('x'),
                                                 Mul(len_x, SymbolRef('y')))),
                                    Constant(0.0)),
                                    [
                                        PostDec('my_x'),
                                        Assign(SymbolRef('xfrac'),
                                               Add(Constant(1.0),
                                                   SymbolRef('xfrac')))
                                    ]),
                                If(Lt(
                                    ArrayRef(SymbolRef('v'),
                                             Add(SymbolRef('x'),
                                                 Mul(len_x, SymbolRef('y')))),
                                    Constant(0.0)),
                                    [
                                        PostDec('my_y'),
                                        Assign(SymbolRef('yfrac'),
                                               Add(Constant(1.0),
                                                   SymbolRef('yfrac')))
                                    ]),
                                Assign(SymbolRef('tmp', c_float()),
                                       Constant(0.0)),
                                If(
                                    And(
                                        And(
                                            GtE(Add(SymbolRef('x'),
                                                    SymbolRef('my_x')),
                                                Constant(0)),
                                            Lt(Add(SymbolRef('x'),
                                                   Add(SymbolRef('my_x'),
                                                       Constant(1))), len_x)
                                        ),
                                        And(
                                            GtE(Add(SymbolRef('y'),
                                                    SymbolRef('my_y')),
                                                Constant(0)),
                                            Lt(Add(SymbolRef('y'),
                                                   Add(SymbolRef('my_y'),
                                                       Constant(1))), len_y)
                                        )
                                    ),
                                    [
                                        AddAssign(
                                            SymbolRef('tmp'),
                                            Mul(Mul(
                                                ArrayRef(
                                                    SymbolRef('input'),
                                                    Add(
                                                        Add(
                                                            SymbolRef('x'),
                                                            SymbolRef('my_x')
                                                        ),
                                                        Mul(len_x, Add(
                                                            SymbolRef('y'),
                                                            SymbolRef('my_y')
                                                        )))),
                                                Sub(Constant(1.0),
                                                    SymbolRef('xfrac'))),
                                                Sub(Constant(1.0),
                                                    SymbolRef('yfrac')))),
                                        AddAssign(
                                            SymbolRef('tmp'),
                                            Mul(Mul(ArrayRef(
                                                SymbolRef('input'),
                                                Add(Add(
                                                    Add(SymbolRef('x'),
                                                        SymbolRef('my_x')),
                                                    Constant(1)),
                                                    Mul(
                                                        len_x,
                                                        Add(SymbolRef('y'),
                                                            SymbolRef('my_y'))
                                                    ))),
                                                SymbolRef('xfrac')),
                                                Sub(Constant(1.0),
                                                    SymbolRef('yfrac')))),
                                        AddAssign(
                                            SymbolRef('tmp'),
                                            Mul(Mul(ArrayRef(
                                                SymbolRef('input'),
                                                Add(Add(
                                                    SymbolRef('x'),
                                                    SymbolRef('my_x')),
                                                    Mul(
                                                        len_x,
                                                        Add(Add(
                                                            SymbolRef('y'),
                                                            SymbolRef('my_y')),
                                                            Constant(1))))),
                                                    Sub(Constant(1.0),
                                                        SymbolRef('xfrac'))),
                                                SymbolRef('yfrac'))),
                                        AddAssign(
                                            SymbolRef('tmp'),
                                            Mul(Mul(ArrayRef(
                                                SymbolRef('input'),
                                                Add(Add(
                                                    Add(SymbolRef('x'),
                                                        SymbolRef('my_x')),
                                                    Constant(1)),
                                                    Mul(
                                                        len_x,
                                                        Add(Add(
                                                            SymbolRef('y'),
                                                            SymbolRef('my_y')),
                                                            Constant(1))))),
                                                    SymbolRef('xfrac')),
                                                SymbolRef('yfrac'))),
                                        ],
                                    Assign(
                                        SymbolRef('tmp'),
                                        ArrayRef(
                                            SymbolRef('input'),
                                            Add(
                                                FunctionCall(
                                                    SymbolRef('clamp'),
                                                    [
                                                        Add(SymbolRef('x'),
                                                            SymbolRef('my_x')),
                                                        Constant(0),
                                                        Sub(len_x, Constant(1))
                                                    ]
                                                ),
                                                Mul(
                                                    len_x,
                                                    FunctionCall(
                                                        SymbolRef('clamp'),
                                                        [
                                                            Add(SymbolRef('y'),
                                                                SymbolRef(
                                                                    'my_y'
                                                                )),
                                                            Constant(0),
                                                            Sub(len_y,
                                                                Constant(1))
                                                        ]
                                                    ),
                                                    )
                                            )
                                        )
                                    )

                                ),
                                Assign(
                                    ArrayRef(SymbolRef('output'),
                                             Add(SymbolRef('x'),
                                                 Mul(len_x, SymbolRef('y')))),
                                    SymbolRef('tmp')
                                )

                            ]
                        )
                    ]
                )
            )
        ]
        proj = Project([CFile("generated", body)])
        return proj

    def finalize(self, tree, program_config):
        arg_cfg = program_config[0]
        param_types = [
            np.ctypeslib.ndpointer(arg[0], arg[2], arg[1])
            for arg in arg_cfg
        ]
        param_types.append(param_types[0])

        fn = WarpImg2DConcreteC()
        return fn.finalize("warp_img_2D", tree, CFUNCTYPE(c_void_p,
                                                          *param_types))


class WarpImg2D(object):
    def __new__(cls, backend='ocl'):
        if backend == 'python':
            cls.__call__ = cls.pure_python
            return object.__new__(cls)
        elif backend == 'c':
            return WarpImg2DLazyC(None)
        elif backend == 'ocl':
            return WarpImg2DLazyOcl(None)
        raise UnsupportedBackendError(
            "Teller found an unsupported backend: {0}".format(backend)
        )


warp_img2d = WarpImg2D()
