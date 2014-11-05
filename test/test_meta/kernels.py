import numpy as np
import pycl as cl
import ctypes as ct

from copy import deepcopy

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.c.nodes import FunctionCall, FunctionDecl, SymbolRef, Constant, \
    Assign, ArrayRef, Add, Ref, CFile, AddAssign, Sub, Cast, Mul, ArrayDef, \
    Div
from ctree.templates.nodes import StringTemplate
from ctree.ocl.nodes import OclFile
from ctree.ocl.macros import clSetKernelArg, get_global_id, NULL
from hindemith.meta.merge import MergeableInfo, FusableKernel, LoopDependence

import ctree.np

ctree.np  # Make pep happy


class OclFunc(ConcreteSpecializedFunction):
    def __init__(self):
        self.context = cl.clCreateContextFromType()
        self.queue = cl.clCreateCommandQueue(self.context)

    def finalize(self, kernel, tree, entry_name, entry_type):
        self.kernel = kernel
        self._c_function = self._compile(entry_name, tree, entry_type)
        return self

    def __call__(self, A):
        a_buf, evt = cl.buffer_from_ndarray(self.queue, A, blocking=False)
        evt.wait()
        B = np.zeros_like(A)
        b_buf, evt = cl.buffer_from_ndarray(self.queue, B, blocking=False)
        evt.wait()
        self._c_function(self.queue, self.kernel, a_buf, b_buf)
        _, evt = cl.buffer_to_ndarray(self.queue, b_buf, B)
        evt.wait()
        return B


def clamp(val, minimum, maximum):
    return FunctionCall(SymbolRef('clamp'), [
        Cast(ct.c_int(), val), Constant(minimum), Constant(maximum)
    ])


class Laplacian2D(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        A = args[0]
        return tuple(np.ctypeslib.ndpointer(A.dtype, A.ndim, A.shape)
                     for _ in args + (args[0], ))

    def transform(self, tree, program_config):
        arg_cfg = program_config[0]
        A = arg_cfg[0]
        B = arg_cfg[1]
        len_x, len_y = A._shape_
        # inner_type = A.__dtype__.type()

        output_index = ArrayRef(SymbolRef("B"),
                                Add(get_global_id(0),
                                    Mul(get_global_id(1), Constant(len_x))))

        kernel = FunctionDecl(
            None, SymbolRef("stencil_kernel"),
            params=[SymbolRef("A", A()).set_global(),
                    SymbolRef("B", B()).set_global()],
            defn=[Assign(output_index,
                         Mul(ArrayRef(SymbolRef('A'),
                             Add(get_global_id(0),
                                 Mul(get_global_id(1), Constant(len_x)))),
                             Constant(-4))),
                  AddAssign(output_index,
                            ArrayRef(
                                SymbolRef('A'),
                                Add(get_global_id(0),
                                    Mul(clamp(Add(get_global_id(1),
                                                  Constant(1)),
                                              0, len_y),
                                        Constant(len_x))))),
                  AddAssign(output_index,
                            ArrayRef(
                                SymbolRef('A'),
                                Add(get_global_id(0),
                                    Mul(clamp(Sub(get_global_id(1),
                                                  Constant(1)),
                                              0, len_y),
                                        Constant(len_x))))),
                  AddAssign(output_index,
                            ArrayRef(
                                SymbolRef('A'),
                                Add(clamp(Add(get_global_id(0), Constant(1)),
                                          0, len_x),
                                    Mul(get_global_id(1), Constant(len_x))))),
                  AddAssign(output_index,
                            ArrayRef(
                                SymbolRef('A'),
                                Add(clamp(Sub(get_global_id(0), Constant(1)),
                                          0, len_x),
                                    Mul(get_global_id(1), Constant(len_x))))),
                  ]
        ).set_kernel()

        file = OclFile("kernel", [kernel])

        control = [
            StringTemplate("""
                #ifdef __APPLE__
                #include <OpenCL/opencl.h>
                #else
                #include <CL/cl.h>
                #endif
            """),
            FunctionDecl(
                None, SymbolRef('control'),
                params=[SymbolRef('queue', cl.cl_command_queue()),
                        SymbolRef('kernel', cl.cl_kernel()),
                        SymbolRef('a', cl.cl_mem()),
                        SymbolRef('b', cl.cl_mem())],
                defn=[
                    ArrayDef(
                        SymbolRef('global', ct.c_ulong()), 2,
                        [Constant(len_x), Constant(len_y)]),
                    ArrayDef(
                        SymbolRef('local', ct.c_ulong()), 2,
                        [Constant(1), Constant(1)]),
                    clSetKernelArg('kernel', 0, ct.sizeof(cl.cl_mem), 'a'),
                    clSetKernelArg('kernel', 1, ct.sizeof(cl.cl_mem), 'b'),
                    FunctionCall(SymbolRef('clEnqueueNDRangeKernel'),
                                 [SymbolRef('queue'), SymbolRef('kernel'),
                                  Constant(2), Constant(0),
                                  SymbolRef('global'), SymbolRef('local'),
                                  Constant(0), NULL(), NULL()]),
                    FunctionCall(SymbolRef('clFinish'), [SymbolRef('queue')])
                ])
        ]

        proj = Project([file, CFile('control', control)])
        entry_type = [None, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem,
                      cl.cl_mem]
        return proj, "control", entry_type

    def finalize(self, proj, entry_point, entry_type):
        fn = OclFunc()
        program = cl.clCreateProgramWithSource(
            fn.context, proj.files[0].codegen()).build()
        ptr = program['stencil_kernel']
        return fn.finalize(ptr, proj, "control", ct.CFUNCTYPE(*entry_type))

    def get_placeholder_output(self, args):
        return np.zeros_like(args[0])

    def get_mergeable_info(self, args):
        arg_cfg = self.args_to_subconfig(args)
        tune_cfg = self.get_tuning_driver()
        program_cfg = (arg_cfg, tune_cfg)
        tree = deepcopy(self.original_tree)
        proj, entry_point, entry_type = self.transform(tree, program_cfg)
        return MergeableInfo(
            proj=proj,
            entry_point=entry_point,
            entry_type=entry_type,
            # TODO: This should use a namedtuple or object to be more explicit
            kernels=[proj.files[0]]
        )

laplacian_2d = Laplacian2D(None)


class YGradient(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        A = args[0]
        return tuple(np.ctypeslib.ndpointer(A.dtype, A.ndim, A.shape)
                     for _ in args + (args[0], ))

    def transform(self, tree, program_config):
        arg_cfg = program_config[0]
        A = arg_cfg[0]
        B = arg_cfg[1]
        len_x, len_y = A._shape_
        # inner_type = A.__dtype__.type()

        output_index = ArrayRef(SymbolRef("B"),
                                Add(get_global_id(0),
                                    Mul(get_global_id(1), Constant(len_x))))

        kernel = FunctionDecl(
            None, SymbolRef("stencil_kernel"),
            params=[SymbolRef("A", A()).set_global(),
                    SymbolRef("B", B()).set_global()],
            defn=[Assign(output_index,
                         Div(Sub(ArrayRef(
                             SymbolRef('A'),
                             Add(get_global_id(0),
                                 Mul(clamp(Add(get_global_id(1),
                                               Constant(1)),
                                           0, len_y),
                                     Constant(len_x)))),
                             ArrayRef(
                                 SymbolRef('A'),
                                 Add(get_global_id(0),
                                     Mul(clamp(Sub(get_global_id(1),
                                                   Constant(1)),
                                               0, len_y),
                                         Constant(len_x))))), Constant(2)))]
        ).set_kernel()

        file = OclFile("kernel", [kernel])

        control = [
            StringTemplate("""
                #ifdef __APPLE__
                #include <OpenCL/opencl.h>
                #else
                #include <CL/cl.h>
                #endif
            """),
            FunctionDecl(
                None, SymbolRef('control'),
                params=[SymbolRef('queue', cl.cl_command_queue()),
                        SymbolRef('kernel', cl.cl_kernel()),
                        SymbolRef('a', cl.cl_mem()),
                        SymbolRef('b', cl.cl_mem())],
                defn=[
                    ArrayDef(
                        SymbolRef('global', ct.c_ulong()), 2,
                        [Constant(len_x), Constant(len_y)]),
                    ArrayDef(
                        SymbolRef('local', ct.c_ulong()), 2,
                        [Constant(16), Constant(16)]),
                    clSetKernelArg('kernel', 0, ct.sizeof(cl.cl_mem), 'a'),
                    clSetKernelArg('kernel', 1, ct.sizeof(cl.cl_mem), 'b'),
                    FunctionCall(SymbolRef('clEnqueueNDRangeKernel'),
                                 [SymbolRef('queue'), SymbolRef('kernel'),
                                  Constant(2), Constant(0),
                                  SymbolRef('global'), SymbolRef('local'),
                                  Constant(0), NULL(), NULL()]),
                    FunctionCall(SymbolRef('clFinish'), [SymbolRef('queue')])
                ])
        ]

        proj = Project([file, CFile('control', control)])
        entry_type = [None, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem,
                      cl.cl_mem]
        return proj, "control", entry_type

    def finalize(self, proj, entry_point, entry_type):
        fn = OclFunc()
        program = cl.clCreateProgramWithSource(
            fn.context, proj.files[0].codegen()).build()
        ptr = program['stencil_kernel']
        return fn.finalize(ptr, proj, "control", ct.CFUNCTYPE(*entry_type))

    def get_placeholder_output(self, args):
        return np.zeros_like(args[0])

    def get_mergeable_info(self, args):
        arg_cfg = self.args_to_subconfig(args)
        tune_cfg = self.get_tuning_driver()
        program_cfg = (arg_cfg, tune_cfg)
        tree = deepcopy(self.original_tree)
        proj, entry_point, entry_type = self.transform(tree, program_cfg)
        control = proj.find(CFile).find(FunctionDecl)
        local_size, global_size = control.defn[:2]
        arg_setters = control.defn[2:4]
        enqueue_call = control.defn[4]
        kernel_decl = proj.find(OclFile).find(FunctionDecl)
        global_loads = []
        global_stores = []
        kernel = proj.find(OclFile)
        return MergeableInfo(
            proj=proj,
            entry_point=entry_point,
            entry_type=entry_type,
            # TODO: This should use a namedtuple or object to be more explicit
            kernels=[kernel],
            fusable_node=FusableKernel(
                (16, 16), tuple(value for value in global_size.body),
                arg_setters, enqueue_call, kernel_decl, global_loads,
                global_stores,
                [LoopDependence(0, (0, -1)),
                 LoopDependence(0, (0, 1))])
        )


class XGradient(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        A = args[0]
        return tuple(np.ctypeslib.ndpointer(A.dtype, A.ndim, A.shape)
                     for _ in args + (args[0], ))

    def transform(self, tree, program_config):
        arg_cfg = program_config[0]
        A = arg_cfg[0]
        B = arg_cfg[1]
        len_x, len_y = A._shape_
        # inner_type = A.__dtype__.type()

        output_index = ArrayRef(SymbolRef("B"),
                                Add(get_global_id(0),
                                    Mul(get_global_id(1), Constant(len_x))))

        kernel = FunctionDecl(
            None, SymbolRef("stencil_kernel"),
            params=[SymbolRef("A", A()).set_global(),
                    SymbolRef("B", B()).set_global()],
            defn=[Assign(output_index,
                         Div(Sub(ArrayRef(
                             SymbolRef('A'),
                             Add(clamp(Add(get_global_id(0), Constant(1)),
                                       0, len_x),
                                 Mul(get_global_id(1),
                                     Constant(len_x)))),
                             ArrayRef(
                                 SymbolRef('A'),
                                 Add(clamp(Sub(get_global_id(0), Constant(1)),
                                           0, len_x),
                                     Mul(get_global_id(1),
                                         Constant(len_x))))), Constant(2)))]
        ).set_kernel()

        file = OclFile("kernel", [kernel])

        control = [
            StringTemplate("""
                #ifdef __APPLE__
                #include <OpenCL/opencl.h>
                #else
                #include <CL/cl.h>
                #endif
            """),
            FunctionDecl(
                None, SymbolRef('control'),
                params=[SymbolRef('queue', cl.cl_command_queue()),
                        SymbolRef('kernel', cl.cl_kernel()),
                        SymbolRef('a', cl.cl_mem()),
                        SymbolRef('b', cl.cl_mem())],
                defn=[
                    ArrayDef(
                        SymbolRef('global', ct.c_ulong()), 2,
                        [Constant(len_x), Constant(len_y)]),
                    ArrayDef(
                        SymbolRef('local', ct.c_ulong()), 2,
                        [Constant(16), Constant(16)]),
                    clSetKernelArg('kernel', 0, ct.sizeof(cl.cl_mem), 'a'),
                    clSetKernelArg('kernel', 1, ct.sizeof(cl.cl_mem), 'b'),
                    FunctionCall(SymbolRef('clEnqueueNDRangeKernel'),
                                 [SymbolRef('queue'), SymbolRef('kernel'),
                                  Constant(2), Constant(0),
                                  SymbolRef('global'), SymbolRef('local'),
                                  Constant(0), NULL(), NULL()]),
                    FunctionCall(SymbolRef('clFinish'), [SymbolRef('queue')])
                ])
        ]

        proj = Project([file, CFile('control', control)])
        entry_type = [None, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem,
                      cl.cl_mem]
        return proj, "control", entry_type

    def finalize(self, proj, entry_point, entry_type):
        fn = OclFunc()
        program = cl.clCreateProgramWithSource(
            fn.context, proj.files[0].codegen()).build()
        ptr = program['stencil_kernel']
        return fn.finalize(ptr, proj, "control", ct.CFUNCTYPE(*entry_type))

    def get_placeholder_output(self, args):
        return np.zeros_like(args[0])

    def get_mergeable_info(self, args):
        arg_cfg = self.args_to_subconfig(args)
        tune_cfg = self.get_tuning_driver()
        program_cfg = (arg_cfg, tune_cfg)
        tree = deepcopy(self.original_tree)
        proj, entry_point, entry_type = self.transform(tree, program_cfg)
        control = proj.find(CFile).find(FunctionDecl)
        local_size, global_size = control.defn[:2]
        arg_setters = control.defn[2:4]
        enqueue_call = control.defn[4]
        kernel_decl = proj.find(OclFile).find(FunctionDecl)
        global_loads = []
        global_stores = []
        kernel = proj.find(OclFile)
        return MergeableInfo(
            proj=proj,
            entry_point=entry_point,
            entry_type=entry_type,
            # TODO: This should use a namedtuple or object to be more explicit
            kernels=[kernel],
            fusable_node=FusableKernel(
                (16, 16), tuple(value for value in global_size.body),
                arg_setters, enqueue_call, kernel_decl, global_loads,
                global_stores,
                [LoopDependence(0, (-1, 0)),
                 LoopDependence(0, (1, 0))])
        )


x_gradient = XGradient(None)
y_gradient = YGradient(None)
