import numpy as np
import pycl as cl
import ctypes as ct

from copy import deepcopy

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.c.nodes import FunctionCall, FunctionDecl, SymbolRef, Constant, \
    Assign, ArrayRef, Add, Ref, CFile
from ctree.templates.nodes import StringTemplate
from ctree.ocl.nodes import OclFile
from ctree.ocl.macros import clSetKernelArg, get_global_id, NULL
from ctree.meta.merge import MergeableInfo, FusableKernel, LoopDependence

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

    def __call__(self, A, B):
        a_buf, evt = cl.buffer_from_ndarray(self.queue, A, blocking=True)
        evt.wait()
        b_buf, evt = cl.buffer_from_ndarray(self.queue, B, blocking=True)
        evt.wait()
        C = np.zeros_like(A)
        c_buf, evt = cl.buffer_from_ndarray(self.queue, C, blocking=True)
        evt.wait()
        self._c_function(self.queue, self.kernel, a_buf, b_buf, c_buf)
        _, evt = cl.buffer_to_ndarray(self.queue, c_buf, C)
        evt.wait()
        return C


class OclAdd(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        A = args[0]
        return tuple(np.ctypeslib.ndpointer(A.dtype, A.ndim, A.shape)
                     for _ in args + (args[0], ))

    def transform(self, tree, program_config):
        arg_cfg = program_config[0]
        A = arg_cfg[0]
        B = arg_cfg[1]
        C = arg_cfg[2]
        len_A = np.prod(A._shape_)
        # inner_type = A.__dtype__.type()

        kernel = FunctionDecl(
            None, SymbolRef("add_kernel"),
            params=[SymbolRef("A", A()).set_global(),
                    SymbolRef("B", B()).set_global(),
                    SymbolRef("C", C()).set_global()],
            defn=[Assign(ArrayRef(SymbolRef("C"), get_global_id(0)),
                         Add(ArrayRef(SymbolRef('A'), get_global_id(0)),
                             ArrayRef(SymbolRef('B'), get_global_id(0))))]
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
                        SymbolRef('b', cl.cl_mem()),
                        SymbolRef('c', cl.cl_mem())],
                defn=[
                    Assign(SymbolRef('global', ct.c_ulong()), Constant(len_A)),
                    Assign(SymbolRef('local', ct.c_ulong()), Constant(32)),
                    clSetKernelArg('kernel', 0, ct.sizeof(cl.cl_mem), 'a'),
                    clSetKernelArg('kernel', 1, ct.sizeof(cl.cl_mem), 'b'),
                    clSetKernelArg('kernel', 2, ct.sizeof(cl.cl_mem), 'c'),
                    FunctionCall(SymbolRef('clEnqueueNDRangeKernel'),
                                 [SymbolRef('queue'), SymbolRef('kernel'),
                                  Constant(1), Constant(0),
                                  Ref(SymbolRef('global')),
                                  Ref(SymbolRef('local')), Constant(0),
                                  NULL(), NULL()]),
                    FunctionCall(SymbolRef('clFinish'), [SymbolRef('queue')])
                ])
        ]

        proj = Project([file, CFile('control', control)])
        # print(proj.files[1])
        entry_type = [None, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem,
                      cl.cl_mem, cl.cl_mem]
        return proj, "control", entry_type

    def finalize(self, proj, entry_point, entry_type):
        fn = OclFunc()
        program = cl.clCreateProgramWithSource(fn.context,
                                               proj.files[0].codegen()).build()
        ptr = program['add_kernel']
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
        arg_setters = control.defn[2:5]
        enqueue_call = control.defn[5]
        kernel_decl = proj.find(OclFile).find(FunctionDecl)
        global_loads = [kernel_decl.defn[0].right.left.left,
                        kernel_decl.defn[0].right.right.left]
        global_stores = [kernel_decl.defn[0]]
        kernel = proj.find(OclFile)
        return MergeableInfo(
            proj=proj,
            entry_point=entry_point,
            entry_type=entry_type,
            # TODO: This should use a namedtuple or object to be more explicit
            kernels=[kernel],
            fusable_node=FusableKernel((32,), (global_size.right.value,),
                                       arg_setters,
                                       enqueue_call, kernel_decl,
                                       global_loads, global_stores,
                                       [LoopDependence(0, (0, )),
                                        LoopDependence(1, (0, )),
                                        LoopDependence(2, (0, ))])
        )


array_add = OclAdd(None)
