from hindemith.types.hmarray import NdArrCfg, kernel_range, hmarray, for_range
from .map import MapOclTransform, ElementReference, \
    StoreOutput

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.c.nodes import SymbolRef, FunctionDecl, CFile
from ctree.nodes import Project
from ctree.ocl import get_context_and_queue_from_devices
from ctree.templates.nodes import StringTemplate
from ctree.transformations import PyBasicConversions
from ctree.frontend import get_ast
from ctree.omp.macros import IncludeOmpHeader
from ctree.omp.nodes import OmpParallelFor

import numpy as np
import ctypes as ct
import pycl as cl
import sys


class CConcreteZipWith(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        self._c_function = self._compile(entry_name, proj, entry_type)

    def __call__(self, *args):
        output = hmarray(np.zeros_like(args[1]))
        self._c_function(*(args[1:] + (output, )))
        return output


class OclConcreteZipWith(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        self._c_function = self._compile(entry_name, proj, entry_type)
        devices = cl.clGetDeviceIDs()
        self.context, self.queue = get_context_and_queue_from_devices(
            [devices[-1]])

    def finalize(self, kernel):
        self.kernel = kernel
        return self

    def __call__(self, f, *args):
        output = hmarray(np.zeros_like(args[0]))
        out_buf, evt = cl.buffer_from_ndarray(self.queue, output,
                                              blocking=True)
        evt.wait()
        output._ocl_buf = out_buf
        output._ocl_dirty = False
        output._host_dirty = True
        processed = [arg.ocl_buf for arg in args]
        processed.extend([out_buf, self.queue, self.kernel])
        self._c_function(*processed)
        return output


class ZipWithFrontendTransformer(PyBasicConversions):
    def visit_FunctionDef(self, node):
        if sys.version_info < (3, 0):
            self.targets = [arg.id for arg in node.args.args]
        else:
            self.targets = [arg.arg for arg in node.args.args]
        node.body = list(map(self.visit, node.body))
        return node

    def visit_Name(self, node):
        for index, target in enumerate(self.targets):
            if target == node.id:
                return ElementReference("arg{}".format(index))
        return PyBasicConversions.visit_Name(self, node)

    def visit_Return(self, node):
        return StoreOutput("arg{}".format(len(self.targets)),
                           self.visit(node.value))


ocl_header = StringTemplate("""
                #ifdef __APPLE__
                #include <OpenCL/opencl.h>
                #else
                #include <CL/cl.h>
                #endif
                """)


class ZipWith(LazySpecializedFunction):
    backend = 'ocl'

    def args_to_subconfig(self, args):
        """TODO: Type check"""
        arg_cfgs = (args[0], )
        out_cfg = None
        for arg in args[1:]:
            arg_cfgs += (NdArrCfg(arg.dtype, arg.ndim, arg.shape), )
            out_cfg = (NdArrCfg(arg.dtype, arg.ndim, arg.shape), )
        return arg_cfgs + out_cfg

    def process_arg_types(self, arg_cfg):
        arg_types, kernel_arg_types = (), ()
        if self.backend == 'c' or self.backend == 'omp':
            for cfg in arg_cfg:
                arg_types += (np.ctypeslib.ndpointer(cfg.dtype, cfg.ndim,
                                                     cfg.shape),)
        elif self.backend == 'ocl':
            for cfg in arg_cfg:
                kernel_arg_types += (
                    np.ctypeslib.ndpointer(cfg.dtype, cfg.ndim, cfg.shape),)
                arg_types += (cl.cl_mem, )
        return arg_types, kernel_arg_types

    def build_type_table(self, arg_types, kernel_arg_types):
        type_table = {}
        if self.backend == 'c' or self.backend == 'omp':
            for index, t in enumerate(arg_types):
                type_table['arg{}'.format(index)] = t
        elif self.backend == 'ocl':
            for index, t in enumerate(kernel_arg_types):
                type_table['arg{}'.format(index)] = t
        return type_table

    def transform(self, tree, program_config):
        arg_cfg, tune_cfg = program_config
        if hasattr(arg_cfg[0], '_hm_symbols'):
            symbols = arg_cfg[0]._hm_symbols
        else:
            symbols = {}
        tree = get_ast(arg_cfg[0])
        arg_cfg = arg_cfg[1:]

        arg_types, kernel_arg_types = self.process_arg_types(arg_cfg)

        tree = ZipWithFrontendTransformer().visit(tree).files[0].body[0].body

        func = FunctionDecl(
            None,
            SymbolRef('zip_with'),
            [SymbolRef('arg{}'.format(index), type())
             for index, type in enumerate(arg_types)],
            []
        )
        proj = Project([CFile('map', [func])])
        type_table = self.build_type_table(arg_types, kernel_arg_types)
        backend = MapOclTransform(symbols, type_table)
        loop_body = list(map(backend.visit, tree))
        shape = arg_cfg[0].shape[::-1]
        if self.backend == 'c' or self.backend == 'omp':
            if self.backend == 'omp':
                func.defn.append(OmpParallelFor())
                proj.files[0].config_target = 'omp'
                proj.files[0].body.insert(0, IncludeOmpHeader())
            func.defn.append(for_range(shape, 1, loop_body))
            entry_type = ct.CFUNCTYPE(*((None,) + arg_types))
            return CConcreteZipWith('zip_with', proj, entry_type)
        elif self.backend == 'ocl':
            proj.files[0].body.insert(0, ocl_header)
            arg_types += (cl.cl_command_queue, cl.cl_kernel)
            control, kernel = kernel_range(shape, shape,
                                           kernel_arg_types, loop_body)
            func.params.extend((
                SymbolRef('queue', cl.cl_command_queue()),
                SymbolRef(kernel.body[0].name.name, cl.cl_kernel())
            ))
            func.defn = control
            entry_type = ct.CFUNCTYPE(*((None,) + arg_types))
            fn = OclConcreteZipWith('zip_with', proj, entry_type)
            program = cl.clCreateProgramWithSource(
                fn.context, kernel.codegen()).build()
            return fn.finalize(program[kernel.body[0].name.name])


specialized_zip_with = ZipWith(None)


def zip_with(f, *args):
    return specialized_zip_with(f, *args)
