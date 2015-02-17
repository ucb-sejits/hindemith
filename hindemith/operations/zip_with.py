from hindemith.types.hmarray import NdArrCfg, hmarray, \
    for_range, Loop
from hindemith.nodes import kernel_range
from .map import MapOclTransform, ElementReference, \
    StoreOutput

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.c.nodes import SymbolRef, FunctionDecl, CFile, FunctionCall, Assign
from ctree.nodes import Project
from ctree.ocl import get_context_and_queue_from_devices
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
from ctree.transformations import PyBasicConversions
from ctree.frontend import get_ast
from ctree.omp.macros import IncludeOmpHeader
from ctree.omp.nodes import OmpParallelFor

import numpy as np
import ctypes as ct
import pycl as cl
import sys
import copy


class CConcreteZipWith(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        self._c_function = self._compile(entry_name, proj, entry_type)

    def __call__(self, *args):
        output = hmarray(np.empty_like(args[0]))
        self._c_function(*(args + (output, )))
        return output


class OclConcreteZipWith(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        self._c_function = self._compile(entry_name, proj, entry_type)
        devices = cl.clGetDeviceIDs()
        print(proj.files[0])
        print(proj.files[1])
        self.context, self.queue = get_context_and_queue_from_devices(
            [devices[-1]])

    def finalize(self, kernel):
        self.kernel = kernel
        return self

    def __call__(self, *args):
        output = hmarray(np.zeros_like(args[0]))
        out_buf = cl.clCreateBuffer(self.context, output.nbytes)
        output._ocl_buf = out_buf
        output._ocl_dirty = False
        output._host_dirty = True
        processed = [arg.ocl_buf for arg in args]
        processed = [self.queue, self.kernel] + processed + [out_buf]
        self._c_function(*processed)
        return output


class ZipWithFrontendTransformer(PyBasicConversions):
    def __init__(self, symbols, params):
        self.symbols = symbols
        self.params = params
        self.seen = {}
        super(ZipWithFrontendTransformer, self).__init__()

    def visit_FunctionDef(self, node):
        if sys.version_info < (3, 0):
            self.targets = [arg.id for arg in node.args.args]
        else:
            self.targets = [arg.arg for arg in node.args.args]
        node.body = list(map(self.visit, node.body))
        return node

    def visit_Call(self, node):
        orig_name = node.func.id
        node = super(ZipWithFrontendTransformer, self).visit_Call(node)
        if isinstance(node, FunctionCall):
            node.func.name = orig_name
        return node

    def visit_Name(self, node):
        for index, target in enumerate(self.targets):
            if target == node.id:
                return ElementReference(self.params[index].name)
        node = PyBasicConversions.visit_Name(self, node)
        if node.name in self.symbols:
            # Handled by backend
            # FIXME: do this in frontend or something
            return node
        if node.name not in self.seen:
            self.seen[node.name] = SymbolRef.unique().name
        node.name = self.seen[node.name]
        return node

    def visit_Return(self, node):
        return StoreOutput(self.params[-1].name,
                           self.visit(node.value))

    def visit_Assign(self, node):
        if len(node.targets) > 1:
            # Raise exception?
            return node
        target = self.visit(node.targets[0])
        value = self.visit(node.value)
        return Assign(target, value)


ocl_header = StringTemplate("""
                #ifdef __APPLE__
                #include <OpenCL/opencl.h>
                #else
                #include <CL/cl.h>
                #endif
                """)


class ZipWith(LazySpecializedFunction):
    backend = 'ocl'

    def __init__(self, symbols, tree):
        super(ZipWith, self).__init__(tree)
        self.symbols = symbols


    def args_to_subconfig(self, args):
        """TODO: Type check"""
        arg_cfgs = ()
        out_cfg = None
        for arg in args:
            arg_cfgs += (NdArrCfg(arg.dtype, arg.ndim, arg.shape), )
            out_cfg = (NdArrCfg(arg.dtype, arg.ndim, arg.shape), )
        return arg_cfgs + out_cfg

    def process_arg_types(self, arg_cfg):
        arg_types, params, kernel_params = (), [], ()
        if self.backend == 'c' or self.backend == 'omp':
            for cfg in arg_cfg:
                arg_types += (np.ctypeslib.ndpointer(cfg.dtype, cfg.ndim,
                                                     cfg.shape),)
                params.append(SymbolRef.unique(sym_type=arg_types[-1]()))
        elif self.backend == 'ocl':
            for cfg in arg_cfg:
                arg_types += (cl.cl_mem, )
                params.append(SymbolRef.unique(sym_type=arg_types[-1]()))
                kernel_params += (SymbolRef(
                    params[-1].name,
                    np.ctypeslib.ndpointer(cfg.dtype, cfg.ndim, cfg.shape)()),)
        return arg_types, params, kernel_params

    def build_type_table(self, params, kernel_params):
        type_table = {}
        if self.backend == 'c' or self.backend == 'omp':
            for index, param in enumerate(params):
                type_table[param.name] = param.type.__class__
        elif self.backend == 'ocl':
            for index, param in enumerate(kernel_params):
                type_table[param.name] = param.type.__class__
        return type_table

    def transform(self, tree, program_config):
        arg_cfg, tune_cfg = program_config

        arg_types, params, kernel_params = self.process_arg_types(arg_cfg)

        tree = ZipWithFrontendTransformer(self.symbols,
            params).visit(tree).files[0].body[0].body

        func = FunctionDecl(
            None,
            SymbolRef('zip_with'),
            params,
            []
        )
        cfile = CFile('map', [func])
        type_table = self.build_type_table(params, kernel_params)
        backend = MapOclTransform(self.symbols, type_table)
        loop_body = list(map(backend.visit, tree))
        shape = arg_cfg[0].shape[::-1]
        if self.backend == 'c' or self.backend == 'omp':
            if self.backend == 'omp':
                func.defn.append(OmpParallelFor())
                cfile.config_target = 'omp'
                cfile.body.insert(0, IncludeOmpHeader())
            func.defn.append(for_range(shape, 1, loop_body))
            return [cfile]
        elif self.backend == 'ocl':
            cfile.config_target = 'opencl'
            cfile.body.insert(0, ocl_header)
            arg_types = (cl.cl_command_queue, cl.cl_kernel) + arg_types
            control, kernel = kernel_range(shape, shape,
                                           kernel_params, loop_body)
            func.params.insert(0, SymbolRef('queue', cl.cl_command_queue()))
            func.params.insert(1, SymbolRef(kernel.body[0].name.name,
                                            cl.cl_kernel()))
            func.defn = control
            return [cfile, kernel]

    def finalize(self, files, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        arg_types, params, kernel_params = self.process_arg_types(arg_cfg)
        proj = Project(files)
        entry_name = 'zip_with'
        if self.backend == 'c' or self.backend == 'omp':
            entry_type = (None,) + arg_types
            entry_type = ct.CFUNCTYPE(*entry_type)
            return CConcreteZipWith(entry_name, proj, entry_type)
        elif self.backend == 'ocl':
            arg_types = (cl.cl_command_queue, cl.cl_kernel) + arg_types
            entry_type = (None,) + arg_types
            entry_type = ct.CFUNCTYPE(*entry_type)
            fn = OclConcreteZipWith(entry_name, proj, entry_type)
            kernel = proj.find(OclFile)
            program = cl.clCreateProgramWithSource(
                fn.context, kernel.codegen()).build()
            return fn.finalize(program[kernel.name])

    def get_placeholder_output(self, args):
        return hmarray(np.empty_like(args[0]))

    def get_ir_nodes(self, args):
        arg_cfg = self.args_to_subconfig(args)
        tree = copy.deepcopy(self.original_tree)

        types = ()
        params = []
        type_table = {}
        for cfg in arg_cfg:
            types += (np.ctypeslib.ndpointer(cfg.dtype, cfg.ndim,
                                             cfg.shape),)
            params.append(SymbolRef.unique(sym_type=types[-1]()))

            type_table[params[-1].name] = types[-1]

        tree = ZipWithFrontendTransformer(
            self.symbols, params).visit(tree).files[0].body[0].body

        backend = MapOclTransform(self.symbols, type_table)
        loop_body = list(map(backend.visit, tree))
        shape = arg_cfg[0].shape[::-1]
        return [Loop(shape, params[:-1], [params[-1]], types, loop_body)]


def zip_with(fn):
    symbols = {}
    if hasattr(fn, '_hm_symbols'):
        symbols = fn._hm_symbols
    return ZipWith(symbols, get_ast(fn))
