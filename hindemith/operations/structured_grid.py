from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.frontend import get_ast
from hindemith.types.hmarray import hmarray, NdArrCfg, kernel_range
from ctree.transformations import PyBasicConversions
from ctree.nodes import CtreeNode
from ctree.c.nodes import SymbolRef, BinaryOp, Mul, Constant, ArrayRef, Op
from ctree.templates.nodes import StringTemplate
from ctree.ocl import get_context_and_queue_from_devices

from copy import deepcopy

import numpy as np
import ctypes as ct
import pycl as cl
import ast


class ArrayReference(CtreeNode):
    _fields = ['index']

    def __init__(self, array=None, index=None):
        self.array = array
        self.index = index

    def label(self):
        return '' + self.array.name


class IterByIndex(CtreeNode):
    _fields = ['targets', 'iter', 'body']

    def __init__(self, targets=None, iter=None, body=None, offsets=None):
        self.targets = targets
        self.iter = iter
        self.body = body
        self.offsets = offsets

    def label(self):
        return 'offsets: ' + self.offsets


class Index(CtreeNode):
    _fields = ['elts']

    def __init__(self, elts=None):
        self.elts = elts

    def label(self):
        return ''


class StructuredGridFrontend(PyBasicConversions):
    def visit_For(self, node):
        target = node.target
        if isinstance(target, ast.Tuple):
            self.offsets = [[] for _ in target.elts]
            targets = Index(target.elts)
        else:
            raise NotImplementedError("Can't handle this type of iteration")
        _iter = self.visit(node.iter)
        body = [self.visit(s) for s in node.body]
        return IterByIndex(targets, _iter, body, self.offsets)

    def visit_Subscript(self, node):
        array = self.visit(node.value)
        index = self.visit(node.slice.value)
        if isinstance(index, ast.Tuple):
            for i, elt in enumerate(index.elts):
                elts = []
                if isinstance(elt, SymbolRef):
                    elts.append(elt)
                elif isinstance(elt, BinaryOp):
                    if isinstance(elt.op, Op.Sub):
                        self.offsets[i].append(-elt.right.value)
                    elif isinstance(elt.op, Op.Add):
                        self.offsets[i].append(elt.right.value)
                else:
                    raise NotImplementedError(
                        "Can't handle this type of index: {}".format(elt))
            index = Index(index.elts)
        else:
            raise NotImplementedError(
                "Can't handle this type of subscript: {}".format(index))
        return ArrayReference(array, index)


class StructuredGridBackend(ast.NodeTransformer):
    def __init__(self, arg_cfg, arg_types, params, kernel_params, border_type,
                 cval):
        self.arg_cfg = arg_cfg
        self.arg_table = {}
        self.arg_types = arg_types
        self.params = params
        self.kernel_params = kernel_params
        self.kernels = []
        self.arg_name_map = {}
        self.curr_offset = None
        self.border_type = border_type
        self.cval = cval
        super(StructuredGridBackend, self).__init__()

    def visit_Project(self, node):
        self.project = node
        node.files[0] = self.visit(node.files[0])
        self.project = None
        return node

    def visit_CFile(self, node):
        self.cfile = node
        # Weirdness because we insert into CFile's body during
        # recursive visiting
        # TODO: Do this more elegantly
        _ = self.visit(node.body[0])
        node.body[1] = _
        self.cfile = None
        return node

    def visit_FunctionDecl(self, node):
        for index, param in enumerate(node.params):
            self.arg_table[param.name] = self.arg_cfg[index]
            self.arg_name_map[param.name] = self.params[index].name
            param.name = self.arg_name_map[param.name]
            param.type = self.arg_types[index]()
        node.params.extend((
            SymbolRef('queue', cl.cl_command_queue()),
        ))
        node.defn = self.visit(node.defn[0])
        for index, kernel in enumerate(self.kernels):
            print(kernel)
            node.params.append(SymbolRef(kernel.name,
                                         cl.cl_kernel()))
            node.name = 'control'
        print(node)
        return node

    def visit_IterByIndex(self, node):
        shape = tuple(reversed(self.arg_table[node.iter.name].shape))
        g_size = list(shape)
        controls, kernels = [], []
        offset = [0 for _ in g_size]
        for index, vals in enumerate(node.offsets):
            negs = list(filter(lambda x: x < 0, vals))
            pos = list(filter(lambda x: x > 0, vals))
            if len(negs) > 0:
                bottom_border = min(negs)
                g_size[index] += bottom_border
                offset[index] = abs(bottom_border)
                if self.border_type == 'constant':
                    b_size = list(shape[:])
                    os = [0 for _ in b_size]
                    os[index] = abs(bottom_border)
                    b_size[1 - index] = abs(bottom_border)
                    b_size[index] -= abs(bottom_border)
                    self.curr_offset = index
                    body = [self.visit(s) for s in deepcopy(node.body)]
                    self.curr_offset = None
                    control, kernel = kernel_range(shape, b_size,
                                                   self.kernel_params,
                                                   body, os)
                    controls.append(control)
                    kernels.append(kernel)
                    self.kernels.append(kernel)
            if len(pos) > 0:
                top_border = min(pos)
                g_size[1 - index] -= top_border
                if self.border_type == 'constant':
                    b_size = list(shape[:])
                    os = [0 for _ in b_size]
                    os[index] = g_size[index] - top_border
                    b_size[index] = abs(top_border)
                    b_size[1 - index] -= abs(top_border)
                    self.curr_offset = index
                    body = [self.visit(s) for s in deepcopy(node.body)]
                    self.curr_offset = None
                    control, kernel = kernel_range(shape, b_size,
                                                   self.kernel_params,
                                                   body, os)
                    controls.append(control)
                    kernels.append(kernel)
                    self.kernels.append(kernel)
        body = [self.visit(s) for s in node.body]
        self.project.files.extend(kernels)

        control, kernel = kernel_range(shape, g_size,
                                       self.kernel_params, body, offset)
        self.kernels.append(kernel)
        for launch in controls:
            control.extend(launch)

        self.project.files.append(kernel)
        self.cfile.body.insert(0, StringTemplate("""
            #ifdef __APPLE__
            #include <OpenCL/opencl.h>
            #else
            #include <CL/cl.h>
            #endif
            """))
        return control

    def visit_ArrayReference(self, node):
        shape = self.arg_table[node.array.name].shape
        elts = node.index.elts
        idx = SymbolRef('loop_idx')
        for index, elt in enumerate(elts):
            if isinstance(elt, BinaryOp):
                if self.curr_offset is not None and index == self.curr_offset:
                    return Constant(self.cval)
                step = np.prod(shape[index + 1:])
                elt.left = idx
                if step > 1:
                    elt.right = Mul(elt.right, Constant(step))
                idx = elt
        return ArrayRef(self.visit(node.array), idx)

    def visit_SymbolRef(self, node):
        if node.name in self.arg_name_map:
            node.name = self.arg_name_map[node.name]
        return node


class OclConcreteStructuredGrid(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        super(OclConcreteStructuredGrid, self).__init__()
        self._c_function = self._compile(entry_name, proj, entry_type)
        devices = cl.clGetDeviceIDs()
        self.context, self.queue = get_context_and_queue_from_devices(
            [devices[-1]])

    def finalize(self, kernels):
        self.kernels = kernels
        return self

    def __call__(self, *args):
        output = None
        out_buf = None
        processed = []
        for arg in args:
            if output is None:
                output = hmarray(np.zeros_like(arg))
                out_buf, evt = cl.buffer_from_ndarray(self.queue, output,
                                                      blocking=True)
                output._ocl_buf = out_buf
                output._ocl_dirty = False
                output._host_dirty = True
            evt.wait()
            processed.append(arg.ocl_buf)
        self._c_function(*(processed + [out_buf, self.queue] + self.kernels))
        return output


class StructuredGrid(LazySpecializedFunction):
    backend = 'ocl'

    def __init__(self, ast, border_type, cval):
        super(StructuredGrid, self).__init__(ast)
        self.border_type = border_type
        self.cval = cval

    def args_to_subconfig(self, args):
        arg_cfgs = ()
        out_cfg = None
        for arg in args:
            arg_cfgs += (NdArrCfg(arg.dtype, arg.ndim, arg.shape), )
            if out_cfg is None:
                out_cfg = (NdArrCfg(arg.dtype, arg.ndim, arg.shape), )
        return arg_cfgs + out_cfg

    def process_arg_cfg(self, arg_cfg):
        arg_types = ()
        kernel_params = ()
        params = []
        for index, cfg in enumerate(arg_cfg):
            if self.backend in {'c', 'omp'}:
                arg_types += (np.ctypeslib.ndpointer(
                    cfg.dtype, cfg.ndim, cfg.shape), )
                params.append(
                    SymbolRef.unique(sym_type=arg_types[-1]())
                )
            else:
                arg_types += (cl.cl_mem, )
                params.append(
                    SymbolRef.unique(sym_type=arg_types[-1]())
                )
                kernel_params += (
                    SymbolRef(params[-1].name,
                              np.ctypeslib.ndpointer(
                                  cfg.dtype, cfg.ndim, cfg.shape)()), )
        return arg_types, params, kernel_params

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        arg_types, params, kernel_params = self.process_arg_cfg(arg_cfg)
        tree = StructuredGridFrontend().visit(tree)
        backend = StructuredGridBackend(arg_cfg, arg_types,
                                        params, kernel_params,
                                        self.border_type, self.cval)
        tree = backend.visit(tree)

        arg_types += (cl.cl_command_queue, )
        arg_types += tuple(cl.cl_kernel for _ in backend.kernels)
        entry_type = ct.CFUNCTYPE(*((None, ) + arg_types))
        fn = OclConcreteStructuredGrid('control', tree, entry_type)
        kernels = []
        for kernel in backend.kernels:
            program = cl.clCreateProgramWithSource(fn.context,
                                                   kernel.codegen()).build()
            kernels.append(program[kernel.body[0].name.name])
        return fn.finalize(kernels)


def structured_grid(border='zero', cval=0):
    def wrapper(fn):
        spec_fn = StructuredGrid(get_ast(fn), border, cval)

        def wrapped(*args):
            return spec_fn(*args)
        return wrapped
    return wrapper
