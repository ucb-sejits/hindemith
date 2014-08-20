import numpy
import ast
import pycl as cl
import ctypes as ct

from ctree.frontend import get_ast
from ctree.jit import ConcreteSpecializedFunction, LazySpecializedFunction
from ctree.c.nodes import CFile, FunctionDecl, Op, FunctionCall, Constant, Add
from ctree.ocl.macros import barrier, CLK_LOCAL_MEM_FENCE
from ctree.ocl.nodes import OclFile
import ctree.np
ctree.np

from hindemith.utils import unique_kernel_name, unique_name

import inspect
import sys


import logging
LOG = logging.getLogger('Hindemith')

# from functools import reduce


def my_exec(f, symbol_table):
    if sys.version_info >= (3, 0):
        exec(f, symbol_table)
    else:
        exec(f) in symbol_table


def fuse(fn):
    """Decorator that will fuse specializers in the body of a function.  This
    optimizer will attempt to fuse specializer calls on various levels to
    improve runtime performance.  It will execute regular python code normally,
    and will output the same result as running the non-fused version of the
    function.

    :fn: A python function.
    :returns: `fused`, higher order python function that takes in the same
    parameters as `fn` and returns the same result(s) as `fn`.

    """
    def fused(*args, **kwargs):
        """
        Fused wrapper around `fn`.  First it get all variables defined in the
        local, global scope.  It then traverses the body of the function
        looking for places where specializer calls are made.  Any specializer
        calls found that can be fused will be (@todo: Tuning should occur here
        instead).

        :*args: Arguments to `fn`
        :**kwargs: Keyword arguments to `fn`.
        :returns: The same return value(s) as `fn`.

        """
        if hasattr(fn, '_hm_fused'):
            return fn._hm_fused(*args, **kwargs)
        tree = get_ast(fn)
        blocks = get_blocks(tree)

        # Build a symbol table with keyword arguments and the function's global
        # scope.
        symbol_table = dict(fn.__globals__, **kwargs)
        # Add all locally defined variables in the current stack
        for frame in inspect.stack()[1:]:
            symbol_table.update(frame[0].f_locals)

        # Add non keyword arguments
        for index, arg in enumerate(args):
            if sys.version_info >= (3, 0):
                symbol_table[tree.body[0].args.args[index]] = arg
            else:
                symbol_table[tree.body[0].args.args[index].id] = arg

        fuser = Fuser(blocks, symbol_table)
        fused_blocks = fuser.do_fusion()
        tree.body[0].body = fused_blocks

        # Remove Decorator
        tree.body[0].decorator_list = []
        tree = ast.fix_missing_locations(tree)
        my_exec(compile(tree, '', 'exec'), symbol_table)
        fn._hm_fused = symbol_table[fn.__name__]
        return symbol_table[fn.__name__](*args, **kwargs)
    return fused


def get_blocks(tree):
    """Convenience method for getting the blocks from an ast

    :param ast.Node tree: A Python AST.
    :returns: A list of AST nodes.
    :rtype: list
    """
    blocks = []
    BlockBuilder(blocks).visit(tree)
    return blocks


class BlockBuilder(ast.NodeVisitor):

    """Docstring for BlockBuilder. """

    def __init__(self, blocks):
        """@todo: to be defined1.

        :blocks: @todo

        """
        ast.NodeVisitor.__init__(self)

        self._blocks = blocks

    def visit_FunctionDef(self, node):
        """@todo: Docstring for visit_FunctionDef.

        :node: @todo
        :returns: @todo

        """
        self._blocks.extend(node.body)


class UniqueNamer(ast.NodeTransformer):
    uid = 0

    def __init__(self):
        ast.NodeTransformer.__init__(self)
        self.seen = {}

    def visit_FunctionCall(self, node):
        # Don't rename functions
        node.args = list(map(self.visit, node.args))
        return node

    def visit_SymbolRef(self, node):
        if node.name in {'float', 'NULL', 'cl_mem', 'CLK_LOCAL_MEM_FENCE'}:
            # Don't rename constants
            return node
        if node.name == '_f16':
            # Weird bug
            return node
        if node.name not in self.seen:
            UniqueNamer.uid += 1
            self.seen[node.name] = '_f%d' % UniqueNamer.uid
        node.name = self.seen[node.name]
        return node


class Fuser(object):

    """Docstring for Fuser. """

    def __init__(self, blocks, symbol_table):
        """@todo: to be defined1.

        :blocks: @todo
        :_symbol_table: @todo

        """
        self._blocks = blocks
        self._symbol_table = symbol_table
        self._defns = []

    def do_fusion(self):
        """@todo: Docstring for do_fusion.

        :blocks: @todo
        :returns: @todo

        """
        fused_blocks = [[self._blocks.pop(0)]]
        for block in self._blocks:
            if self._is_fusable(fused_blocks[-1][-1], block):
                fused_blocks[-1].append(block)
            else:
                fused_blocks.append([block])
        return list(map(self._fuse, fused_blocks))

    def _is_fusable(self, block_1, block_2):
        """@todo: Docstring for _is_fusable.

        :block_1: @todo
        :block_2: @todo
        :returns: @todo

        """
        if isinstance(block_1, ast.Assign) and \
                isinstance(block_2, ast.Assign) or \
                isinstance(block_2, ast.Return):
            if isinstance(block_1.value, ast.Call) and \
               isinstance(block_2.value, ast.Call):
                func_1 = self._symbol_table[block_1.value.func.id]
                func_2 = self._symbol_table[block_2.value.func.id]
                return isinstance(func_1, Fusable) and \
                    isinstance(func_2, Fusable)
        return False

    def _fuse(self, blocks):
        """@todo: Docstring for _fuse.

        :block_1: @todo
        :block_2: @todo
        :returns: @todo

        """
        if len(blocks) == 1:
            return blocks[0]

        projects, entry_types, entry_points, kernel_names, arg_list, \
            arg_nodes_list, fusable_nodes, outputs, is_return = \
            self.get_fusable_info(blocks)

        lazy = LazyFused(None)

        def transform(tree, program_cfg):

            project = fuse_at_project_level(projects, entry_points)
            fuse_fusables(fusable_nodes)
            print(project.files[0])
            print(project.files[1])

            return project

        def finalize(project, program_cfg):
            fn = FusedFn(outputs, is_return)
            ocl_file = project.find(OclFile)
            # print(ocl_file)
            kernel_ptrs = get_kernel_ptrs(ocl_file, fn)

            argtypes = [None, cl.cl_command_queue, cl.cl_kernel]

            for arg in project.files[0].body[-1].params:
                if isinstance(arg.type, ct.c_float):
                    argtypes.append(ct.c_float)
                elif isinstance(arg.type, cl.cl_mem):
                    argtypes.append(cl.cl_mem)
            # print(project.files[0])
            # print(project.files[1])
            return fn.finalize(
                entry_points[0], project, ct.CFUNCTYPE(*argtypes), kernel_ptrs
            )

        lazy.transform = transform
        lazy.finalize = finalize
        func_name = unique_kernel_name()
        self._symbol_table[func_name] = lazy
        tree = ast.Call(
            func=ast.Name(id=func_name, ctx=ast.Load()),
            args=arg_nodes_list, keywords=[]
        )
        if is_return:
            tree = ast.Return(tree)
        else:
            tree = ast.Expr(tree)
        return tree

    def get_fusable_info(self, blocks):
        projects = []
        entry_types = []
        entry_points = []
        kernel_names = []
        arg_list = []
        arg_nodes_list = []
        fusable_nodes = []
        outputs = []
        is_return = False
        for block in blocks:
            if isinstance(block, ast.Return):
                is_return = True
            specializer = self._symbol_table[block.value.func.id]
            arg_nodes_list.extend(block.value.args)
            args = tuple(
                self._symbol_table[arg.id] if isinstance(arg, ast.Name) else
                arg.n for arg in block.value.args
            )
            arg_list.extend(args)
            program_cfg = (specializer.args_to_subconfig(args), None)
            output = specializer.generate_output(program_cfg)
            arg_list.append(output)
            if not is_return:
                target = block.targets[0].id
            else:
                target = unique_name()

            self._symbol_table[target] = output
            arg_nodes_list.append(ast.Name(target, ast.Load()))

            outputs.append(output)
            tree, entry_type, entry_point = specializer.transform(
                specializer.original_tree,
                program_cfg
            )
            kernel_names.extend(kernel_names)
            projects.append(tree)
            entry_types.append(entry_type)
            entry_points.append(entry_point)
            fusable_nodes.extend(specializer.fusable_nodes)
            specializer.finalize(tree, entry_type, entry_point)
        return (projects, entry_types, entry_points, kernel_names, arg_list,
                arg_nodes_list, fusable_nodes, outputs, is_return)


class LazyFused(LazySpecializedFunction):
    pass


def fuse_at_project_level(projects, entry_points):
    """@todo: Docstring for fuse_at_project_level.

    :projects: @todo
    :returns: @todo

    """
    project = projects.pop(0)
    for proj in projects:
        project.files.extend(proj.files)
    project.files = fuse_at_file_level(project.files, entry_points)
    return project


def fuse_at_file_level(files, entry_points):
    """@todo: Docstring for fuse_at_file_level.

    :files: @todo
    :returns: @todo

    """
    # FIXME: Support all file types
    c_file = CFile('fused_c_file', [])
    ocl_file = OclFile('fuse_ocl_file', [])

    file_map = {
        CFile: c_file,
        OclFile: ocl_file
    }
    for _file in files:
        file_map[type(_file)].body.extend(_file.body)

    c_file = fuse_entry_points(c_file, entry_points)
    return [c_file, ocl_file]


def fuse_entry_points(c_file, entry_points):
    """@todo: Docstring for fuse_at_function_level.

    :c_file: @todo
    :ocl_file: @todo
    :returns: @todo

    """
    # Fuse all entry_points together
    func_decls = find_and_remove_entry_points(entry_points, c_file)
    fused_func = uniqueify_names(func_decls.pop(0))
    for func in func_decls:
        unique = uniqueify_names(func)
        fused_func.params.extend(unique.params)
        fused_func.defn.extend(unique.defn)
    c_file.body.append(fused_func)
    return c_file


def fuse_fusables(nodes):
    """@todo: Docstring for fuse_fusables.

    :nodes: @todo
    :returns: @todo

    """
    # FIXME: Assuming all KernelCalls
    kernel = nodes[0]
    kernel._global_size_decl.delete()
    kernel._local_size_decl.delete()
    offset = len(kernel._setargs)
    kernel._kernel.name = nodes[-1]._kernel.name
    kernel._enqueue_call.delete()
    kernel._finish_call.delete()
    if kernel._load_shared_memory_block is not None and \
            nodes[1]._load_shared_memory_block  is not None:
        import operator
        from functools import reduce
        import ctypes as ct
        # Calculate new local size
        # FIXME: Assumes going from 2 -> 4
        local_size = reduce(
            operator.mul,
            (item.value + 4 for item in kernel._local_size_decl.body),
            ct.sizeof(ct.c_float)
        )
        kernel._setargs[-1].args[2].value = local_size
        BlockSizeChanger().visit(kernel._load_shared_memory_block[1])
        BlockSizeChanger().visit(kernel._load_shared_memory_block[-1])
    to_remove = set()
    # FIXME: Assuming fusability
    for node in nodes[1:]:
        # kernel._control.defn.extend(node._control.defn)
        # del node._control.defn[:]
        to_remove.update(param.name for param in node._control.params[:2])
        for setter in node._setargs:
            setter.args[0].name = kernel._setargs[0].args[0].name
            setter.args[1].value += offset
        offset += len(node._setargs)
        unique = UniqueNamer().visit(node._kernel)
        kernel._kernel.params.extend(unique.params)
        kernel._kernel.defn.append(barrier(CLK_LOCAL_MEM_FENCE()))
        kernel._kernel.defn.extend(unique.defn)
        node._kernel.delete()
        if node is not nodes[-1]:
            node._global_size_decl.delete()
            node._local_size_decl.delete()
            node._enqueue_call.delete()
            node._finish_call.delete()
        else:
            node._finish_call.args[0].name = kernel._control.params[0].name
            node._enqueue_call.args[0].name = kernel._control.params[0].name
            node._enqueue_call.args[1].name = kernel._control.params[1].name

    remove_symbol_from_params(kernel._control.params, to_remove)


class BlockSizeChanger(ast.NodeTransformer):
    def visit_BinaryOp(self, node):
        if isinstance(node.op, Op.Add) and \
            isinstance(node.left, FunctionCall) and \
                node.left.func.name is 'get_local_size':
            return Add(node.left, Constant(node.right.value * 2))
        else:
            node.left = self.visit(node.left)
            node.right = self.visit(node.right)
        return node


def remove_symbol_from_params(params, names):
    new_params = []
    for param in params:
        if param.name not in names:
            new_params.append(param)
    params[:] = new_params


def get_kernel_ptrs(ocl_file, fn):
    program = cl.clCreateProgramWithSource(fn.context,
                                           ocl_file.codegen()).build()
    ptrs = []
    for statement in ocl_file.body:
        if isinstance(statement, FunctionDecl) and not statement.deleted:
            ptrs.append(program[statement.name])
    return ptrs


def uniqueify_names(function):
    return UniqueNamer().visit(function)


def find_and_remove_entry_points(entry_points, c_file):
    """@todo: Docstring for find_and_remove_entry_points.

    :entry_points: @todo
    :c_file: @todo
    :returns: @todo

    """
    results = []
    EntryPointFindAndRemover(entry_points, results).visit(c_file)
    return results


class EntryPointFindAndRemover(ast.NodeTransformer):

    """Docstring for EntryPointFindAndRemover. """

    def __init__(self, entry_points, results):
        """@todo: to be defined1.

        :entry_points: @todo
        :results: @todo

        """
        ast.NodeTransformer.__init__(self)

        self._entry_points = entry_points
        self._results = results

    def visit_FunctionDecl(self, node):
        """@todo: Docstring for visit_FunctionDecl.

        :node: @todo
        :returns: @todo

        """
        if node.name in self._entry_points:
            self._results.append(node)
        return []


class FusedFn(ConcreteSpecializedFunction):

    """Docstring for FusedFn. """

    def __init__(self, outputs, is_return):
        """@todo: to be defined1. """
        ConcreteSpecializedFunction.__init__(self)
        self.device = cl.clGetDeviceIDs()[-1]
        self.context = cl.clCreateContext([self.device])
        self.queue = cl.clCreateCommandQueue(self.context)
        self.orig_args = ()
        self.arg_buf_map = {}
        self.outputs = outputs
        self.is_return = is_return

    def finalize(self, entry_point_name, project, entry_point_typesig,
                 kernels):
        self._c_function = self._compile(entry_point_name, project,
                                         entry_point_typesig)
        self.kernels = kernels
        return self

    def _process_args(self, args):
        processed = ()
        for arg in args:
            if isinstance(arg, numpy.ndarray):
                if arg.ctypes.data in self.arg_buf_map:
                    processed += (self.arg_buf_map[arg.ctypes.data],)
                else:
                    buf, evt = cl.buffer_from_ndarray(self.queue, arg,
                                                      blocking=False)
                    evt.wait()
                    processed += (buf,)
                    self.arg_buf_map[arg.ctypes.data] = buf
            else:
                processed += (arg,)
        return processed

    def __call__(self, *args):
        processed = self._process_args(args)
        args = []
        args.append(self.queue)
        args.append(self.kernels[0])
        args.extend(processed)
        self._c_function(*args)
        return self._process_outputs()

    def _process_outputs(self):
        retvals = ()
        if self.is_return:
            output = self.outputs[-1]
            buf = self.arg_buf_map[output.ctypes.data]
            out, evt = cl.buffer_to_ndarray(self.queue, buf, output)
            evt.wait()
            return out
        for output in self.outputs:
            try:
                buf = self.arg_buf_map[output.ctypes.data]
                out, evt = cl.buffer_to_ndarray(self.queue, buf, output)
                evt.wait()
                retvals += (out,)
            except KeyError:
                # TODO: Make this a better exception
                raise Exception("Could not find corresponding buffer")
        return retvals
        # ret_vals = []
        # for output, out_like in zip(outputs, out_likes):
        #     if isinstance(output, cl.cl_mem):
        #         out, evt = cl.buffer_to_ndarray(self.queue, output,
        #                                         like=out_like)
        #         evt.wait()
        #         ret_vals.append(out)
        #     else:
        #         ret_vals.append(output.value)
        # return ret_vals


class Fusable(object):

    """Docstring for Fusable. """

    def __init__(self):
        """@todo: to be defined1. """
        self.fusable_nodes = []


class KernelCall(object):

    """Docstring for KernelCall. """

    def __init__(self, control, kernel, global_size, global_size_decl,
                 local_size, local_size_decl, enqueue_call, finish_call,
                 setargs, load_shared_memory_block=None):
        """@todo: to be defined1.

        :control: @todo
        :kernel: @todo
        :global_size: @todo
        :local_size: @todo

        """
        self._control = control
        self._kernel = kernel
        self._global_size = global_size
        self._global_size_decl = global_size_decl
        self._local_size = local_size
        self._local_size_decl = local_size_decl
        self._enqueue_call = enqueue_call
        self._finish_call = finish_call
        self._setargs = setargs
        self._load_shared_memory_block = load_shared_memory_block
