__author__ = 'leonardtruong'

import ast
from ctree.ocl import get_context_and_queue_from_devices
from ctree.ocl.macros import get_local_id, get_local_size, get_group_id
from ctree.c.nodes import SymbolRef, Constant, Op, Assign, Add, For, \
    AddAssign, Lt, Mul, Sub
import pycl as cl
import ctypes as ct
import numpy as np
from ctree.nodes import Project
from .util import get_unique_func_name, UniqueNamer, find_entry_point, \
    SymbolReplacer

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from functools import reduce


class ConcreteMerged(ConcreteSpecializedFunction):
    def __init__(self):
        devices = cl.clGetDeviceIDs()
        # Default to last device for now
        # TODO: Allow settable devices via params or env variables
        self.context, self.queue = get_context_and_queue_from_devices(
            [devices[-1]])

    def finalize(self, proj, entry_name, entry_type, kernels, outputs, retvals):
        self.__entry_type = entry_type
        self._c_function = self._compile(entry_name, proj,
                                         ct.CFUNCTYPE(*entry_type))
        self.__kernels = kernels
        self.__outputs = outputs
        self.__retvals = retvals
        return self

    def __call__(self, *args, **kwargs):
        processed = []
        events = []
        kernel_index = 0
        arg_index = 0
        outputs = []
        for index, argtype in enumerate(self.__entry_type[1:]):
            if argtype is cl.cl_command_queue:
                processed.append(self.queue)
            elif argtype is cl.cl_kernel:
                kernel = self.__kernels[kernel_index]
                program = cl.clCreateProgramWithSource(
                    self.context, kernel.codegen()).build()
                processed.append(program[kernel.body[0].name.name])
                kernel_index += 1
            elif index in self.__outputs:
                buf, evt = cl.buffer_from_ndarray(self.queue,
                                                  np.zeros_like(args[0]),
                                                  blocking=False)
                processed.append(buf)
                outputs.append(buf)
                events.append(evt)
            elif isinstance(args[arg_index], np.ndarray) and \
                    argtype is cl.cl_mem:
                buf, evt = cl.buffer_from_ndarray(self.queue, args[arg_index],
                                                  blocking=False)
                processed.append(buf)
                events.append(evt)
                arg_index += 1
        cl.clWaitForEvents(*events)
        self._c_function(*processed)

        retvals = ()
        for index in self.__retvals:
            buf, evt = cl.buffer_to_ndarray(self.queue, outputs[index],
                                            like=args[0], blocking=True)
            evt.wait()
            retvals += (buf, )
        if len(retvals) > 1:
            return retvals
        return retvals[0]


class MergedSpecializedFunction(LazySpecializedFunction):
    def __init__(self, tree, entry_name, entry_type, kernels, output_indexes,
                 retval_indexes):
        super(MergedSpecializedFunction, self).__init__(None)
        self.__original_tree = tree
        self.__entry_name = entry_name
        self.__entry_type = entry_type
        self.__kernels = kernels
        self.__output_indexes = output_indexes
        self.__retval_indexes = retval_indexes

    def transform(self, tree, program_config):
        fn = ConcreteMerged()
        return fn.finalize(self.__original_tree, self.__entry_name,
                           self.__entry_type, self.__kernels,
                           self.__output_indexes, self.__retval_indexes)


def replace_symbol_in_tree(tree, old, new):
    replacer = SymbolReplacer(old, new)
    for statement in tree.defn:
        replacer.visit(statement)
    return tree


def perform_merge(entry_points):
    merged_entry = entry_points.pop(0)
    for point in entry_points:
        merged_entry.params.extend(point.params)
        merged_entry.defn.extend(point.defn)
        point.delete()
    return merged_entry


def remove_seen_symbols(args, param_map, entry_point, entry_type):
    to_remove_symbols = set()
    to_remove_types = set()
    for index, arg in enumerate(args):
        if arg in param_map:
            param = entry_point.params[index + 2].name
            to_remove_symbols.add(param)
            to_remove_types.add(index + 3)
            replace_symbol_in_tree(entry_point, param, param_map[arg])
        else:
            param_map[arg] = entry_point.params[index + 2].name
    entry_point.params = [p for p in entry_point.params
                          if p.name not in to_remove_symbols]
    return [type for index, type in enumerate(entry_type)
            if index not in to_remove_types]


def get_merged_arguments(block):
    args = []
    seen_args = set()
    for statement in block.statements:
        for source in statement.sources:
            if source in block.live_ins and \
               source not in seen_args:
                seen_args.add(source)
                args.append(ast.Name(source, ast.Load()))
    return args


def fusable(node_1, node_2):
    if len(node_1.local_size) != len(node_2.local_size) or \
       len(node_1.global_size) != len(node_2.global_size):
        return False
    for i in range(len(node_1.global_size)):
        if node_1.local_size[i] != node_2.local_size[i] or \
           node_1.global_size[i] != node_2.global_size[i]:
            return False
    return True


def get_dependencies(node_2):
    for dependence in node_2.loop_dependencies:
        for dim in dependence.vector:
            if dim != 0:
                return False


class LocalPromoter(ast.NodeTransformer):
    def __init__(self, target):
        self.target = target
        self.promote = False

    def visit_BinaryOp(self, node):
        if isinstance(node.op, Op.ArrayRef) and node.left.name == self.target:
            self.promote = True
            node.right = self.visit(node.right)
            self.promote = False
        else:
            node.left = self.visit(node.left)
            node.right = self.visit(node.right)
        return node

    def visit_FunctionCall(self, node):
        if self.promote:
            if node.func.name == 'get_global_id':
                return SymbolRef('local_id')
            elif node.func.name == 'clamp':
                return self.visit(node.args[0].value)
        else:
            node.args = [self.visit(arg) for arg in node.args]
        return node


def promote_to_local_block(old, new, body):
    SymbolReplacer(old, new).visit(body)
    LocalPromoter(new).visit(body)


class GlobalPromoter(ast.NodeTransformer):
    def visit_FunctionCall(self, node):
        if node.func.name == 'get_global_id':
            return SymbolRef('global_id')
        node.args = [self.visit(arg) for arg in node.args]
        return node


def redefine(body):
    promoter = GlobalPromoter()
    body = [promoter.visit(line) for line in body]
    body.insert(0, Assign(SymbolRef('global_id', ct.c_int()),
                          Add(Sub(SymbolRef('local_id'), Constant(1)),
                              Mul(get_local_size(0),
                                  get_group_id(0)))))
    return body


def fuse_nodes(prev, next):
    """TODO: Docstring for fuse_nodes.

    :prev: TODO
    :next: TODO
    :returns: TODO

    """
    if fusable(prev, next):
        arg_symbols = []
        arg_symbols.append(
            [setter.args[3].arg.name for setter in prev.arg_setters])
        arg_symbols.append(
            [setter.args[3].arg.name for setter in next.arg_setters])
        new_kernel = next.arg_setters[0].args[0]
        next.kernel_decl.defn = prev.kernel_decl.defn + next.kernel_decl.defn
        for dependence in next.loop_dependencies:
            if dependence.target >= len(next.sources):
                break
            if next.sources[dependence.target] in prev.sinks and \
               max(map(abs, dependence.vector)) > 0:
                promote_to_local_block(
                    next.kernel_decl.params[dependence.target].name,
                    prev.kernel_decl.params[-1].name, next.kernel_decl)
                next.kernel_decl.params[dependence.target]._local = True
                next.kernel_decl.params[dependence.target]._global = False
                prev.arg_setters[-1].delete()
                next.arg_setters[dependence.target].args[3] = SymbolRef('NULL')
                size = (next.local_size[i] + abs(dependence.vector[i]) * 2
                        for i in range(len(dependence.vector)))
                local_size = reduce(lambda x, y: x * y, size)
                next.arg_setters[dependence.target].args[2] = Constant(
                    local_size * ct.sizeof(cl.cl_float))
                prev_body = next.kernel_decl.defn[:len(prev.kernel_decl.defn)]
                del next.kernel_decl.defn[:len(prev.kernel_decl.defn)]
                prev_body = redefine(prev_body)
                next.kernel_decl.defn.insert(
                    0, For(Assign(SymbolRef('local_id', ct.c_int()),
                                  SymbolRef('get_local_id(0)')),
                           Lt(SymbolRef('local_id'), Constant(local_size)),
                           AddAssign(SymbolRef('local_id'), Constant(32)),
                           prev_body
                       ))
                next.kernel_decl.defn.insert(
                    1, SymbolRef('barrier(CLK_LOCAL_MEM_FENCE)'))
                next.kernel_decl.defn.insert(
                    2, Assign(SymbolRef('local_id', ct.c_int()),
                              Add(get_local_id(0), Constant(1))))
                prev.kernel_decl.params.pop()
                break
        incr = 0
        for setter in prev.arg_setters:
            if not setter.deleted:
                incr += 1
        for setter in next.arg_setters:
            setter.args[1].value += incr
        next.arg_setters = prev.arg_setters + next.arg_setters
        for setter in prev.arg_setters:
            setter.args[0] = new_kernel
        next.kernel_decl.params = prev.kernel_decl.params + \
            next.kernel_decl.params
        print(next.kernel_decl)
        prev.kernel_decl.defn = [SymbolRef('return')]
        prev.enqueue_call.delete()


def merge_entry_points(composable_block, env):
    """
    A hideosly complex function that needs to be cleaned up and modularized
    Proceed at your own risk.
    """
    args = get_merged_arguments(composable_block)
    merged_entry_type = []
    entry_points = []
    param_map = {}
    files = []
    merged_kernels = []
    output_indexes = []
    curr_fusable = None
    retval_indexes = []
    target_ids = composable_block.live_outs.intersection(composable_block.kill)
    for statement in composable_block.statements:
        specializer = statement.specializer
        output_name = statement.sinks[0]
        arg_vals = tuple(env[source] for source in statement.sources)
        env[output_name] = specializer.get_placeholder_output(arg_vals)
        mergeable_info = specializer.get_mergeable_info(arg_vals)
        proj, entry_point, entry_type, kernels = mergeable_info.proj, \
            mergeable_info.entry_point, mergeable_info.entry_type, \
            mergeable_info.kernels
        files.extend(proj.files)
        uniquifier = UniqueNamer()
        uniquifier.visit(proj)
        merged_kernels.extend(kernels)
        print(proj.files[0])
        entry_point = find_entry_point(uniquifier.seen[entry_point], proj)
        param_map[output_name] = entry_point.params[-1].name
        entry_points.append(entry_point)
        entry_type = remove_seen_symbols(statement.sources, param_map,
                                         entry_point, entry_type)
        merged_entry_type.extend(entry_type[1:])
        if output_name in target_ids:
            retval_indexes.append(len(output_indexes))
        output_indexes.append(len(merged_entry_type) - 1)
        fusable_node = mergeable_info.fusable_node
        if fusable_node is not None:
            sources = [source for source in statement.sources]
            sinks = [sink for sink in statement.sinks]
            fusable_node.sources = sources
            fusable_node.sinks = sinks
            if curr_fusable is not None:
                fuse_nodes(curr_fusable, fusable_node)
            curr_fusable = fusable_node

    merged_entry_type.insert(0, None)
    merged_entry = perform_merge(entry_points)

    targets = [ast.Name(id, ast.Store()) for id in target_ids]
    merged_name = get_unique_func_name(env)
    env[merged_name] = MergedSpecializedFunction(
        Project(files), merged_entry.name.name, merged_entry_type,
        merged_kernels, output_indexes, retval_indexes
    )
    print(merged_entry)
    # print(files[2])
    value = ast.Call(ast.Name(merged_name, ast.Load()), args, [], None, None)
    return ast.Assign(targets, value)


class MergeableInfo(object):
    def __init__(self, proj=None, entry_point=None, entry_type=None,
                 kernels=None, fusable_node=None):
        self.proj = proj
        self.entry_point = entry_point
        self.entry_type = entry_type
        self.kernels = kernels
        self.fusable_node = fusable_node


class FusableNode(object):
    def __init__(self):
        self.sources = []
        self.sinks = []


class FusableKernel(FusableNode):
    def __init__(self, local_size, global_size, arg_setters, enqueue_call,
                 kernel_decl, global_loads, global_stores,
                 loop_dependencies):
        """

        :param ctree.c.nodes.Assign local_size :
        :param ctree.c.nodes.Assign global_size :
        :param list[ctree.c.nodes.FunctionCall] arg_setters:
        :param ctree.c.nodes.FunctionCall enqueue_call:
        :param ctree.c.nodes.FunctionDecl kernel_decl:
        :param list[SymbolRef] global_loads:
        :param list[ctree.c.nodes.Assign] global_stores:
        :param list[LoopDependenceVector] loop_dependencies:
        """
        super(FusableKernel, self).__init__()
        self.local_size = local_size
        self.global_size = global_size
        self.arg_setters = arg_setters
        self.enqueue_call = enqueue_call
        self.kernel_decl = kernel_decl
        self.global_loads = global_loads
        self.global_stores = global_stores
        self.loop_dependencies = loop_dependencies


class LoopDependence(object):
    def __init__(self, target, vector):
        self.target = target
        self.vector = vector
