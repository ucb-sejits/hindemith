__author__ = 'leonardtruong'

import ast
from ctree.ocl import get_context_and_queue_from_devices
from ctree.ocl.macros import get_local_id, get_local_size, get_group_id
from ctree.c.nodes import SymbolRef, Constant, Op, Assign, Add, For, \
    AddAssign, Lt, Mul, Sub, FunctionDecl, FunctionCall, ArrayDef, If, \
    And, CFile
from ctree.ocl.nodes import OclFile
from ctree.nodes import Project
import pycl as cl
import ctypes as ct
import numpy as np
from .util import get_unique_func_name, SymbolReplacer
from ..nodes import kernel_range, ocl_header
from hindemith.types.hmarray import hmarray, empty

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction


class ConcreteMerged(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type, output_idxs,
                 retval_idxs):
        devices = cl.clGetDeviceIDs()
        # Default to last device for now
        # TODO: Allow settable devices via params or env variables
        self.context, self.queue = get_context_and_queue_from_devices(
            [devices[-1]])
        self._c_function = self._compile(entry_name, proj, entry_type)
        self.output_idxs = output_idxs
        self.retval_idxs = retval_idxs

    def finalize(self, kernel):
        self.kernel = kernel
        return self

    def __call__(self, *args):
        processed = []
        outputs = []
        out_idxs = self.output_idxs[:]
        for index, arg in enumerate(args):
            if isinstance(arg, hmarray):
                processed.append(arg.ocl_buf)
            if index in self.retval_idxs:
                outputs.append(arg)
            if len(out_idxs) > 0 and index == out_idxs[0]:
                out_idxs.pop(0)
                output = empty(arg.shape, arg.dtype)
                output._host_dirty = True
                processed.append(output.ocl_buf)
                outputs.append(output)

        for idx in out_idxs:
            output = empty(arg.shape, arg.dtype)
            processed.append(output.ocl_buf)
            output._host_dirty = True
            outputs.append(output)


        self._c_function(*([self.queue, self.kernel] + processed))
        if len(outputs) == 1:
            return outputs[0]
        return outputs


class MergedSpecializedFunction(LazySpecializedFunction):
    def __init__(self, tree, entry_type, output_idxs, retval_idxs):
        super(MergedSpecializedFunction, self).__init__(None)
        self.tree = tree
        self.entry_type = entry_type
        self.output_idxs = output_idxs
        self.retval_idxs = retval_idxs

    def transform(self, tree, program_config):
        tree = self.tree
        fn = ConcreteMerged('control', tree, self.entry_type, self.output_idxs,
                            self.retval_idxs)
        kernel = tree.find(OclFile)
        program = cl.clCreateProgramWithSource(
            fn.context, kernel.codegen()).build()
        return fn.finalize(program[kernel.body[0].name.name])


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
        prev.kernel_decl.defn = [SymbolRef('return')]
        prev.enqueue_call.delete()


class PromoteToRegister(ast.NodeTransformer):
    def __init__(self, name):
        super(PromoteToRegister, self).__init__()
        self.name = name

    def visit_BinaryOp(self, node):
        if isinstance(node.op, Op.ArrayRef):
            if node.left.name == self.name:
                return SymbolRef(node.left.name)
        return super(PromoteToRegister, self).generic_visit(node)


def fuse(sources, sinks, nodes, to_promote, env):
    seen = {}
    fused_body = []
    kernel_params = []
    param_types = [None, cl.cl_command_queue, cl.cl_kernel]
    params = [SymbolRef('queue', cl.cl_command_queue())]
    args = []
    output_idxs = []
    retval_idxs = []
    for loop in nodes:
        body = loop.body
        for param, _type in zip(loop.sources, loop.types):
            source = sources.pop(0)
            # skip constants
            # FIXME: More elegant way to handle this case
            while type(env[source]) in {int, float}:
                source = sources.pop(0)
            if source in seen:
                visitor = SymbolReplacer(param.name, seen[source])
                body = [visitor.visit(s) for s in body]
                if source in to_promote:
                    visitor = PromoteToRegister(seen[source])
                    body = [visitor.visit(s) for s in body]
            else:
                seen[source] = param.name
                param_types.append(cl.cl_mem)
                params.append(SymbolRef(param.name, cl.cl_mem()))
                kernel_params.append(SymbolRef(param.name, _type()))
                args.append(ast.Name(source, ast.Load()))

        # FIXME: Assuming fusability
        # FIXME: Assuming one sink per node
        sink = sinks.pop(0)
        if sink in to_promote:
            seen[sink] = loop.sinks[0].name
            body.insert(0, SymbolRef(seen[sink],
                                     loop.types[-1]._dtype_.type()))
            visitor = SymbolReplacer(loop.sinks[0].name, seen[sink])
            body = [visitor.visit(s) for s in body]
            visitor = PromoteToRegister(loop.sinks[0].name)
            body = [visitor.visit(s) for s in body]
        elif sink in seen:
            visitor = SymbolReplacer(loop.sinks[0].name, seen[sink])
            body = [visitor.visit(s) for s in body]
            for index, param in enumerate(params):
                if param.name == seen[sink]:
                    retval_idxs.append(index - 1)
        else:
            output = loop.sinks[0]
            seen[sink] = output.name
            kernel_params.append(SymbolRef(output.name, loop.types[-1]()))
            params.append(SymbolRef(output.name, cl.cl_mem()))
            param_types.append(cl.cl_mem)
            # args.append(ast.Name(sink, ast.Load()))
            if len(output_idxs) > 0 and output_idxs[-1] >= len(args) - 1:
                output_idxs.append(output_idxs[-1] + 1)
            else:
                output_idxs.append(len(args) - 1)
        fused_body.extend(body)
    control = FunctionDecl(None, SymbolRef('control'), params, [])
    control_body, kernel = kernel_range(nodes[0].shape, nodes[0].shape,
                                        kernel_params, fused_body)
    # print(kernel)
    params.insert(1, SymbolRef(kernel.body[0].name.name, cl.cl_kernel()))
    control.defn = control_body
    proj = Project([CFile('control', [ocl_header, control]), kernel])
    return proj, ct.CFUNCTYPE(*param_types), args, output_idxs, retval_idxs


def merge_entry_points(composable_block, env):
    fused_sources_set = []
    fused_sources_list = []
    fused_sinks_list = []
    fused_sinks_set = []
    fused_nodes = []
    for statement in composable_block.statements:
        arg_vals = tuple(env[source] for source in statement.sources)
        # dependencies = set(fused_sinks_set).intersection(statement.sources)
        specializer = statement.specializer
        # fused_sources_set |= set(statement.sources)
        fused_sources_set.extend(
            filter(lambda s: s not in fused_sources_set, statement.sources))
        fused_sources_list.extend(statement.sources)
        fused_sinks_list.extend(statement.sinks)
        # fused_sinks_set |= set(statement.sinks)
        fused_sinks_set.extend(
            filter(lambda s: s not in fused_sinks_set, statement.sinks))
        fused_nodes.extend(specializer.get_ir_nodes(arg_vals))
    to_promote = []
    to_promote = list(
        filter(lambda s: s not in
               composable_block.live_ins.union(composable_block.live_outs),
               fused_sources_set))
    proj, entry_type, args, output_idxs, retval_idxs = fuse(
        fused_sources_list, fused_sinks_list, fused_nodes, to_promote, env)
    target_ids = composable_block.live_outs.intersection(composable_block.kill)
    if len(target_ids) > 1:
        targets = [ast.Tuple([ast.Name(id, ast.Store())
                             for id in target_ids], ast.Store())]
    else:
        targets = [ast.Name(list(target_ids)[0], ast.Store())]
    merged_name = get_unique_func_name(env)
    env[merged_name] = MergedSpecializedFunction(proj, entry_type,
                                                 output_idxs, retval_idxs)
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
