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
from .util import get_unique_func_name, SymbolReplacer, RemoveRedcl
from ..nodes import kernel_range, ocl_header
from hindemith.types.hmarray import hmarray, empty

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction


class OclConcreteMerged(ConcreteSpecializedFunction):
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
            while len(out_idxs) > 0 and index == out_idxs[0]:
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

        # cl.clFinish(self.queue)
        self._c_function(*([self.queue, self.kernel] + processed))
        if len(outputs) == 1:
            return outputs[0]
        return outputs


class MergedSpecializedFunction(LazySpecializedFunction):
    def __init__(self, tree, entry_type, output_idxs, retval_idxs):
        super(MergedSpecializedFunction, self).__init__(ast.Module())
        self.tree = tree
        self.entry_type = entry_type
        self.output_idxs = output_idxs
        self.retval_idxs = retval_idxs

    def transform(self, tree, program_config):
        tree = self.tree
        return tree.files

    def finalize(self, files, arg_cfg):
        tree = self.tree
        fn = OclConcreteMerged('control', tree, self.entry_type, self.output_idxs,
                            self.retval_idxs)
        kernel = tree.find(OclFile)
        program = cl.clCreateProgramWithSource(
            fn.context, kernel.codegen()).build()
        return fn.finalize(program[kernel.name])


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
    local_blocks = []
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
            body.insert(0,
                SymbolRef(seen[sink],
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
            if len(output_idxs) > 0 and output_idxs[-1] > len(args) - 1:
                output_idxs.append(output_idxs[-1] + 1)
            else:
                output_idxs.append(len(args) - 1)
        local_blocks.extend(loop.local_mem)
        fused_body.extend(body)
    control = FunctionDecl(None, SymbolRef('control'), params, [])
    control_body, kernel = kernel_range(nodes[0].shape, nodes[0].shape,
                                        kernel_params, fused_body, local_mem=local_blocks)
    print(kernel)
    params.insert(1, SymbolRef(kernel.body[0].name.name, cl.cl_kernel()))
    control.defn = control_body
    print(control)
    proj = Project([CFile('control', [ocl_header, control],
        config_target='opencl'), kernel])
    return proj, ct.CFUNCTYPE(*param_types), args, output_idxs, retval_idxs


def merge_entry_points(composable_block, env):
    fused_sources_set = []
    fused_sources_list = []
    fused_sinks_list = []
    fused_sinks_set = []
    fused_nodes = []
    remover = RemoveRedcl()
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
        nodes = specializer.get_ir_nodes(arg_vals)
        for node in nodes:
            node.body = list(map(remover.visit, node.body))
        fused_nodes.extend(nodes)
    to_promote = []
    to_promote = list(
        filter(lambda s: s not in
               composable_block.live_ins.union(composable_block.live_outs),
               fused_sources_set))
    proj, entry_type, args, output_idxs, retval_idxs = fuse(
        fused_sources_list, fused_sinks_list, fused_nodes, to_promote, env)
    target_ids = composable_block.live_outs.intersection(composable_block.kill)
    if len(target_ids) > 1:
        target_list = []
        for statement in composable_block.statements:
            for sink in statement.sinks:
                if sink in target_ids:
                    target_ids.discard(sink)
                    target_list.append(ast.Name(sink, ast.Store()))
        targets = [ast.Tuple(target_list, ast.Store())]
    else:
        targets = [ast.Name(list(target_ids)[0], ast.Store())]
    print(output_idxs)
    print(retval_idxs)
    merged_name = get_unique_func_name(env)
    env[merged_name] = MergedSpecializedFunction(proj, entry_type,
                                                 output_idxs, retval_idxs)
    value = ast.Call(ast.Name(merged_name, ast.Load()), args, [], None, None)
    return ast.Assign(targets, value)


class LoopDependence(object):
    def __init__(self, target, vector):
        self.target = target
        self.vector = vector
