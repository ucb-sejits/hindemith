__author__ = 'leonardtruong'

import ast
from ctree.ocl import get_context_and_queue_from_devices
from ctree.ocl.macros import get_local_id, get_local_size, get_group_id
from ctree.c.nodes import SymbolRef, Constant, Op, Assign, Add, For, \
    AddAssign, Lt, Mul, Sub, FunctionDecl, FunctionCall, ArrayDef, If, \
    And
from ctree.ocl.nodes import OclFile
import pycl as cl
import ctypes as ct
import numpy as np
from .util import get_unique_func_name, UniqueNamer, find_entry_point, \
    SymbolReplacer

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from functools import reduce
from ctree.ocl.macros import clSetKernelArg, NULL, get_global_id


def gen_ocl_loop_index(shape):
    base = get_global_id(0)
    for index in range(1, len(shape)):
        curr = Mul(get_global_id(index),
                   Constant(reduce(lambda x, y: x * y, shape[:index], 1)))
        base = Add(curr, base)
    return Assign(SymbolRef('loop_idx', ct.c_int()), base)


def gen_kernel_cond(global_size, shape, offset):
    conds = ()
    for index, g, s in zip(range(len(global_size)), global_size, shape):
        if s < g:
            conds += (Lt(get_global_id(index),
                         Constant(s + offset[index])), )
    if len(conds) == 0:
        return None
    cond = conds[0]
    for c in conds[1:]:
        cond = And(c, cond)
    return cond


def process_arg_types(params, kernel):
    control = []
    for index, param in enumerate(params):
        control.append(
            clSetKernelArg(kernel, index, ct.sizeof(cl.cl_mem), param.name))
    return control


def get_local_size(shape):
    """
    Generate local size from shape.  If the size is less than 32, set it to
    that else, set it to 32.

    TODO: This should be dynamic with respect to the maximum amount of compute
    units

    :param tuple shape: The shape of the array being iterated over
    """
    local_size = ()
    if len(shape) == 2:
        for dim in shape:
            if dim > 32:
                local_size += (32, )
            else:
                local_size += (dim,)
    else:
        local_size = (32, )
    return local_size


unique_kernel_num = -1


def unique_kernel_name():
    global unique_kernel_num
    unique_kernel_num += 1
    return "_kernel{}".format(unique_kernel_num)


def kernel_range(shape, kernel_range, params, body, offset=None):
    """
    Factory method for generating an OpenCL kernel corresponding
    to a set of nested for loops.  Returns the control logic for
    setting the arguments and launching the kernel as well as the
    kernel itself.

    TODO: Make local size computation dynamic
    """
    unique_name = unique_kernel_name()
    control = process_arg_types(params, unique_name)

    global_size = ()
    for d in kernel_range:
        if d % 32 != 0 and d > 32:
            global_size += ((d + 31) & (~31),)
        else:
            global_size += (d,)

    if offset is None:
        offset = [0 for _ in global_size]

    local_size = get_local_size(global_size)

    global_size_decl = 'global_size{}'.format(unique_name)
    local_size_decl = 'local_size{}'.format(unique_name)
    offset_decl = 'offset{}'.format(unique_name)
    control.extend([
        ArrayDef(SymbolRef(global_size_decl, ct.c_size_t()),
                 Constant(len(shape)), global_size),
        ArrayDef(SymbolRef(local_size_decl, ct.c_size_t()),
                 Constant(len(shape)), local_size),
        ArrayDef(SymbolRef(offset_decl, ct.c_size_t()),
                 Constant(len(offset)), offset),
        FunctionCall(
            SymbolRef('clEnqueueNDRangeKernel'), [
                SymbolRef('queue'), SymbolRef(unique_name),
                Constant(len(shape)), SymbolRef(offset_decl),
                SymbolRef(global_size_decl), SymbolRef(local_size_decl),
                Constant(0), NULL(), NULL()
            ]
        ),
        FunctionCall(SymbolRef('clFinish'), [SymbolRef('queue')])
    ])
    body.insert(0, gen_ocl_loop_index(shape))
    cond = gen_kernel_cond(global_size, kernel_range, offset)
    if cond:
        body = If(cond, body)
    kernel = FunctionDecl(
        None,
        SymbolRef(unique_name),
        params,
        body
    )
    for index, param in enumerate(params):
        if isinstance(param.type, np.ctypeslib._ndptr):
            kernel.params[index].set_global()
            if index < len(params) - 1:
                kernel.params[index].set_const()
    kernel.set_kernel()
    return control, OclFile(unique_name, [kernel])


class ConcreteMerged(ConcreteSpecializedFunction):
    def __init__(self):
        devices = cl.clGetDeviceIDs()
        # Default to last device for now
        # TODO: Allow settable devices via params or env variables
        self.context, self.queue = get_context_and_queue_from_devices(
            [devices[-1]])

    def finalize(self, proj, entry_name, entry_type, kernels, outputs,
                 retvals):
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


def fuse(sources, sinks, nodes, to_promote):
    seen = {}
    fused_body = []
    kernel_params = []
    param_types = []
    params = []
    for loop in nodes:
        body = loop.body
        for param, _type in zip(loop.sources, loop.types):
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
        else:
            output = loop.sinks[0]
            seen[sink] = output.name
            kernel_params.append(SymbolRef(output.name, loop.types[-1]()))
            params.append(SymbolRef(output.name, cl.cl_mem()))
            param_types.append(cl.cl_mem)
        fused_body.extend(body)
    control = FunctionDecl(None, SymbolRef('control'), params, [])
    control_body, kernel = kernel_range(nodes[0].shape, nodes[0].shape,
                                        kernel_params, fused_body)
    control.defn = control_body
    return control, kernel


def merge_entry_points(composable_block, env):
    fused_sources_set = []
    fused_sources_list = []
    fused_sinks_list = []
    fused_sinks_set = []
    fused_nodes = []
    for statement in composable_block.statements:
        arg_vals = tuple(env[source] for source in statement.sources)
        dependencies = set(fused_sinks_set).intersection(statement.sources)
        specializer = statement.specializer
        output_name = statement.sinks[0]
        env[output_name] = specializer.get_placeholder_output(arg_vals)
        # fused_sources_set |= set(statement.sources)
        for source in statement.sources:
            if source not in fused_sources_set:
                fused_sources_set.append(source)
        fused_sources_list.extend(statement.sources)
        fused_sinks_list.extend(statement.sinks)
        # fused_sinks_set |= set(statement.sinks)
        for sink in statement.sinks:
            if sink not in fused_sinks_set:
                fused_sinks_set.append(sink)
        fused_nodes.extend(specializer.get_ir_nodes(arg_vals))
    to_promote = []
    for src in fused_sources_set:
        if src not in composable_block.live_ins:
            to_promote.append(src)
    fused = fuse(fused_sources_list, fused_sinks_list, fused_nodes, to_promote)
    target_ids = composable_block.live_outs.intersection(composable_block.kill)
    raise NotImplementedError()
    # args = get_merged_arguments(composable_block)
    # merged_entry_type = []
    # entry_points = []
    # param_map = {}
    # files = []
    # merged_kernels = []
    # output_indexes = []
    # curr_fusable = None
    # retval_indexes = []
    # target_ids = composable_block.live_outs.intersection(composable_block.kill)
    # print(composable_block.live_outs)
    # print(composable_block.live_ins)
    # for statement in composable_block.statements:
    #     specializer = statement.specializer
    #     output_name = statement.sinks[0]
    #     arg_vals = tuple(env[source] for source in statement.sources)
    #     env[output_name] = specializer.get_placeholder_output(arg_vals)
    #     mergeable_info = specializer.get_mergeable_info(arg_vals)
    #     proj, entry_point, entry_type, kernels = mergeable_info.proj, \
    #         mergeable_info.entry_point, mergeable_info.entry_type, \
    #         mergeable_info.kernels
    #     files.extend(proj.files)
    #     for p in statement.sources:
    #         if p[:2] == '_t':
    #             print(p)
    #     uniquifier = UniqueNamer()
    #     proj = uniquifier.visit(proj)
    #     merged_kernels.extend(kernels)
    #     entry_point = find_entry_point(uniquifier.seen[entry_point], proj)
    #     param_map[output_name] = entry_point.params[-1].name
    #     entry_points.append(entry_point)
    #     entry_type = remove_seen_symbols(statement.sources, param_map,
    #                                      entry_point, entry_type)
    #     merged_entry_type.extend(entry_type[1:])
    #     if output_name in target_ids:
    #         retval_indexes.append(len(output_indexes))
    #     output_indexes.append(len(merged_entry_type) - 1)
    #     fusable_node = mergeable_info.fusable_node
    #     if fusable_node is not None:
    #         sources = [source for source in statement.sources]
    #         sinks = [sink for sink in statement.sinks]
    #         fusable_node.sources = sources
    #         fusable_node.sinks = sinks
    #         if curr_fusable is not None:
    #             fuse_nodes(curr_fusable, fusable_node)
    #         curr_fusable = fusable_node
    #
    # merged_entry_type.insert(0, None)
    # merged_entry = perform_merge(entry_points)
    #
    # if len(target_ids) > 1:
    #     targets = [ast.Tuple([ast.Name(id, ast.Store())
    #                          for id in target_ids], ast.Store())]
    # else:
    #     targets = [ast.Name(target_ids[0], ast.Store())]
    # merged_name = get_unique_func_name(env)
    # print(files[0])
    # print(files[-1])
    # env[merged_name] = MergedSpecializedFunction(
    #     Project(files), merged_entry.name.name, merged_entry_type,
    #     merged_kernels, output_indexes, retval_indexes
    # )
    # value = ast.Call(ast.Name(merged_name, ast.Load()), args, [], None, None)
    # return ast.Assign(targets, value)


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
