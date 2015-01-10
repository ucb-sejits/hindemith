from ctree.c.nodes import SymbolRef, Constant, Assign, Add, \
    Lt, Mul, FunctionDecl, FunctionCall, ArrayDef, If, \
    And, For, AddAssign
from functools import reduce
from ctree.ocl.macros import clSetKernelArg, NULL, get_global_id
import pycl as cl
import ctypes as ct
from ctree.ocl.nodes import OclFile
import numpy as np
from ctree.templates.nodes import StringTemplate


def gen_ocl_loop_index(shape):
    base = get_global_id(len(shape) - 1)
    for index in reversed(range(len(shape) - 1)):
        curr = Mul(get_global_id(index),
                   Constant(reduce(lambda x, y: x * y, shape[index + 1:], 1)))
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


def process_arg_types(params, local_mem, kernel):
    control = []
    for index, param in enumerate(params):
        control.append(
            clSetKernelArg(kernel, index, ct.sizeof(cl.cl_mem), param.name))
    if local_mem is not None:
        for i, param in enumerate(local_mem):
            control.append(
                clSetKernelArg(kernel, index + 1 + i, 34 * 34 * 4, NULL()))

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

ocl_header = StringTemplate("""
                #ifdef __APPLE__
                #include <OpenCL/opencl.h>
                #else
                #include <CL/cl.h>
                #endif
                """)


def kernel_range(shape, kernel_range, params, body, offset=None, local_mem=None):
    """
    Factory method for generating an OpenCL kernel corresponding
    to a set of nested for loops.  Returns the control logic for
    setting the arguments and launching the kernel as well as the
    kernel itself.

    TODO: Make local size computation dynamic
    """
    unique_name = unique_kernel_name()
    control = process_arg_types(params, local_mem, unique_name)

    global_size = ()
    # for d in kernel_range:
    d = kernel_range[0]
    if d % 32 != 0 and d > 32:
        global_size += ((d + 31) & (~31),)
    else:
        global_size += (d,)
    global_size += tuple(kernel_range[1:])

    if offset is None:
        offset = [0 for _ in global_size]

    local_size = get_local_size(global_size)
    # print(global_size)
    # print(local_size)

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
                SymbolRef(global_size_decl), NULL(),
                Constant(0), NULL(), NULL()
            ]
        ),
        FunctionCall(SymbolRef('clFinish'), [SymbolRef('queue')])
    ])
    body.insert(0, gen_ocl_loop_index(shape))
    cond = gen_kernel_cond(global_size, kernel_range, offset)
    if cond:
        body = If(cond, body)
    args = params
    if local_mem is not None:
        args += local_mem
    kernel = FunctionDecl(
        None,
        SymbolRef(unique_name),
        args,
        body
    )
    for index, param in enumerate(params):
        if isinstance(param.type, np.ctypeslib._ndptr):
            kernel.params[index].set_global()
            # if index < len(params) - 1:
            #     kernel.params[index].set_const()
    kernel.set_kernel()
    return control, OclFile(unique_name, [kernel])



class LoopVarGenerator():
    def __init__(self):
        self.curr = 0

    def __call__(self):
        self.curr += 1
        return "x{}".format(self.curr)

next_loop_var = LoopVarGenerator()


def gen_loop_index(loop_vars, shape):
    base = SymbolRef(loop_vars[-1])
    for index, var in reversed(list(enumerate(loop_vars[:-1]))):
        curr = Mul(SymbolRef(var),
                   Constant(reduce(lambda x, y: x * y, shape[:index + 1], 1)))
        base = Add(curr, base)
    return Assign(SymbolRef('loop_idx', ct.c_int()), base)


def for_range(r, step, body):
    loop_vars = []
    curr_body = []
    next_loop_var = LoopVarGenerator()
    loop_vars.append(next_loop_var())
    node = For(Assign(SymbolRef(loop_vars[-1], ct.c_int()), Constant(0)),
               Lt(SymbolRef(loop_vars[-1]), Constant(r[-1])),
               AddAssign(SymbolRef(loop_vars[-1]), step),
               curr_body)
    for dim in r[:-1]:
        next_body = []
        loop_vars.append(next_loop_var())
        curr_body.append(
            For(Assign(SymbolRef(loop_vars[-1], ct.c_int()), Constant(0)),
                Lt(SymbolRef(loop_vars[-1]), Constant(dim)),
                AddAssign(SymbolRef(loop_vars[-1]), step),
                next_body)
        )
        curr_body = next_body
    curr_body.append(gen_loop_index(loop_vars, r))
    curr_body.extend(body)
    return node
