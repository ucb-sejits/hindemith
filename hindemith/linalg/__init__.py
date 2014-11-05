from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.c.nodes import FunctionDecl, SymbolRef, For, ArrayRef, Add, Assign, \
    Constant, AddAssign, Lt, Mul, Sub, Div, CFile
from ctree.nodes import Project
import ctree.np
ctree.np
import numpy as np
from collections import namedtuple
import ctypes as ct
from functools import reduce


NdArrCfg = namedtuple('NdArrCfg', ['dtype', 'ndim', 'shape'])
ScalarCfg = namedtuple('ScalarCfg', ['dtype'])


curr_u = 0


def next_loop_var():
    global curr_u
    curr_u += 1
    return "_l{}".format(curr_u)


def gen_loop_index(loop_vars, shape):
    base = SymbolRef(loop_vars[-1])
    for index, var in reversed(tuple(enumerate(loop_vars[:-1]))):
        curr = Mul(SymbolRef(var),
                   Constant(reduce(lambda x, y: x * y, shape[index + 1:], 1)))
        base = Add(curr, base)
    return Assign(SymbolRef('loop_idx', ct.c_int()), base)


def for_range(r, step, body):
    loop_vars = []
    curr_body = []
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
                Lt(SymbolRef(loop_vars[-1]), Constant(r[-1])),
                AddAssign(SymbolRef(loop_vars[-1]), step),
                next_body)
        )
        curr_body = next_body
    curr_body.append(gen_loop_index(loop_vars, r))
    curr_body.extend(body)
    return node


py_to_ctypes = {
    int: ct.c_int,
    float: ct.c_float
}

op_map = {
    '+': Add,
    '-': Sub,
    '*': Mul,
    '/': Div
}


class ConcreteEltOp(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        self._c_function = self._compile(entry_name, proj, entry_type)

    def __call__(self, *args):
        output = None
        for arg in args:
            if isinstance(arg, np.ndarray):
                output = np.zeros_like(arg)
                break
        self._c_function(args[0], args[1], output)
        return output


class EltWiseArrayOp(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        arg_cfgs = ()
        out_cfg = None
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg_cfgs += (NdArrCfg(arg.dtype, arg.ndim, arg.shape), )
                out_cfg = (NdArrCfg(arg.dtype, arg.ndim, arg.shape), )
            else:
                arg_cfgs += (ScalarCfg(type(arg)), )
        return arg_cfgs + out_cfg

    def transform(self, tree, program_cfg):
        op = op_map[tree]
        arg_cfg, tune_cfg = program_cfg
        arg_types = ()
        op_args = ()
        for index, cfg in enumerate(arg_cfg):
            if isinstance(cfg, NdArrCfg):
                arg_types += (np.ctypeslib.ndpointer(
                    cfg.dtype, cfg.ndim, cfg.shape), )
                if index < 2:
                    op_args += (ArrayRef(SymbolRef('arg{}'.format(index)),
                                         SymbolRef('loop_idx')), )
            else:
                arg_types += (py_to_ctypes[cfg.dtype], )
                if index < 2:
                    op_args += (SymbolRef('arg{}'.format(index)), )

        func = FunctionDecl(
            None,
            SymbolRef('op'),
            [SymbolRef('arg0', arg_types[0]()),
             SymbolRef('arg1', arg_types[1]()),
             SymbolRef('output', arg_types[2]()),
             ],
            [for_range(
                arg_cfg[2].shape, 1,
                [Assign(ArrayRef(SymbolRef('output'), SymbolRef('loop_idx')),
                        op(*op_args))])]
        )
        entry_type = ct.CFUNCTYPE(*((None,) + arg_types))
        return ConcreteEltOp('op', Project([CFile('op', [func])]), entry_type)


add = EltWiseArrayOp('+')
sub = EltWiseArrayOp('-')
mul = EltWiseArrayOp('*')
div = EltWiseArrayOp('/')
