from ctree.frontend import get_ast
from .basic_blocks import get_basic_block, separate_composable_blocks, \
    process_composable_blocks
from .liveness_analysis import perform_liveness_analysis
from .util import SymbolTable

import inspect
import sys
# from copy import deepcopy
import ast


def meta(func):
    """
    Decorator entry point for meta-specializers
    Example usage:
        @meta
        def func(a, b):
            c = array_add(a, b)
            d = array_sub(b, c)
            return array_mul(d, c)
    """
    original_ast = get_ast(func)
    orig_basic_block = get_basic_block(original_ast)
    func._hm_cache = {}

    def meta_specialized(*args, **kwargs):
        if hasattr(func, '_hm_cache'):
            # print("Cache hit")
            if args[0].shape in func._hm_cache:
                return func._hm_cache[args[0].shape](*args, **kwargs)
        # TODO: This should be done lazily as symbols are needed
        # could be problematic/slow with a large stack
        symbol_table = SymbolTable(dict(func.__globals__, **kwargs),
                                   inspect.stack()[1:])
        for index, arg in enumerate(args):
            name = original_ast.body[0].args.args[index]
            if sys.version_info >= (3, 0):
                symbol_table[name.arg] = arg
            else:
                symbol_table[name.id] = arg
        basic_block = separate_composable_blocks(orig_basic_block,
                                                 symbol_table)
        print(basic_block)
        basic_block = perform_liveness_analysis(basic_block)
        print(basic_block)
        basic_block = process_composable_blocks(basic_block, symbol_table)
        print(basic_block)
        fn = get_callable(basic_block, symbol_table)
        func._hm_cache[args[0].shape] = fn
        return fn(*args, **kwargs)

    return meta_specialized


def my_exec(func, env):
    """
    Special exec for handling Python 2 -> 3 syntax change
    """
    if sys.version_info >= (3, 0):
        exec(func, env)
    else:
        exec(func) in env, env


def get_callable(basic_block, env):
    """
    Takes in a BasicBlock and returns a Python callable corresponding to its
    body.
    """
    if sys.version_info >= (3, 0):
        tree = ast.Module(
            [ast.FunctionDef(basic_block.name, basic_block.params,
                             list(basic_block.body), [], None)]
        )
    else:
        tree = ast.Module(
            [ast.FunctionDef(basic_block.name, basic_block.params,
                             list(basic_block.body), [])]
        )
    ast.fix_missing_locations(tree)
    # TODO: We have to pass in the real env dict here, is this problematic?
    exec(compile(tree, filename="file", mode="exec"), env._env, env._env)
    return env[basic_block.name]
