from ctree.frontend import get_ast
# import ast
import inspect

from hindemith.cfg import ControlFlowGraph
import sys


def hm(fn):
    tree = get_ast(fn)
    cfg = ControlFlowGraph(tree.body[0])
    symbol_table = {}
    frame = inspect.stack()[1][0]
    while frame is not None:
        symbol_table.update(frame.f_locals)
        symbol_table.update(frame.f_globals)
        frame = frame.f_back

    def wrapped(*args, **kwargs):
        env = symbol_table
        for index, arg in enumerate(tree.body[0].args.args):
            if sys.version_info < (3, 0):
                env[arg.id] = args[index]
            else:
                env[arg.arg] = args[index]
        if cfg.compiled is None:
            # Make mutable copy of base symbol table
            cfg.start_block = cfg.build_composable_blocks(env, cfg.start_block)
            print(cfg)
            cfg.perform_liveness_analysis()
            cfg.compiled = cfg.compile_composable_blocks(env, cfg.start_block)
        fn = cfg.compile_to_fn(env)
        return fn(*args, **kwargs)
    return wrapped
