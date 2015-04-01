from ctree.frontend import get_ast
# import ast
import inspect

from hindemith.cfg import ControlFlowGraph


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
        # Make mutable copy of base symbol table
        env = dict(symbol_table)
        for index, arg in enumerate(tree.body[0].args.args):
            env[arg.id] = args[index]
        cfg.build_composable_blocks(env)
        print(cfg)
        cfg.compile_composable_blocks(env)
        print(cfg)
        cfg.compile_to_fn(env)
        fn = cfg.compile_to_fn(env)
        return fn(*args, **kwargs)
    return wrapped
