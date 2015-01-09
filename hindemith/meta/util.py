__author__ = 'leonardtruong'

import ast


class SymbolTable(dict):
    """
    Wrapper around a Python Dictionary to support lazy lookup of
    symbols in the current frame.
    """
    def __init__(self, env, stack):
        super(SymbolTable, self).__init__()
        self._env = env
        self._stack = stack

    def __getitem__(self, symbol):
        try:
            return self._env[symbol]
        except KeyError:
            # Check stack for symbol
            for frame in self._stack:
                if symbol in frame[0].f_locals:
                    # Cache it
                    self._env[symbol] = frame[0].f_locals[symbol]
                    return self._env[symbol]
            # TODO: Make this a more meaningful exception
            raise KeyError(symbol)

    def __setitem__(self, symbol, item):
        self._env[symbol] = item

    def __contains__(self, symbol):
        try:
            self[symbol]
            return True
        except KeyError:
            return False


class UniqueNamer(ast.NodeTransformer):
    curr = -1

    def __init__(self):
        self.seen = {}

    def gen_tmp(self):
        UniqueNamer.curr += 1
        return "_f{}".format(UniqueNamer.curr)

    def visit_FunctionCall(self, node):
        node.args = [self.visit(arg) for arg in node.args]
        return node

    def visit_FunctionDecl(self, node):
        node.name = self.visit(node.name)
        node.params = [self.visit(p) for p in node.params]
        node.defn = [self.visit(s) for s in node.defn]
        return node

    def visit_SymbolRef(self, node):
        if node.name == 'NULL':
            return node
        if node.name not in self.seen:
            self.seen[node.name] = self.gen_tmp()
        node.name = self.seen[node.name]
        return node


def get_unique_func_name(env):
    cnt = 0
    name = "_merged_f0"
    while name in env:
        cnt += 1
        name = "_merged_f{}".format(cnt)
    return name


class EntryPointFinder(ast.NodeVisitor):
    def __init__(self, entry_name):
        self.entry_name = entry_name
        self.entry_point = None

    def visit_FunctionDecl(self, node):
        if node.name.name == self.entry_name:
            self.entry_point = node


def find_entry_point(entry_name, tree):
    finder = EntryPointFinder(entry_name)
    finder.visit(tree)
    if not finder.entry_point:
        raise Exception("Could not find entry point {}".format(entry_name))
    return finder.entry_point


class SymbolReplacer(ast.NodeTransformer):
    def __init__(self, old, new):
        self._old = old
        self._new = new

    def visit_SymbolRef(self, node):
        if node.name == self._old:
            node.name = self._new
        return node


class RemoveRedcl(ast.NodeTransformer):
    def __init__(self):
        super(RemoveRedcl, self).__init__()
        self._decled = set()

    def visit_For(self, node):
        return node

    def visit_SymbolRef(self, node):
        if node.type is not None:
            if node.name in self._decled:
                node.type = None
            else:
                self._decled.add(node.name)
        return node

        
