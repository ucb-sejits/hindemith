from hindemith.meta.util import UniqueNamer, get_unique_func_name, \
    find_entry_point, SymbolReplacer
from ctree.c.nodes import Assign, SymbolRef, Add, CFile, FunctionDecl
import unittest


class TestUniqueNamer(unittest.TestCase):
    def test_simple(self):
        tree1 = Assign(SymbolRef('a'), Add(SymbolRef('b'), SymbolRef('c')))
        tree2 = Assign(SymbolRef('a'), Add(SymbolRef('b'), SymbolRef('c')))
        uniquifier = UniqueNamer()
        uniquifier.visit(tree1)
        uniquifier.visit(tree2)
        self.assertEqual(tree1.left.name, tree2.left.name)
        self.assertEqual(tree1.right.left.name, tree2.right.left.name)
        self.assertEqual(tree1.right.right.name, tree2.right.right.name)

    def test_different_trees(self):
        tree1 = Assign(SymbolRef('a'), Add(SymbolRef('b'), SymbolRef('c')))
        tree2 = Assign(SymbolRef('a'), Add(SymbolRef('b'), SymbolRef('c')))
        UniqueNamer().visit(tree1)
        UniqueNamer().visit(tree2)
        self.assertNotEqual(tree1.left.name, tree2.left.name)
        self.assertNotEqual(tree1.right.left.name, tree2.right.left.name)
        self.assertNotEqual(tree1.right.right.name, tree2.right.right.name)


class TestUniqueFuncName(unittest.TestCase):
    def test_simple(self):
        env = {'_merged_f0'}
        self.assertNotEqual('_merged_f0',
                            get_unique_func_name(env))
        self.assertEqual('_merged_f1',
                         get_unique_func_name(env))


class TestFindEntryPoint(unittest.TestCase):
    def test_simple(self):
        tree = CFile('test', [FunctionDecl(None, SymbolRef('entry'), [], [])])
        actual = tree.body[0]
        self.assertEqual(actual, find_entry_point('entry', tree))


class TestSymbolReplacer(unittest.TestCase):
    def test_simple(self):
        tree = Assign(SymbolRef('a'), Add(SymbolRef('a'), SymbolRef('b')))
        SymbolReplacer('a', 'd').visit(tree)
        self.assertEqual(tree.left.name, 'd')
        self.assertEqual(tree.right.left.name, 'd')
