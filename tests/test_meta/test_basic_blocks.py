import unittest
from ctree.frontend import get_ast

from hindemith.meta.basic_blocks import get_basic_block, \
    separate_composable_blocks, NonComposableBlock, ComposableBlock

import ast

from ctree.jit import LazySpecializedFunction


@unittest.skip("Unique variable generation changes break print test")
class TestBasicBlockBuilder(unittest.TestCase):
    def _check_args(self, actual, expected):
        for act, exp in zip(actual, expected):
            if isinstance(act, ast.Name):
                self.assertEqual(act.id, exp)
            elif isinstance(act, ast.Num):
                self.assertEqual(act.n, exp)

    def test_simple_return(self):
        def func(a, b):
            return a + b

        tree = get_ast(func)
        basic_block = get_basic_block(tree)
        self.assertEqual(len(basic_block), 2)
        self.assertEqual(basic_block[0].targets[0].id, '_t0')
        self.assertEqual(
            basic_block[0].value.func.value.id, 'a')
        self.assertEqual(
            basic_block[0].value.func.attr, '__add__')
        self._check_args(basic_block[0].value.args, ['b'])
        self.assertIsInstance(basic_block[1], ast.Return)
        self.assertEqual(basic_block[1].value.id, '_t0')

    def test_simple_body(self):
        def func(a, b):
            c = a * b
            return c * 3

        tree = get_ast(func)
        basic_block = get_basic_block(tree)
        self.assertEqual(len(basic_block), 3)
        self.assertEqual(basic_block[0].targets[0].id, 'c')
        self.assertEqual(
            basic_block[0].value.func.value.id, 'a')
        self.assertEqual(
            basic_block[0].value.func.attr, '__mul__')
        self._check_args(basic_block[0].value.args, ['b'])
        self.assertEqual(basic_block[1].targets[0].id, '_t0')
        self.assertEqual(
            basic_block[1].value.func.value.id, 'c')
        self.assertEqual(
            basic_block[1].value.func.attr, '__mul__')
        self._check_args(basic_block[1].value.args, [3])
        self.assertIsInstance(basic_block[2], ast.Return)
        self.assertEqual(basic_block[2].value.id, '_t0')

    def test_unpack_expression(self):
        def func(a, b, c):
            return a * b + c

        tree = get_ast(func)
        basic_block = get_basic_block(tree)
        print(basic_block)
        self.assertEqual(len(basic_block), 3)
        self.assertEqual(basic_block[0].targets[0].id, '_t1')
        self.assertEqual(
            basic_block[0].value.func.value.id, 'a')
        self.assertEqual(
            basic_block[0].value.func.attr, '__mul__')
        self._check_args(basic_block[0].value.args, ['b'])
        self.assertEqual(basic_block[1].targets[0].id, '_t0')
        self.assertEqual(
            basic_block[1].value.func.value.id, '_t1')
        self.assertEqual(
            basic_block[1].value.func.attr, '__add__')
        self._check_args(basic_block[1].value.args, ['c'])
        self.assertIsInstance(basic_block[2], ast.Return)
        self.assertEqual(basic_block[2].value.id, '_t0')

    def test_unpack_precedence(self):
        def func(a, b, c):
            return a + b * c

        tree = get_ast(func)
        basic_block = get_basic_block(tree)
        print(basic_block)
        self.assertEqual(len(basic_block), 3)
        self.assertEqual(basic_block[0].targets[0].id, '_t1')
        self.assertEqual(
            basic_block[0].value.func.value.id, 'b')
        self.assertEqual(
            basic_block[0].value.func.attr, '__mul__')
        self._check_args(basic_block[0].value.args, ['c'])
        self.assertEqual(basic_block[1].targets[0].id, '_t0')
        self.assertEqual(
            basic_block[1].value.func.value.id, 'a')
        self.assertEqual(
            basic_block[1].value.func.attr, '__add__')
        self._check_args(basic_block[1].value.args, ['_t1'])
        self.assertIsInstance(basic_block[2], ast.Return)
        self.assertEqual(basic_block[2].value.id, '_t0')

    def test_simple_function_call(self):
        def z(a):
            return a

        def func(a):
            return z(a)

        tree = get_ast(func)
        basic_block = get_basic_block(tree)
        print(basic_block)
        self.assertEqual(len(basic_block), 2)
        self.assertEqual(basic_block[0].targets[0].id, '_t0')
        self.assertIsInstance(basic_block[0].value, ast.Call)
        self.assertEqual(basic_block[0].value.func.id, 'z')
        self.assertEqual(basic_block[0].value.args[0].id, 'a')
        self.assertIsInstance(basic_block[1], ast.Return)
        self.assertEqual(basic_block[1].value.id, '_t0')

    def test_function_call_expr_arg(self):
        def z(a):
            return a

        def func(a, b):
            return z(a + b)

        tree = get_ast(func)
        basic_block = get_basic_block(tree)
        print(basic_block)
        self.assertEqual(len(basic_block), 3)
        self.assertEqual(basic_block[0].targets[0].id, '_t1')
        self.assertIsInstance(basic_block[0].value, ast.Call)
        self.assertEqual(basic_block[0].value.func.value.id, 'a')
        self.assertEqual(basic_block[0].value.func.attr, '__add__')
        self._check_args(basic_block[0].value.args, ['b'])
        self.assertEqual(basic_block[0].targets[0].id, '_t1')
        self.assertIsInstance(basic_block[0].value, ast.Call)
        self.assertEqual(basic_block[1].value.func.id, 'z')
        self._check_args(basic_block[1].value.args, ['_t1'])
        self.assertIsInstance(basic_block[2], ast.Return)
        self.assertEqual(basic_block[2].value.id, '_t0')

    def test_number_binop(self):
        def f(a):
            return 3 * a

        tree = get_ast(f)
        basic_block = get_basic_block(tree)
        self.assertEqual(len(basic_block), 2)


@unittest.skip("Unique variable generation changes break print test")
class TestBasicBlockPrint(unittest.TestCase):
    def _check(self, func, expected):
        block = get_basic_block(get_ast(func))
        self.assertEqual(repr(block), expected)

    def test_simple(self):
        def func(a, b):
            return a + b

        self._check(func, """
BasicBlock
  Name: func
  Params: a, b
  Body:
    _t0 = a.__add__(b)
    return _t0
""")

    def test_multi_line(self):
        def z(a, b):
            return a + b

        def func(a, b):
            c = a * b
            d = z(c)
            return d * z(a + b, a - b)

        self._check(func, """
BasicBlock
  Name: func
  Params: a, b
  Body:
    c = a.__mul__(b)
    d = z(c)
    _t2 = a.__add__(b)
    _t3 = a.__sub__(b)
    _t1 = z(_t2, _t3)
    _t0 = d.__mul__(_t1)
    return _t0
""")


class TestLSF(LazySpecializedFunction):
    def get_placeholder_output(self, args):
        return args[0]


@unittest.skip("Unique variable generation changes break print test")
class TestComposableBlocks(unittest.TestCase):
    def test_no_composable(self):
        a = 3
        b = 1

        def func(a, b):
            return a + b

        tree = get_ast(func)
        basic_block = get_basic_block(tree)
        basic_block = separate_composable_blocks(basic_block,
                                                 dict(globals(), **locals()))
        self.assertIsInstance(basic_block[0], NonComposableBlock)
        self.assertEqual(len(basic_block), 1)

    def test_one_composable(self):
        lsf = TestLSF(None)
        a = 3
        b = 1

        def func(a, b):
            a = lsf(a)
            b = lsf(b)
            return a + b

        tree = get_ast(func)
        basic_block = get_basic_block(tree)
        basic_block = separate_composable_blocks(basic_block,
                                                 dict(globals(), **locals()))
        self.assertIsInstance(basic_block[0], ComposableBlock)
        self.assertIsInstance(basic_block[1], NonComposableBlock)
        self.assertEqual(len(basic_block), 2)

    def test_two_composable(self):
        lsf = TestLSF(None)
        lsf2 = TestLSF(None)
        a = 3
        b = 1

        def func(a, b):
            a = lsf(a)
            b = lsf(b)
            c = a + b
            d = lsf2(c)
            return lsf2(d)

        tree = get_ast(func)
        basic_block = get_basic_block(tree)
        basic_block = separate_composable_blocks(basic_block,
                                                 dict(globals(), **locals()))
        self.assertIsInstance(basic_block[0], ComposableBlock)
        self.assertIsInstance(basic_block[1], NonComposableBlock)
        self.assertIsInstance(basic_block[2], ComposableBlock)
        self.assertIsInstance(basic_block[3], NonComposableBlock)
        self.assertEqual(len(basic_block), 4)


@unittest.skip("Unique variable generation changes break print test")
class TestPrintComposableBlocks(unittest.TestCase):
    def test_no_composable(self):
        a = 3
        b = 1

        def func(a, b):
            return a + b

        tree = get_ast(func)
        basic_block = get_basic_block(tree)
        basic_block = separate_composable_blocks(basic_block, dict(globals(), **locals()))
        self.assertEqual(repr(basic_block), """
BasicBlock
  Name: func
  Params: a, b
  Body:
    NonComposableBlock:
      _t0 = a.__add__(b)
      return _t0
""")

    def test_one_composable(self):
        lsf = TestLSF(None)
        a = 3
        b = 1

        def func(a, b):
            a = lsf(a)
            b = lsf(b)
            return a + b

        tree = get_ast(func)
        basic_block = get_basic_block(tree)
        basic_block = separate_composable_blocks(basic_block, dict(globals(), **locals()))
        self.assertEqual(repr(basic_block), """
BasicBlock
  Name: func
  Params: a, b
  Body:
    ComposableBlock:
      a = lsf(a)
      b = lsf(b)
    NonComposableBlock:
      _t0 = a.__add__(b)
      return _t0
""")

    def test_two_composable(self):
        lsf = TestLSF(None)
        lsf2 = TestLSF(None)
        a = 3
        b = 1

        def func(a, b):
            a = lsf(a)
            b = lsf(b)
            c = a + b
            d = lsf2(c)
            return lsf2(d)

        tree = get_ast(func)
        basic_block = get_basic_block(tree)
        basic_block = separate_composable_blocks(basic_block, dict(globals(), **locals()))
        self.assertEqual(repr(basic_block), """
BasicBlock
  Name: func
  Params: a, b
  Body:
    ComposableBlock:
      a = lsf(a)
      b = lsf(b)
    NonComposableBlock:
      c = a.__add__(b)
    ComposableBlock:
      d = lsf2(c)
      _t0 = lsf2(d)
    NonComposableBlock:
      return _t0
""")
