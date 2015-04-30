from hindemith.core import compose, get_ast, UnpackBinOps, ReplaceArrayOps
from hindemith.types import hmarray
from hindemith.operations.array import ArrayAdd, ArraySub, ArrayMul, ArrayDiv, \
    Square, Sqrt, ArrayScalarAdd, ArrayScalarSub, ArrayScalarDiv, \
    ArrayScalarMul
import numpy as np
import unittest
import ast


class TestArrays(unittest.TestCase):
    def setUp(self):
        self.a = hmarray.random((256, 256), _range=(0, 255))
        self.b = hmarray.random((256, 256), _range=(0, 255))
        self.c = hmarray.random((256, 256), _range=(0, 255))

    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_add(self):
        @compose
        def fn(a, b, c):
            c = ArrayAdd(a, b)
            return c

        a, b, c = self.a, self.b, self.c
        c = fn(a, b, c)
        c.sync_host()
        self._check(c, a + b)

    def test_sub(self):
        @compose
        def fn(a, b, c):
            c = ArraySub(a, b)
            return c

        a, b, c = self.a, self.b, self.c
        c = fn(a, b, c)
        c.sync_host()
        self._check(c, a - b)

    def test_mul(self):
        @compose
        def fn(a, b, c):
            c = ArrayMul(a, b)
            return c

        a, b, c = self.a, self.b, self.c
        c = fn(a, b, c)
        c.sync_host()
        self._check(c, a * b)

    def test_div(self):
        @compose
        def fn(a, b, c):
            c = ArrayDiv(a, b)
            return c

        a, b, c = self.a, self.b, self.c
        c = fn(a, b, c)
        c.sync_host()
        np.testing.assert_array_almost_equal(c, a / b, decimal=2)

    def test_square(self):
        @compose
        def fn(a, b):
            b = Square(a)
            return b

        a, b = self.a, self.b
        b = fn(a, b)
        b.sync_host()
        np.testing.assert_array_almost_equal(b, np.square(a), decimal=2)

    def test_sqrt(self):
        @compose
        def fn(a, b):
            b = Sqrt(a)
            return b

        a, b = self.a, self.b
        b = fn(a, b)
        b.sync_host()
        np.testing.assert_array_almost_equal(b, np.sqrt(a), decimal=2)

    def test_scalar_add(self):
        @compose
        def fn(a, b, c):
            c = ArrayScalarAdd(a, b)
            return c

        a, c = self.a, self.b
        b = 3.23
        c = fn(a, b, c)
        c.sync_host()
        np.testing.assert_array_almost_equal(c, a + b, decimal=4)

    def test_scalar_sub(self):
        @compose
        def fn(a, b, c):
            c = ArrayScalarSub(a, b)
            return c

        a, c = self.a, self.b
        b = 3.23
        c = fn(a, b, c)
        c.sync_host()
        np.testing.assert_array_almost_equal(c, a - b, decimal=4)

    def test_scalar_Mul(self):
        @compose
        def fn(a, b, c):
            c = ArrayScalarMul(a, b)
            return c

        a, c = self.a, self.b
        b = 3.23
        c = fn(a, b, c)
        c.sync_host()
        np.testing.assert_array_almost_equal(c, a * b, decimal=4)

    def test_scalar_div(self):
        @compose
        def fn(a, b, c):
            c = ArrayScalarDiv(a, b)
            return c

        a, c = self.a, self.b
        b = 3.0
        c = fn(a, b, c)
        c.sync_host()
        np.testing.assert_array_almost_equal(c, a / b, decimal=4)

    def test_composed(self):
        @compose
        def fn(a, b, c):
            c = ArrayAdd(a, b)
            c = ArrayMul(c, b)
            c = ArrayDiv(a, c)
            return c

        a, b, c = self.a, self.b, self.c
        c = fn(a, b, c)
        c.sync_host()
        np.testing.assert_array_almost_equal(c, a / ((a + b) * b), decimal=4)


class TestArrayUnpacker(unittest.TestCase):
    def setUp(self):
        self.a = hmarray.random((256, 256), _range=(0, 255))
        self.b = hmarray.random((256, 256), _range=(0, 255))
        self.c = hmarray.random((256, 256), _range=(0, 255))

    def test_unpack_binops(self):
        def fn(a, b):
            _t2 = a + b
            _t1 = a / _t2
            _t0 = _t1 * b
            return _t0

        expected = get_ast(fn)

        def fn(a, b):
            return a / (a + b) * b

        result = UnpackBinOps().visit(get_ast(fn))
        self.assertEqual(ast.dump(expected), ast.dump(result))

    def test_conver_arrayops(self):
        def fn(a, b):
            _t2 = ArrayAdd(a, b)
            _t1 = ArrayDiv(a, _t2)
            _t0 = ArrayMul(_t1, b)
            return _t0

        expected = get_ast(fn)

        def fn(a, b):
            return a / (a + b) * b

        symbol_table = {
            'a': self.a,
            'b': self.b
        }

        unpacked = UnpackBinOps().visit(get_ast(fn))
        result = ReplaceArrayOps(symbol_table).visit(unpacked)
        self.assertEqual(ast.dump(expected), ast.dump(result))
