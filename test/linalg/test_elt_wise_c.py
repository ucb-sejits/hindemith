import unittest
from hindemith.linalg import add, sub, mul, div, EltWiseArrayOp
import numpy as np

EltWiseArrayOp.backend = 'c'


class TestElt(unittest.TestCase):
    def setUp(self):
        self.a = np.random.rand(1024, 1024).astype(np.float32) * 100
        self.b = np.random.rand(1024, 1024).astype(np.float32) * 100
        self.c = np.random.rand(1024, 1024).astype(np.float32) * 100

    def _check(self, expected, actual):
        try:
            np.testing.assert_array_almost_equal(expected, actual)
        except AssertionError as e:
            self.fail(e)

    def test_simple_add(self):
        a, b = self.a, self.b
        self._check(add(a, b), a + b)

    def test_multi_add(self):
        a, b, c = self.a, self.b, self.c
        self._check(add(add(a, b), c), a + b + c)

    def test_simple_mul(self):
        a, b = self.a, self.b
        self._check(mul(a, b), a * b)

    def test_multi_mul(self):
        a, b, c = self.a, self.b, self.c
        self._check(mul(mul(a, b), c), a * b * c)

    def test_simple_sub(self):
        a, b = self.a, self.b
        self._check(sub(a, b), a - b)

    def test_multi_sub(self):
        a, b, c = self.a, self.b, self.c
        self._check(sub(sub(a, b), c), a - b - c)

    def test_simple_div(self):
        a, b = self.a, self.b
        self._check(div(a, b), a / b)

    def test_multi_div(self):
        a, b, c = self.a, self.b, self.c
        self._check(div(div(a, b), c), a / b / c)

    def test_combined1(self):
        a, b, c = self.a, self.b, self.c
        self._check(div(add(a, b), c), (a + b) / c)

    def test_combined2(self):
        a, b, c = self.a, self.b, self.c
        self._check(mul(add(a, b), c), (a + b) * c)

    def test_combined3(self):
        a, b, c = self.a, self.b, self.c
        self._check(mul(div(a, b), c), (a / b) * c)

    def test_combined4(self):
        a, b, c = self.a, self.b, self.c
        self._check(mul(sub(a, b), c), (a - b) * c)


class TestScalarArrayElt(unittest.TestCase):
    def setUp(self):
        self.a = np.random.rand(1024, 1024).astype(np.float32) * 100
        self.b = np.random.rand(1024, 1024).astype(np.float32) * 100
        self.c = np.random.rand(1024, 1024).astype(np.float32) * 100

    def _check(self, expected, actual):
        try:
            np.testing.assert_array_almost_equal(expected, actual)
        except AssertionError as e:
            self.fail(e)

    def test_simple_add1(self):
        a = self.a
        self._check(add(a, 3), a + 3)

    def test_simple_add2(self):
        a = self.a
        self._check(add(3, a), a + 3)

    def test_simple_mul1(self):
        a = self.a
        self._check(mul(a, 3), a * 3)

    def test_simple_mul2(self):
        a = self.a
        self._check(mul(3, a), a * 3)

    def test_simple_sub1(self):
        a = self.a
        self._check(sub(a, 3), a - 3)

    def test_simple_sub2(self):
        a = self.a
        self._check(sub(3, a), 3 - a)

    def test_simple_div1(self):
        a = self.a
        self._check(div(a, 3), a / 3)

    def test_simple_div2(self):
        a = self.a
        self._check(div(3, a), 3 / a)
