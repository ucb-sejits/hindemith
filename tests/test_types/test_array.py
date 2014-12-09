import unittest
import numpy as np
from hindemith.types.hmarray import hmarray
from hindemith.types.hmarray import add, sub, mul, div


class TestOverride(unittest.TestCase):
    def setUp(self):
        self.a = np.random.rand(32, 64).astype(np.float32) * 100
        self.b = np.random.rand(32, 64).astype(np.float32) * 100
        self.c = np.random.rand(32, 64).astype(np.float32) * 100

    def _check(self, expected, actual):
        expected.copy_to_host_if_dirty()
        expected = np.copy(expected)
        try:
            np.testing.assert_array_almost_equal(expected, actual, decimal=1)
        except AssertionError as e:
            self.fail(e)

    def test_simple_add(self):
        a, b = self.a, self.b
        self._check(hmarray(a) + hmarray(b), a + b)

    def test_simple_sub(self):
        a, b = self.a, self.b
        self._check(hmarray(a) - hmarray(b), a - b)

    def test_simple_mul(self):
        a, b = self.a, self.b
        self._check(hmarray(a) * hmarray(b), a * b)

    def test_simple_div(self):
        a, b = self.a, self.b
        self._check(hmarray(a) / hmarray(b), a / b)


class TestElt(unittest.TestCase):
    def setUp(self):
        self.a = np.random.rand(640, 480).astype(np.float32) * 100
        self.b = np.random.rand(640, 480).astype(np.float32) * 10000
        self.c = np.random.rand(640, 480).astype(np.float32) * 2000

    def _check(self, expected, actual):
        expected.copy_to_host_if_dirty()
        expected = np.copy(expected)
        try:
            np.testing.assert_array_almost_equal(expected, actual, decimal=1)
        except AssertionError as e:
            self.fail(e)

    def test_simple_add(self):
        a, b = self.a, self.b
        hm_a, hm_b = hmarray(a), hmarray(b)
        self._check(add(hm_a, hm_b), a + b)

    def test_multi_add(self):
        a, b, c = self.a, self.b, self.c
        hm_a, hm_b, hm_c = hmarray(a), hmarray(b), hmarray(c)
        self._check(add(add(hm_a, hm_b), hm_c), a + b + c)

    def test_simple_mul(self):
        a, b = self.a, self.b
        hm_a, hm_b = hmarray(a), hmarray(b)
        self._check(mul(hm_a, hm_b), a * b)

    def test_multi_mul(self):
        a, b, c = self.a, self.b, self.c
        hm_a, hm_b, hm_c = hmarray(a), hmarray(b), hmarray(c)
        self._check(mul(mul(hm_a, hm_b), hm_c), a * b * c)

    def test_simple_sub(self):
        a, b = self.a, self.b
        hm_a, hm_b = hmarray(a), hmarray(b)
        self._check(sub(hm_a, hm_b), a - b)

    def test_multi_sub(self):
        a, b, c = self.a, self.b, self.c
        hm_a, hm_b, hm_c = hmarray(a), hmarray(b), hmarray(c)
        self._check(sub(sub(hm_a, hm_b), hm_c), a - b - c)

    def test_simple_div(self):
        a, b = self.a, self.b
        hm_a, hm_b = hmarray(a), hmarray(b)
        self._check(div(hm_a, hm_b), a / b)

    def test_multi_div(self):
        a, b, c = self.a, self.b, self.c
        hm_a, hm_b, hm_c = hmarray(a), hmarray(b), hmarray(c)
        self._check(div(div(hm_a, hm_b), hm_c), a / b / c)

    def test_combined1(self):
        a, b, c = self.a, self.b, self.c
        hm_a, hm_b, hm_c = hmarray(a), hmarray(b), hmarray(c)
        self._check(div(add(hm_a, hm_b), hm_c), (a + b) / c)

    def test_combined2(self):
        a, b, c = self.a, self.b, self.c
        hm_a, hm_b, hm_c = hmarray(a), hmarray(b), hmarray(c)
        self._check(mul(add(hm_a, hm_b), hm_c), (a + b) * c)

    def test_combined3(self):
        a, b, c = self.a, self.b, self.c
        hm_a, hm_b, hm_c = hmarray(a), hmarray(b), hmarray(c)
        self._check(mul(div(hm_a, hm_b), hm_c), (a / b) * c)

    def test_combined4(self):
        a, b, c = self.a, self.b, self.c
        hm_a, hm_b, hm_c = hmarray(a), hmarray(b), hmarray(c)
        self._check(mul(sub(hm_a, hm_b), hm_c), (a - b) * c)


class TestScalarArrayElt(unittest.TestCase):
    def setUp(self):
        self.a = np.random.rand(4, 4).astype(np.float32) * 100
        self.b = np.random.rand(4, 4).astype(np.float32) * 100
        self.c = np.random.rand(4, 4).astype(np.float32) * 100

    def _check(self, expected, actual):
        expected.copy_to_host_if_dirty()
        expected = np.copy(expected)
        try:
            np.testing.assert_array_almost_equal(expected, actual, decimal=3)
        except AssertionError as e:
            self.fail(e)

    def test_simple_add1(self):
        a = self.a
        hm_a = hmarray(a)
        self._check(add(hm_a, 3), a + 3)

    def test_simple_add2(self):
        a = self.a
        hm_a = hmarray(a)
        self._check(add(3, hm_a), a + 3)

    def test_simple_mul1(self):
        a = self.a
        hm_a = hmarray(a)
        self._check(mul(hm_a, 3), a * 3)

    def test_simple_mul2(self):
        a = self.a
        hm_a = hmarray(a)
        self._check(mul(3, hm_a), a * 3)

    def test_simple_sub1(self):
        a = self.a
        hm_a = hmarray(a)
        self._check(sub(hm_a, 3), a - 3)

    def test_simple_sub2(self):
        a = self.a
        hm_a = hmarray(a)
        self._check(sub(3, hm_a), 3 - a)

    def test_simple_div1(self):
        a = self.a
        hm_a = hmarray(a)
        self._check(div(hm_a, 3), a / 3)

    def test_simple_div2(self):
        a = self.a
        hm_a = hmarray(a)
        self._check(div(3, hm_a), 3 / a)
