import unittest
import numpy as np
from hindemith.types import hmarray


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
