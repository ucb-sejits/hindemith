__author__ = 'leonardtruong'

import unittest
import numpy as np

from hindemith.operations.dense_linear_algebra.array_op import ArrayAdd, ArrayMul, ArraySub, \
    ArrayDiv
from hindemith.types.common import Array


@unittest.skip("Deprecated")
class TestArrayOps(unittest.TestCase):
    def _check(self, specializer):
        rand1 = Array(
            'testArr1', np.random.rand(60, 80).astype(np.float32) * 100
        )
        rand2 = Array(
            'testArr2', np.random.rand(60, 80).astype(np.float32) * 100
        )
        specialized = specializer(rand1.name, rand1.data, backend='ocl')
        actual = specialized(rand2)
        python = specializer(rand1.name, rand1.data, backend='python')
        expected = python(rand2)
        try:
            np.testing.assert_array_almost_equal(
                actual.data, expected.data, decimal=3
            )
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def test_simple_array_add(self):
        self._check(ArrayAdd)

    def test_simple_array_sub(self):
        self._check(ArraySub)

    def test_simple_array_mul(self):
        self._check(ArrayMul)

    def test_simple_array_div(self):
        self._check(ArrayDiv)


@unittest.skip("Deprecated")
class TestNativeArrayOps(unittest.TestCase):
    def _check(self, actual, expected):
        try:
            np.testing.assert_array_almost_equal(
                actual.data, expected, decimal=3
            )
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

    def setUp(self):
        self.rand1 = Array(
            'testArr1', np.random.rand(60, 80).astype(np.float32) * 100
        )
        self.rand2 = Array(
            'testArr2', np.random.rand(60, 80).astype(np.float32) * 100
        )

    def test_native_array_add(self):
        actual = self.rand1 + self.rand2
        expected = self.rand1.data + self.rand2.data
        self._check(actual, expected)

    def test_native_array_sub(self):
        actual = self.rand1 - self.rand2
        expected = self.rand1.data - self.rand2.data
        self._check(actual, expected)

    def test_native_array_mul(self):
        actual = self.rand1 * self.rand2
        expected = self.rand1.data * self.rand2.data
        self._check(actual, expected)

    def test_native_array_div(self):
        actual = self.rand1 / self.rand2
        expected = self.rand1.data / self.rand2.data
        self._check(actual, expected)
