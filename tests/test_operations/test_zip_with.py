import unittest

import numpy as np

from hindemith.operations.zip_with import zip_with, ZipWith
from hindemith.utils import symbols
from hindemith.types.hmarray import hmarray


class TestZipWithOcl(unittest.TestCase):
    def setUp(self):
        ZipWith.backend = 'ocl'
        self.a = np.random.rand(32768).astype(np.float32) * 100
        self.b = np.random.rand(32768).astype(np.float32) * 100

    def _check(self, actual, expected):
        actual.copy_to_host_if_dirty()
        actual = np.copy(actual)
        try:
            np.testing.assert_array_almost_equal(actual, expected,
                                                 decimal=5)
        except AssertionError as e:
            self.fail(e)

    def test_simple(self):
        a, b = self.a, self.b

        def f(a, b):
            if a + b > 100:
                return a + b
            else:
                return a - b

        specialized = zip_with(f)
        actual = specialized(hmarray(a), hmarray(b))
        expected = np.array([f(x, y) for x, y in zip(a.tolist(), b.tolist())])
        self._check(actual, expected)

    def test_complex(self):

        def square(x):
            return x**2

        l = .15
        theta = .3
        symbs = {
            'l': l,
            'theta': theta
        }

        @symbols(symbs)
        def th(p_elt, I1wg_elt):
            threshold = l * theta * pow(abs(I1wg_elt), 2)
            if p_elt < -threshold:
                return l * theta * I1wg_elt
            elif p_elt > threshold:
                return -l * theta * I1wg_elt
            else:
                return -threshold * (I1wg_elt / pow(abs(I1wg_elt), 2))

        a, b = self.a, self.b
        specialized = zip_with(th)
        actual = specialized(hmarray(a), hmarray(b))
        expected = np.array([th(x, y) for x, y in zip(a.tolist(), b.tolist())])
        self._check(actual, expected)


class TestZipWithC(unittest.TestCase):
    def setUp(self):
        ZipWith.backend = 'c'
        self.a = np.random.rand(512, 1024).astype(np.float32) * 100
        self.b = np.random.rand(512, 1024).astype(np.float32) * 100

    def _check(self, actual, expected):
        actual.copy_to_host_if_dirty()
        actual = np.copy(actual)
        try:
            np.testing.assert_array_almost_equal(actual, expected,
                                                 decimal=4)
        except AssertionError as e:
            self.fail(e)

    def test_simple(self):
        a, b = self.a, self.b

        def f(a, b):
            if a + b > 100:
                return a + b
            else:
                return a - b
        specialized = zip_with(f)
        actual = specialized(hmarray(a), hmarray(b))
        thv = np.vectorize(f, otypes=[np.float32])
        expected = thv(a, b)
        self._check(actual, expected)

    def test_complex(self):

        l = .15
        theta = .3
        symbs = {
            'l': l,
            'theta': theta
        }

        @symbols(symbs)
        def th(p_elt, I1wg_elt):
            threshold = l * theta * I1wg_elt
            if p_elt < -threshold:
                return l * theta * I1wg_elt
            elif p_elt > threshold:
                return -l * theta * I1wg_elt
            else:
                return -threshold * p_elt

        def py_th(p_elt, I1wg_elt):
            threshold = l * theta * I1wg_elt
            if p_elt < -threshold:
                return l * theta * I1wg_elt
            elif p_elt > threshold:
                return -l * theta * I1wg_elt
            else:
                return -threshold * p_elt

        a, b = self.a, self.b
        specialized = zip_with(th)
        actual = specialized(hmarray(a), hmarray(b))
        thv = np.vectorize(py_th, otypes=[np.float32])
        expected = thv(a, b)
        self._check(actual, expected)
