import unittest

import numpy as np

from hindemith.operations.zip_with import zip_with
from hindemith.utils import symbols
from hindemith.types.hmarray import hmarray


class TestZipWith(unittest.TestCase):
    def setUp(self):
        self.a = np.random.rand(1024).astype(np.float32) * 100
        self.b = np.random.rand(1024).astype(np.float32) * 100

    def _check(self, actual, expected):
        actual.copy_to_host_if_dirty()
        actual.view(np.ndarray)
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

        actual = zip_with(f, hmarray(a), hmarray(b))
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
        actual = zip_with(th, hmarray(a), hmarray(b))
        expected = np.array([th(x, y) for x, y in zip(a.tolist(), b.tolist())])
        self._check(actual, expected)
