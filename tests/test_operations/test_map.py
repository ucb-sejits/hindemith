__author__ = 'leonardtruong'

import unittest
import numpy as np

from hindemith.operations.map import hmmap
from hindemith.types.hmarray import hmarray


class TestMap(unittest.TestCase):
    def setUp(self):
        self.a = np.random.rand(1024, 1024).astype(np.float32) * 100

    def _check(self, actual, expected):
        try:
            actual.copy_to_host_if_dirty()
            actual = actual.view(np.ndarray)
            np.testing.assert_array_almost_equal(actual, expected)
        except AssertionError as e:
            self.fail(e)

    def test_simple_map(self):
        a = self.a

        def func(a):
            if a < 50:
                return 0
            else:
                return 1

        @hmmap
        def specfunc(a):
            if a < 50:
                return 0
            else:
                return 1

        vfunc = np.vectorize(func)
        self._check(specfunc(hmarray(a)), vfunc(a))
