import unittest
from hindemith.core import compose
from hindemith.types import hmarray
from hindemith.operations.concat import ConcatForward
import numpy as np


class TestSoftmax(unittest.TestCase):
    def test_forward(self):
        @compose
        def fn(a, b, c, d):
            d = ConcatForward(a, b, c)
            return d

        a = hmarray.random((16, 12, 55, 55))
        b = hmarray.random((16, 12, 55, 55))
        c = hmarray.random((16, 12, 55, 55))
        d = hmarray.zeros((16, 36, 55, 55))
        expected = hmarray.zeros((16, 36, 55, 55))
        fn(a, b, c, d)
        d.sync_host()
        np.testing.assert_array_almost_equal(d[:16, 0:12, ...], a)
        np.testing.assert_array_almost_equal(d[:16, 12:24, ...], b)
        np.testing.assert_array_almost_equal(d[:16, 24:36, ...], c)
