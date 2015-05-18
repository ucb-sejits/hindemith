import unittest
import hindemith as hm
from hindemith.core import compose
from hindemith.types import hmarray
from hindemith.operations.concat import ConcatForward
import numpy as np


class TestConcat(unittest.TestCase):
    def test_forward(self):
        @compose
        def fn(a, b, c, d):
            d = ConcatForward(a, b, c)
            return d

        a = hm.random((16, 12, 55, 55))
        b = hm.random((16, 12, 55, 55))
        c = hm.random((16, 12, 55, 55))
        d = hm.zeros((16, 36, 55, 55))
        d = fn(a, b, c, d)
        d.sync_host()
        np.testing.assert_array_almost_equal(d[:16, 0:12, ...], a)
        np.testing.assert_array_almost_equal(d[:16, 12:24, ...], b)
        np.testing.assert_array_almost_equal(d[:16, 24:36, ...], c)
