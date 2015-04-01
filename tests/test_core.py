import unittest
import numpy as np
from hindemith.types.array import hmarray
from hindemith.core import hm


class TestCore(unittest.TestCase):
    def test_add(self):
        @hm
        def fn(a, b):
            return a + b

        a = np.random.rand(512, 512).astype(np.float32).view(hmarray)
        b = np.random.rand(512, 512).astype(np.float32).view(hmarray)

        c = fn(a, b)
        np.testing.assert_allclose(c, a + b)
