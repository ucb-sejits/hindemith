import unittest
import numpy as np
from hindemith.types.array import hmarray
from hindemith.core import hm


class TestCore(unittest.TestCase):
    def test_add(self):
        @hm
        def fn(a, b):
            return a + b

        a = hmarray((512, 512), np.float32)
        b = hmarray((512, 512), np.float32)

        c = fn(a, b)
        np.testing.assert_allclose(c, a + b)
