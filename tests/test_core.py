import unittest
import numpy as np
from hindemith.types.vector import Vector
from hindemith.core import hm


class TestCore(unittest.TestCase):
    def test_add(self):
        @hm
        def fn(a, b):
            return a + b

        a = Vector.rand(512, np.float32)
        b = Vector.rand(512, np.float32)

        c = fn(a, b)
        c.sync()
        np.testing.assert_allclose(c.data, a.data + b.data)
