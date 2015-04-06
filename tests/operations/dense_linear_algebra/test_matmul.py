import unittest
import numpy as np
from hindemith.types import NDArray
from hindemith.core import hm
from hindemith.operations import MMult


class TestDot(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_simple(self):
        @hm
        def fn(a, b, c):
            c = MMult(a, b)
            return c

        a = NDArray.rand((512, 512), np.float32) * 255
        a.ocl_dirty = True
        b = NDArray.rand((512, 512), np.float32)
        b.ocl_dirty = True
        c = NDArray.rand((512, 512), np.float32)

        c = fn(a, b, c)
        c.sync()
        self._check(c, a.dot(b))
        
    def test_not_square(self):
        @hm
        def fn(a, b, c):
            c = MMult(a, b)
            return c

        a = NDArray.rand((512, 256), np.float32) * 100
        a.ocl_dirty = True
        b = NDArray.rand((256, 128), np.float32)
        b.ocl_dirty = True
        c = NDArray.rand((512, 128), np.float32)

        c = fn(a, b, c)
        c.sync()
        self._check(c, a.dot(b))
        
