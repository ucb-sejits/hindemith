import unittest
import numpy as np
from hindemith.types.vector import Vector
from hindemith.core import hm
from ctree.util import Timer


class TestCore(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_add(self):
        @hm
        def fn(a, b):
            return a + b

        a = Vector.rand(512, np.float32)
        b = Vector.rand(512, np.float32)

        c = fn(a, b)
        c.sync()
        self._check(c.data, a.data + b.data)

    def test_two_adds(self):
        @hm
        def fn(a, b, c):
            return a + b + c

        a = Vector.rand(512, np.float32)
        b = Vector.rand(512, np.float32)
        c = Vector.rand(512, np.float32)

        d = fn(a, b, c)
        d.sync()
        np.testing.assert_allclose(d.data, a.data + b.data + c.data)

    def test_intermediate(self):
        @hm
        def fn(a, b, c):
            d = a + b
            return a + b * c + d

        a = Vector.rand(512, np.float32)
        b = Vector.rand(512, np.float32)
        c = Vector.rand(512, np.float32)

        d = fn(a, b, c)
        d.sync()
        d_py = a.data + b.data
        py_result = a.data + b.data * c.data + d_py
        np.testing.assert_allclose(d.data, py_result)
