import unittest
import numpy as np
from hindemith.types import Vector, Matrix
from hindemith.core import hm


class TestCore(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_array_almost_equal(actual, expected, decimal=2)

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
        self._check(d.data, a.data + b.data + c.data)

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
        self._check(d.data, py_result)

    def test_for(self):
        @hm
        def fn(a, b):
            c = a + b
            for i in range(10):
                c = a + c
            return c

        a = Vector.rand(512, np.float32)
        b = Vector.rand(512, np.float32)
        c = Vector.rand(512, np.float32)

        c = fn(a, b)
        c.sync()
        expected = a.data + b.data
        for i in range(10):
            expected = a.data + expected
        self._check(c.data, expected)

    def test_matrix(self):
        @hm
        def fn(a, b):
            c = a + b
            for i in range(10):
                c = a + c
            return c

        a = Matrix.rand((512, 512), np.float32)
        b = Matrix.rand((512, 512), np.float32)
        c = Matrix.rand((512, 512), np.float32)

        c = fn(a, b)
        c.sync()
        expected = a.data + b.data
        for i in range(10):
            expected = a.data + expected
        self._check(c.data, expected)

    def test_multiple_calls(self):
        @hm
        def fn(a, b):
            c = a + b
            for i in range(10):
                c = a + c
            return c

        for i in range(3):
            a = Matrix.rand((512, 512), np.float32)
            b = Matrix.rand((512, 512), np.float32)
            c = Matrix.rand((512, 512), np.float32)

            c = fn(a, b)
            c.sync()
            expected = a.data + b.data
            for i in range(10):
                expected = a.data + expected
            self._check(c.data, expected)

    def test_scalars(self):
        @hm
        def fn(a, b, alpha):
            c = a + b
            for i in range(10):
                c = alpha / a + c - alpha
            return c

        a = Matrix.rand((512, 512), np.float32)
        b = Matrix.rand((512, 512), np.float32)
        c = Matrix.rand((512, 512), np.float32)

        c = fn(a, b, 4.6)
        c.sync()
        expected = a.data + b.data
        for i in range(10):
            expected = 4.6 / a.data + expected - 4.6
        print(c.data)
        self._check(c.data, expected)
