import unittest
import numpy as np
from hindemith.types import NDArray
from hindemith.core import hm


class TestCore(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_array_almost_equal(actual, expected, decimal=1)

    def test_add(self):
        @hm
        def fn(a, b):
            return a + b

        a = NDArray.rand((512, ), np.float32)
        b = NDArray.rand((512, ), np.float32)

        c = fn(a, b)
        c.sync()
        self._check(c, a + b)

    def test_two_adds(self):
        @hm
        def fn(a, b, c):
            return a + b + c

        a = NDArray.rand((512, ), np.float32)
        b = NDArray.rand((512, ), np.float32)
        c = NDArray.rand((512, ), np.float32)

        d = fn(a, b, c)
        d.sync()
        self._check(d, a + b + c)

    def test_intermediate(self):
        @hm
        def fn(a, b, c):
            d = a + b
            return a + b * c + d

        a = NDArray.rand((512, ), np.float32)
        b = NDArray.rand((512, ), np.float32)
        c = NDArray.rand((512, ), np.float32)

        d = fn(a, b, c)
        d.sync()
        d_py = a + b
        py_result = a + b * c + d_py
        self._check(d, py_result)

    def test_for(self):
        @hm
        def fn(a, b):
            c = a + b
            for i in range(10):
                c = a + c
            return c

        a = NDArray.rand((512, ), np.float32)
        b = NDArray.rand((512, ), np.float32)
        c = NDArray.rand((512, ), np.float32)

        c = fn(a, b)
        c.sync()
        expected = a + b
        for i in range(10):
            expected = a + expected
        self._check(c, expected)

    def test_matrix(self):
        @hm
        def fn(a, b):
            c = a + b
            for i in range(10):
                c = a + c
            return c

        a = NDArray.rand((512, 512), np.float32)
        b = NDArray.rand((512, 512), np.float32)
        c = NDArray.rand((512, 512), np.float32)

        c = fn(a, b)
        c.sync()
        expected = a + b
        for i in range(10):
            expected = a + expected
        self._check(c, expected)

    def test_multiple_calls(self):
        @hm
        def fn(a, b):
            c = a + b
            for i in range(10):
                c = a + c
            return c

        for i in range(3):
            a = NDArray.rand((512, 512), np.float32)
            b = NDArray.rand((512, 512), np.float32)
            c = NDArray.rand((512, 512), np.float32)

            c = fn(a, b)
            c.sync()
            expected = a + b
            for i in range(10):
                expected = a + expected
            self._check(c, expected)

    def test_scalars(self):
        @hm
        def fn(a, b, alpha):
            c = a + b
            for i in range(10):
                c = alpha / a + c - alpha
            return c

        a = NDArray.rand((512, 512), np.float32)
        b = NDArray.rand((512, 512), np.float32)
        c = NDArray.rand((512, 512), np.float32)

        c = fn(a, b, 4.6)
        c.sync()
        expected = a + b
        for i in range(10):
            expected = 4.6 / a + expected - 4.6
        self._check(c, expected)

    def test_scalars_inline(self):
        @hm
        def fn(a, b):
            c = a + b
            for i in range(10):
                c = 3.2 / a + c - 1.8
            return c

        a = NDArray.rand((512, 512), np.float32)
        b = NDArray.rand((512, 512), np.float32)
        c = NDArray.rand((512, 512), np.float32)

        c = fn(a, b)
        c.sync()
        expected = a + b
        for i in range(10):
            expected = 3.2 / a + expected - 1.8
        self._check(c, expected)
