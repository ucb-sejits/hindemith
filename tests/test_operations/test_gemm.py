from hindemith.operations.gemm import gemm
import unittest
from hindemith import hmarray
import numpy as np


class TestGemm(unittest.TestCase):
    def test_simple(self):
        a = (np.random.rand(256, 256) * 100).astype(np.int32).astype(np.float32)
        b = (np.random.rand(256, 256) * 100).astype(np.int32).astype(np.float32)
        # c = hmarray(np.random.rand(256, 256).astype(np.float32) * 100)
        # a = np.ones((256, 256), np.float32)
        # b = np.ones((256, 256), np.float32)
        c = hmarray(np.zeros((256, 256), np.float32))
        gemm(hmarray(a), hmarray(b), c, 1.0, 0.0)
        c.copy_to_host_if_dirty()
        c = np.copy(c)
        np.testing.assert_allclose(c, np.dot(a, b) * 1.0)

    def test_nonsquare(self):
        a = (np.random.rand(512, 256) * 100).astype(np.int32).astype(np.float32)
        b = (np.random.rand(256, 64) * 100).astype(np.int32).astype(np.float32)
        c = np.random.rand(512, 64).astype(np.float32) * 100
        # a = np.ones((256, 256), np.float32)
        # b = np.ones((256, 256), np.float32)
        expected = np.dot(a, b) * 2.4 + .3 * c
        actual = hmarray(c)
        gemm(hmarray(a), hmarray(b), actual, 2.4, .3)
        actual.copy_to_host_if_dirty()
        actual = np.copy(actual)
        np.testing.assert_allclose(actual, expected, rtol=1e-06)

    def test_non_multiple(self):
        a = (np.random.rand(500, 300) * 100).astype(np.int32).astype(np.float32)
        b = (np.random.rand(300, 100) * 100).astype(np.int32).astype(np.float32)
        c = np.random.rand(500, 100).astype(np.float32) * 100
        # a = np.ones((256, 256), np.float32)
        # b = np.ones((256, 256), np.float32)
        expected = np.dot(a, b) * 2.4 + .3 * c
        actual = hmarray(c)
        gemm(hmarray(a), hmarray(b), actual, 2.4, .3)
        actual.copy_to_host_if_dirty()
        actual = np.copy(actual)
        np.testing.assert_allclose(actual, expected, rtol=1e-06)
