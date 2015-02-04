from hindemith.operations.gemm import gemm
import unittest
from hindemith import hmarray
import numpy as np


class TestGemm(unittest.TestCase):
    def test_simple(self):
        a = (np.random.rand(256, 256).astype(np.int32) * 100).astype(np.float32)
        b = (np.random.rand(256, 256).astype(np.int32) * 100).astype(np.float32)
        # c = hmarray(np.random.rand(256, 256).astype(np.float32) * 100)
        # a = np.ones((256, 256), np.float32)
        # b = np.ones((256, 256), np.float32)
        c = hmarray(np.zeros((256, 256), np.float32))
        gemm(hmarray(a), hmarray(b), c, 2.4, 0.0)
        c.copy_to_host_if_dirty()
        np.testing.assert_allclose(c, np.dot(a, b) * 2.4)
