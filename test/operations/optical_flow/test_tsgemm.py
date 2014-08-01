__author__ = 'leonardtruong'

import unittest
import numpy as np
from hindemith.operations.optical_flow.tsgemm import TSGemm


class TestTSGemm(unittest.TestCase):
    def test_simple(self):
        tsgemm = TSGemm()
        A = np.random.rand(10, 10, 10).astype(np.float32) * 100
        B = np.random.rand(10, 10, 10).astype(np.float32) * 100
        C = np.random.rand(10, 10).astype(np.float32) * 100
        actual = tsgemm(A, B, C)
        print(actual)
        # expected = A * B
        # try:
        #     np.testing.assert_array_almost_equal(actual, expected)
        # except AssertionError as e:
        #     self.fail("Outputs not equal: %s" % e.message)
