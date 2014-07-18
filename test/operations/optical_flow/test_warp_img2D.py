__author__ = 'leonardtruong'

import unittest
import numpy as np

from teller.operations.optical_flow.warp_img2D import WarpImg2D
from teller.operations.dense_linear_algebra.array_op import Array

class TestWarpImg2D(unittest.TestCase):
    def test_simple_warp_img_2D(self):
        specialized = WarpImg2D(backend='ocl')
        python = WarpImg2D(backend='c')
        rand_arr = Array('testArr', np.random.rand(60, 80).astype(np.float32) * 100)
        u = Array('u', np.random.rand(60, 80).astype(np.float32) * 100)
        v = Array('v', np.random.rand(60, 80).astype(np.float32) * 100)
        actual = specialized(rand_arr, u, v)
        expected = python(rand_arr, u, v)
        try:
            np.testing.assert_array_almost_equal(actual.data, expected.data)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e.message)

