__author__ = 'leonardtruong'

import unittest
import numpy as np


class HMBaseTest(unittest.TestCase):
    def setUp(self):
        self.a = np.random.rand(1024, 512).astype(np.float32) * 100
        self.b = np.random.rand(1024, 512).astype(np.float32) * 100

    def _check(self, actual, expected):
        try:
            actual.copy_to_host_if_dirty()
            actual = actual.view(np.ndarray)
            np.testing.assert_array_almost_equal(actual, expected, decimal=5)
        except AssertionError as e:
            self.fail(e)
