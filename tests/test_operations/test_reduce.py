from .base_test import HMBaseTest

from hindemith.operations.reduce import sum
from hindemith.types.hmarray import hmarray
import numpy as np


class TestReduce(HMBaseTest):
    def test_simple_sum(self):
        a = np.ones(1024).astype(np.float32)
        hm_a = hmarray(a)
        actual = sum(hm_a)
        self.assertEqual(actual, np.sum(a))

    def test_2d_sum(self):
        a = np.ones((1024, 1024)).astype(np.float32)
        hm_a = hmarray(a)
        actual = sum(hm_a)
        self.assertEqual(actual, np.sum(a))
