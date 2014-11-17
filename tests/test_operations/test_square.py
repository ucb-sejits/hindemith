__author__ = 'leonardtruong'

from .base_test import HMBaseTest

from hindemith.types.hmarray import hmarray, square

import numpy as np


class TestSqrt(HMBaseTest):
    def test_simple(self):
        a = self.a
        self._check(square(hmarray(a)), np.square(a))
