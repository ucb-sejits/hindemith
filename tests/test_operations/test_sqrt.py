__author__ = 'leonardtruong'

from .base_test import HMBaseTest

from hindemith.operations.map import sqrt
from hindemith.types.hmarray import hmarray

import numpy as np


class TestSqrt(HMBaseTest):
    def test_simple(self):
        a = self.a
        self._check(sqrt(hmarray(a)), np.sqrt(a))
