import unittest
import numpy as np

from hindemith.types.common import Array
from hindemith.operations.optical_flow.reductions import *

class TestReductions(unittest.TestCase):
    def test_pure_python(self):
        x = Array('x',np.random.rand(100,100))
        sum = Sum(pure_python=True)
        min = Min(pure_python=True)
        max = Max(pure_python=True)
        try:
            assert sum(x) == np.sum(x.data)
        except AssertionError as e:
            self.fail("Sum not equal! %s" % e.message)
        try:
            assert min(x) == np.min(x.data)
        except AssertionError as e:
            self.fail("Min not equal! %s" % e.message)
        try:
            assert max(x) == np.max(x.data)
        except AssertionError as e:
            self.fail("Max not equal! %s" % e.message)

    def test_specialized(self):
        x = Array('x',np.random.rand(100,100))
        sum = Sum()
        min = Min()
        max = Max()
        try:
            assert sum(x) == np.sum(x.data)
        except AssertionError as e:
            self.fail("Sum not equal! %s" % e.message)
        try:
            assert min(x) == np.min(x.data)
        except AssertionError as e:
            self.fail("Min not equal! %s" % e.message)
        try:
            assert max(x) == np.max(x.data)
        except AssertionError as e:
            self.fail("Max not equal! %s" % e.message)