import unittest
from math import exp
from hindemith.core import compose
from hindemith.types import hmarray
import hindemith as hm
from hindemith.operations.softmax import SoftmaxForward


class TestSoftmax(unittest.TestCase):
    def test_forward(self):
        @compose
        def fn(bottom, label, top):
            top = SoftmaxForward(bottom, label)
            return top

        bottom = hm.random((6, 13, 1), _range=(-5, 5))
        label = hmarray((6, ))
        for i in range(6):
            label = i % 6
        top = hmarray((6, 13, 1))
        fn(bottom, label, top)

        top.sync_host()
        for i in range(6):
            sum = 0
            for c in range(13):
                sum += top[i, c, 0]
            self.assertTrue(sum > .999)
            self.assertTrue(sum < 1.001)
            scale = 0
            for c in range(13):
                scale += exp(bottom[i, c, 0])
            for c in range(13):
                self.assertGreaterEqual(top[i, c, 0] + 1e-4,
                                        exp(bottom[i, c, 0]) / scale)
                self.assertLessEqual(top[i, c, 0] - 1e-4,
                                     exp(bottom[i, c, 0]) / scale)



