import unittest

from hindemith.operations.optical_flow.apply_hs_preconditioner import \
    apply_hs_preconditioner

import numpy as np


class TestApplyHSPreconditioner(unittest.TestCase):
    def test_simple(self):
        Ix2 = np.random.rand(120, 80).astype(np.float32) * 256
        Iy2 = np.random.rand(120, 80).astype(np.float32) * 256
        IxIy = np.random.rand(120, 80).astype(np.float32) * 256
        r0 = np.random.rand(120, 80).astype(np.float32) * 256
        r1 = np.random.rand(120, 80).astype(np.float32) * 256

        class Obj(object):
            def __init__(self, data):
                self.data = data

        data = np.array([1.0/12.0, 8.0/12.0, -8.0/12.0, -1.0/12.0],
                        dtype=np.float32)
        D = Obj(data)
        result = apply_hs_preconditioner(D, Ix2, Iy2, IxIy, r0, r1)
        print(result)
