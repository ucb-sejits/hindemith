import unittest
import numpy as np

from hindemith.operations.optical_flow.set_brox_redblack_matrix import \
    set_brox_redblack_matrix


class TestSetBroxRedblackMatrix(unittest.TestCase):
    spmat = np.random.rand(120, 60, 5).astype(np.float32) * 100
    psi = np.random.rand(120, 60).astype(np.float32) * 100
    alpha = .1
    result = set_brox_redblack_matrix(spmat, psi, alpha)
    print("Result:", result)
    print("spmat:", spmat)
