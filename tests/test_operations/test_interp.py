from .base_test import HMBaseTest
from hindemith.operations.interp import LinearInterp
from hindemith.types.hmarray import hmarray
import numpy as np


def remap(im, f1, f2):
    output = np.zeros_like(f1)
    for i in range(f1.shape[0]):
        for j in range(f2.shape[1]):
            x = f1[i, j]
            y = f2[i, j]
            xx = int(x)
            yy = int(y)
            tx = x - xx
            ty = y - yy
            if xx > f1.shape[1] - 2 or yy > f1.shape[0] - 2:
                output[i, j] = 0
            else:
                output[i, j] = im[yy, xx] * (1 - tx) * (1 - ty) + \
                    im[yy, xx + 1] * tx * (1 - ty) + \
                    im[yy + 1, xx] * (1 - tx) * ty + \
                    im[yy + 1, xx + 1] * tx * ty
    return output


class TestInterp(HMBaseTest):
    def test_simple(self):
        interp = LinearInterp(None)
        a = np.random.rand(640, 480).astype(np.float32) * 255
        b = np.random.rand(640, 480).astype(np.float32) * 3
        c = np.random.rand(640, 480).astype(np.float32) * 3
        actual = interp(hmarray(a), hmarray(b), hmarray(c))
        expected = remap(a, b, c)
        actual.copy_to_host_if_dirty()
        actual = np.copy(actual)
        np.testing.assert_array_almost_equal(actual, expected, decimal=1)
