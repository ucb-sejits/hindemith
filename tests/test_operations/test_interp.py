from .base_test import HMBaseTest
from hindemith.operations.interp import interp_linear
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
            # if xx > f1.shape[1] - 5 or yy > f1.shape[0] - 5 or xx < 1 or yy < 1:
            #     output[i, j] = 0
            # else:
            #     val1 = im[yy-1,xx-1 + 1] + 0.5 * ty*(im[yy-1,xx-1 + 2] - im[yy-1,xx-1] + ty*(2.0*im[yy-1,xx-1] - 5.0*im[yy-1,xx-1 + 1] + 4.0*im[yy-1,xx-1 + 2] - im[yy-1,xx-1 + 3] + ty*(3.0*(im[yy-1,xx-1 + 1] - im[yy-1,xx-1 + 2]) + im[yy-1,xx-1 + 3] - im[yy-1,xx-1])));
            #
            #     val2 = im[yy,xx-1 + 1] + 0.5 * ty*(im[yy,xx-1 + 2] - im[yy,xx-1] + ty*(2.0*im[yy,xx-1] - 5.0*im[yy,xx-1 + 1] + 4.0*im[yy,xx-1 + 2] - im[yy,xx-1 + 3] + ty*(3.0*(im[yy,xx-1 + 1] - im[yy,xx-1 + 2]) + im[yy,xx-1 + 3] - im[yy,xx-1])));
            #
            #     val3 = im[yy+1,xx-1 + 1] + 0.5 * ty*(im[yy+1,xx-1 + 2] - im[yy+1,xx-1] + ty*(2.0*im[yy+1,xx-1] - 5.0*im[yy+1,xx-1 + 1] + 4.0*im[yy+1,xx-1 + 2] - im[yy+1,xx-1 + 3] + ty*(3.0*(im[yy+1,xx-1 + 1] - im[yy+1,xx-1 + 2]) + im[yy+1,xx-1 + 3] - im[yy+1,xx-1])));
            #
            #     val4 = im[yy+2,xx-1 + 1] + 0.5 * ty*(im[yy+2,xx-1 + 2] - im[yy+2,xx-1] + ty*(2.0*im[yy+2,xx-1] - 5.0*im[yy+2,xx-1 + 1] + 4.0*im[yy+2,xx-1 + 2] - im[yy+2,xx-1 + 3] + ty*(3.0*(im[yy+2,xx-1 + 1] - im[yy+2,xx-1 + 2]) + im[yy+2,xx-1 + 3] - im[yy+2,xx-1])));
            #     output[i, j] = val2 + 0.5 * tx*(val3 - val1 + tx*(2.0*val1 - 5.0*val2 + 4.0*val3 - val4 + tx*(3.0*(val2 - val3) + val4 - val1)));
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
        a = np.random.rand(128, 64).astype(np.float32) * 255
        b = np.random.rand(128, 64).astype(np.float32) * 3
        c = np.random.rand(128, 64).astype(np.float32) * 3
        actual = interp_linear(hmarray(a), hmarray(b), hmarray(c))
        expected = remap(a, b, c)
        actual.copy_to_host_if_dirty()
        actual = np.copy(actual)
        np.testing.assert_array_almost_equal(actual, expected, decimal=1)
