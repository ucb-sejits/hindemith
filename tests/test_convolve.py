__author__ = 'leonardtruong'

import unittest
import hindemith as hm
from hindemith.types import hmarray
from hindemith.core import compose
from hindemith.operations.convolve import Convolve2D
import numpy as np


def convolve(data, filters):
    output = np.zeros_like(data)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            for i in range(filters.shape[0]):
                for j in range(filters.shape[1]):
                    in_y = min(max(y + i - (filters.shape[0] // 2), 0), data.shape[0] - 1)
                    in_x = min(max(x + j - (filters.shape[1] // 2), 0), data.shape[1] - 1)
                    output[y, x] += data[in_y, in_x] * filters[i, j]
    return output


@unittest.skip("Broken")
class TestConvolve(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_convolve(self):
        data = hm.random((48, 48), _range=(0, 1))
        filters = hm.random((3, 3), _range=(-1, 1))
        output = hmarray.zeros_like(data)

        @compose
        def fn(data, filters, output):
            output = Convolve2D(data, filters)
            return output

        output = fn(data, filters, output)
        expected = convolve(data, filters)
        output.sync_host()
        # np.testing.assert_array_almost_equal(output[2:-2, 2:-2], expected[2:-2, 2:-2], decimal=4)
        np.testing.assert_array_almost_equal(output, expected, decimal=4)

