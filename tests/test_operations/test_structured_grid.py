from .base_test import HMBaseTest
import numpy as np

from hindemith.operations.structured_grid import structured_grid
from hindemith.types.hmarray import hmarray


def py_divergence(v1, v2):
    div = np.zeros_like(v1)
    div[1:, 1:] = v2[1:, 1:] - v2[:-1, 1:] + v1[1:, 1:] - v1[1:, :-1]
    div[1:, 0] = v2[1:, 0] - v2[:-1, 0] + v1[1:, 0]
    div[0, 1:] = v2[0, 1:] + v1[0, 1:] - v1[0, :-1]
    div[0, 0] = v1[0, 0] + v2[0, 0]
    return div


@structured_grid(border='zero')
def divergence(v1, v2, output):
    for y, x in output:
        output[y, x] = v1[y, x] + v2[y, x] - v1[y, x - 1] - v2[y - 1, x]


def py_dx(m):
    dx = np.zeros_like(m)
    dx[..., :-1] = m[..., 1:] - m[..., :-1]
    dx[-1, ...] = -m[-1, ...]
    return dx


def py_dy(m):
    dy = np.zeros_like(m)
    dy[:-1, ...] = m[1:, ...] - m[:-1, ...]
    dy[..., -1] = -m[..., -1]
    return dy


@structured_grid(border='zero')
def dx(src, output):
    for y, x in output:
        output[y, x] = src[y, x + 1] - src[y, x]


@structured_grid(border='zero')
def dy(src, output):
    for y, x in output:
        output[y, x] = src[y + 1, x] - src[y, x]


class TestStructuredGrid(HMBaseTest):
    def test_divergence(self):
        a, b = self.a, self.b
        hm_a, hm_b = hmarray(a), hmarray(b)
        actual = divergence(hm_a, hm_b)
        expected = py_divergence(a, b)
        try:
            actual.copy_to_host_if_dirty()
            actual = np.copy(actual)
            np.testing.assert_array_almost_equal(actual[1:, 1:],
                                                 expected[1:, 1:],
                                                 decimal=3)
        except AssertionError as e:
            self.fail(e)

    def test_gradient(self):
        a = self.a[:4, :4]
        hm_a = hmarray(a)
        actual = dx(hm_a)
        expected = py_dx(a)
        try:
            actual.copy_to_host_if_dirty()
            actual = np.copy(actual)
            np.testing.assert_array_almost_equal(actual[:-1, :-1],
                                                 expected[:-1, :-1],
                                                 decimal=3)
        except AssertionError as e:
            self.fail(e)

        actual = dy(hm_a)
        expected = py_dy(a)
        try:
            actual.copy_to_host_if_dirty()
            actual = np.copy(actual)
            np.testing.assert_array_almost_equal(actual[:-1, :-1],
                                                 expected[:-1, :-1],
                                                 decimal=3)
        except AssertionError as e:
            self.fail(e)
