import cv2
import numpy as np
from hindemith.types.hmarray import hmarray, indices, zeros
from hindemith.operations.interp import interp_linear


def pyr_down(m, n_scales, n):
    pyr = [hmarray(m)]
    for _ in range(n_scales - 1):
        scaled = tuple(s * n for s in m.shape)
        y, x = indices(scaled)
        curr = pyr[-1]
        y = y * (curr.shape[0] / scaled[0])
        x = x * (curr.shape[1] / scaled[1])
        pyr.append(interp_linear(curr, x, y))
    return pyr


def pyr_up(m, shape):
    y, x = indices(shape)
    y = y * (m.shape[0] / shape[0])
    x = x * (m.shape[1] / shape[1])
    return interp_linear(m, x, y)


def np_pyr_down(m, n_scales, n):
    pyr = [m]
    for _ in range(n_scales - 1):
        scaled = tuple(s * n for s in m.shape)
        y, x = np.indices(scaled).astype(np.float32)
        curr = pyr[-1]
        y = y * (curr.shape[0] / scaled[0])
        x = x * (curr.shape[1] / scaled[1])
        pyr.append(cv2.remap(curr, x, y, cv2.INTER_LINEAR))
    return pyr


def np_pyr_up(m, shape):
    y, x = np.indices(shape).astype(np.float32)
    y = y * (m.shape[0] / shape[0])
    x = x * (m.shape[1] / shape[1])
    return cv2.remap(m, x, y, cv2.INTER_LINEAR)


class Solver(object):
    def __init__(self, num_scales, scale_factor):
        """
        :param int num_scales: The number of scales at which to run the solver
        :param float scale_factor: The scaling factor for each level.
        """
        self.num_scales = num_scales
        self.scale_factor = scale_factor

    def solve(self, im0, im1, u, v):
        raise NotImplementedError("Solver subclass must implement solve")

    def __call__(self, im0, im1):
        im0 = cv2.GaussianBlur(im0, (5, 5), .7)
        im1 = cv2.GaussianBlur(im1, (5, 5), .7)
        im0 = im0.astype(np.float32)
        im1 = im1.astype(np.float32)
        im0_pyr = pyr_down(im0, self.num_scales, self.scale_factor)
        im1_pyr = pyr_down(im1, self.num_scales, self.scale_factor)
        u1 = zeros(im0_pyr[-1].shape, np.float32)
        u2 = zeros(im0_pyr[-1].shape, np.float32)
        for s in reversed(range(self.num_scales)):
            u1, u2 = self.solve(im0_pyr[s], im1_pyr[s], u1, u2)
            if s > 0:
                next_size = im0_pyr[s - 1].shape
                u1 = pyr_up(u1, next_size) * (1.0 / self.scale_factor)
                u2 = pyr_up(u2, next_size) * (1.0 / self.scale_factor)
        u1.copy_to_host_if_dirty()
        u2.copy_to_host_if_dirty()
        return u1, u2


class NumpySolver(Solver):
    def __call__(self, im0, im1):
        im0 = cv2.GaussianBlur(im0, (5, 5), .7)
        im1 = cv2.GaussianBlur(im1, (5, 5), .7)
        im0 = im0.astype(np.float32)
        im1 = im1.astype(np.float32)
        im0_pyr = np_pyr_down(im0, self.num_scales, self.scale_factor)
        im1_pyr = np_pyr_down(im1, self.num_scales, self.scale_factor)
        u1 = np.zeros(im0_pyr[-1].shape, np.float32)
        u2 = np.zeros(im0_pyr[-1].shape, np.float32)
        for s in reversed(range(self.num_scales)):
            u1, u2 = self.solve(im0_pyr[s], im1_pyr[s], u1, u2)
            if s > 0:
                next_size = im0_pyr[s - 1].shape
                u1 = np_pyr_up(u1, next_size) * (1.0 / self.scale_factor)
                u2 = np_pyr_up(u2, next_size) * (1.0 / self.scale_factor)
        return u1, u2
