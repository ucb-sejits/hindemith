import numpy as np
import sys
import cv2
from hindemith.operations.zip_with import zip_with, ZipWith
from hindemith.types import hmarray
from hindemith.utils import symbols
# import logging
# logging.basicConfig(level=20)


from hindemith.types.hmarray import EltWiseArrayOp

EltWiseArrayOp.backend = 'ocl'
ZipWith.backend = 'ocl'


num_warps = 5
n_scales = 5
n_inner = 30
n_outer = 10
median_filtering = 5
theta = .3
tau = .25
l = .15  # lambda
epsilon = 0.01
n = .8

symbol_table = {
    'num_warps': num_warps,
    'n_scales': n_scales,
    'theta': theta,
    'tau': tau,
    'l': l,
    'epsilon': epsilon,
    'n': n
}


@symbols(symbol_table)
def th(rho_elt, gradient_elt, delta_elt, u_elt):
    threshold = l * theta * gradient_elt
    if rho_elt < -threshold:
        return l * theta * delta_elt + u_elt
    elif rho_elt > threshold:
        return -l * theta * delta_elt + u_elt
    elif gradient_elt > 1e-10:
        return -rho_elt / gradient_elt * delta_elt + u_elt
    else:
        return 0


@symbols(symbol_table)
def ocl_th(rho_elt, gradient_elt, delta_elt, u_elt):
    threshold = float(l * theta) * gradient_elt
    if rho_elt < -threshold:
        return float(l * theta) * delta_elt + u_elt
    elif rho_elt > threshold:
        return float(-l * theta) * delta_elt + u_elt
    elif gradient_elt > 1e-10:
        return -rho_elt / gradient_elt * delta_elt + u_elt
    else:
        return float(0)


def threshold(u1, u2, rho_c, gradient, I1wx, I1wy):
    rho_c = hmarray(rho_c)
    gradient = hmarray(gradient)
    I1wx = hmarray(I1wx)
    I1wy = hmarray(I1wy)
    u1 = hmarray(u1)
    u2 = hmarray(u2)
    rho = rho_c + I1wx * u1 + I1wy * u2
    v1 = zip_with(ocl_th, rho, gradient, I1wx, u1)
    v2 = zip_with(ocl_th, rho, gradient, I1wy, u2)
    return v1, v2
    # v1 = zip_with(th, hmarray(rho), hmarray(gradient),
    #               hmarray(I1wx), hmarray(u[..., 0]))
    # v2 = zip_with(th, hmarray(rho), hmarray(gradient),
    #               hmarray(I1wy), hmarray(u[..., 1]))
    # v1.copy_to_host_if_dirty()
    # v2.copy_to_host_if_dirty()
    # return np.dstack((v1.view(np.ndarray), v2.view(np.ndarray)))


def centered_gradient(m):
    return np.gradient(m)


def forward_gradient(m):
    dx, dy = np.zeros_like(m), np.zeros_like(m)
    dy[:-1, ...] = m[1:, ...] - m[:-1, ...]
    dx[..., :-1] = m[..., 1:] - m[..., :-1]
    dy[..., -1] = 0.0
    dx[-1, ...] = 0.0
    return dx, dy


def divergence(v1, v2):
    div = np.zeros_like(v1)
    div[1:, 1:] = v2[1:, 1:] - v2[:-1, 1:] + v1[1:, 1:] - v1[1:, :-1]
    div[1:, 0] = v2[1:, 0] - v2[:-1, 0] + v1[1:, 0]
    div[0, 1:] = v2[0, 1:] + v1[0, 1:] - v1[0, :-1]
    div[0, 0] = v1[0, 0] + v2[0, 0]
    return div


def pyr_down(m, n_scales, n):
    pyr = [m]
    for _ in range(n_scales - 1):
        pyr.append(cv2.resize(pyr[-1], None, fx=n, fy=n))
    return pyr


def pyr_up(m, shape):
    return cv2.resize(m, shape)


def build_flow_map(u1, u2):
    idxs = np.indices(u1.shape).astype(np.float32)
    _x = idxs[1] + u1
    _y = idxs[0] + u2
    return _x, _y


def warp(im, f1, f2):
    return cv2.remap(im, f1, f2, cv2.INTER_CUBIC)


def compute_flow(I0, I1, u1, u2):
    p11 = np.zeros(I1.shape, dtype=np.float32)
    p12 = np.zeros(I1.shape, dtype=np.float32)
    p21 = np.zeros(I1.shape, dtype=np.float32)
    p22 = np.zeros(I1.shape, dtype=np.float32)
    I1y, I1x = centered_gradient(I1)
    I1x = I1x.astype(np.float32)
    I1y = I1y.astype(np.float32)

    for w in range(num_warps):
        _f1, _f2 = build_flow_map(u1, u2)
        I1w = warp(I1, _f1, _f2)
        I1wx = warp(I1x, _f1, _f2)
        I1wy = warp(I1y, _f1, _f2)
        grad = (np.square(I1wx) + np.square(I1wy))
        rho_c = I1w - I1wx * u1 - I1wy * u2 - I0
        n0 = 0
        error = sys.maxint
        while n0 < n_outer and error > epsilon * epsilon * I0.size:
            u1 = cv2.medianBlur(u1, median_filtering)
            u2 = cv2.medianBlur(u2, median_filtering)
            n1 = 0
            while n1 < n_inner and error > epsilon * epsilon * I0.size:
                v1, v2 = threshold(u1, u2, rho_c, grad, I1wx, I1wy)
                div_p1 = hmarray(divergence(p11, p12))
                div_p2 = hmarray(divergence(p21, p22))
                u1_old = u1
                u2_old = u2
                u1 = v1 + div_p1 * theta
                u2 = v2 + div_p2 * theta
                u1.copy_to_host_if_dirty()
                u2.copy_to_host_if_dirty()
                u1 = np.copy(u1)
                u2 = np.copy(u2)
                error = np.sum(np.square(u1 - u1_old) + np.square(u2 - u2_old))
                u1x, u1y = forward_gradient(u1)
                u2x, u2y = forward_gradient(u2)
                hypot1 = np.sqrt(np.square(u1x) + np.square(u1y))
                hypot2 = np.sqrt(np.square(u2x) + np.square(u2y))
                p11 = (p11 + u1x * tau / theta) / \
                      (1.0 + hypot1 * tau / theta)
                p12 = (p12 + u1y * tau / theta) / \
                      (1.0 + hypot1 * tau / theta)
                p21 = (p21 + u2x * tau / theta) / \
                      (1.0 + hypot2 * tau / theta)
                p22 = (p22 + u2y * tau / theta) / \
                      (1.0 + hypot2 * tau / theta)
                n1 += 1
            n0 += 1
    return u1, u2


def tvl1(im0, im1):
    im0 = im0.astype(np.float32)
    im1 = im1.astype(np.float32)
    # im0 = (cv2.GaussianBlur(im0))
    # im1 = (cv2.GaussianBlur(im1))
    im0_pyr = pyr_down(im0, n_scales, n)
    im1_pyr = pyr_down(im1, n_scales, n)
    u1 = np.zeros(im0_pyr[-1].shape, dtype=np.float32)
    u2 = np.zeros(im0_pyr[-1].shape, dtype=np.float32)
    for s in reversed(range(n_scales)):
        u1, u2, = compute_flow(im0_pyr[s], im1_pyr[s], u1, u2)
        if s > 0:
            u1 = pyr_up(u1, im0_pyr[s - 1].shape[::-1]) * (1.0 / n)
            u2 = pyr_up(u2, im0_pyr[s - 1].shape[::-1]) * (1.0 / n)
    return u1, u2

import os
file_path = os.path.dirname(os.path.realpath(__file__))

frame0 = cv2.imread(file_path + '/frame0.png')
frame1 = cv2.imread(file_path + '/frame1.png')
im0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
im1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# tvl1(im0, im1)
# import cProfile
# cProfile.run('tvl1(im0, im1)')
# exit()
u1, u2 = tvl1(im0, im1)
mag, ang = cv2.cartToPolar(u1, u2)
hsv = np.zeros_like(frame0)
hsv[..., 1] = 255
hsv[..., 0] = ang*180/np.pi/2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('frame1', rgb)

# Baseline
# hsv2 = np.zeros_like(frame0)
# flow = cv2.calcOpticalFlowFarneback(im0, im1, 0.5, 5, 15, 3, 5, 1.2, 0)
# mag2, ang2 = cv2.cartToPolar(flow[..., 0], flow[..., 1])
# hsv2[..., 1] = 255
# hsv2[..., 0] = ang2*180/np.pi/2
# hsv2[..., 2] = cv2.normalize(mag2, None, 0, 255, cv2.NORM_MINMAX)
# rgb2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
# cv2.imshow('frame2', rgb2)
k = cv2.waitKey() & 0xff
