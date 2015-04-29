from hindemith.core import compose
from hindemith.types import hmarray
import numpy as np
import cv2
import sys


num_warps = 2
n_scales = 2
n_inner = 10
n_outer = 10
median_filtering = 5
theta = .3
tau = .25
l = .15  # lambda
epsilon = 0.01
n = .8


def py_th(rho_elt, gradient_elt, delta_elt, u_elt):
    thresh = l * theta * gradient_elt
    if rho_elt < -thresh:
        return l * theta * delta_elt + u_elt
    elif rho_elt > thresh:
        return -l * theta * delta_elt + u_elt
    elif gradient_elt > 1e-10:
        return -rho_elt / gradient_elt * delta_elt + u_elt
    else:
        return 0


py_thresh = np.vectorize(py_th, otypes=[np.float32])


def py_threshold(u1, u2, rho_c, gradient, I1wx, I1wy):
    rho = rho_c + I1wx * u1 + I1wy * u2
    v1 = py_thresh(rho, gradient, I1wx, u1)
    v2 = py_thresh(rho, gradient, I1wy, u2)
    return v1, v2


def py_forward_gradient(m):
    dx, dy = np.zeros_like(m), np.zeros_like(m)
    dy[:-1, ...] = m[1:, ...] - m[:-1, ...]
    dx[..., :-1] = m[..., 1:] - m[..., :-1]
    return dx, dy


def py_divergence(v1, v2):
    div = np.zeros_like(v1)
    div[1:, 1:] = v2[1:, 1:] - v2[:-1, 1:] + v1[1:, 1:] - v1[1:, :-1]
    div[1:, 0] = v2[1:, 0] - v2[:-1, 0] + v1[1:, 0]
    div[0, 1:] = v2[0, 1:] + v1[0, 1:] - v1[0, :-1]
    # div[0, 0] = v1[0, 0] + v2[0, 0]
    return div


def centered_gradient(m):
    return np.gradient(m)


def build_flow_map(idxs, u1, u2):
    _x = idxs[1].__add__(u1)
    _y = idxs[0].__add__(u2)
    return _x, _y


def cl_build_flow_map(xs, ys, u1, u2):
    _x = xs + u1
    _y = ys + u2
    return _x, _y


def warp(im, f1, f2):
    return cv2.remap(im, f1, f2, cv2.INTER_LINEAR)


def py_flow(I0, I1):
    u1 = hmarray.zeros(I1.shape)
    u2 = hmarray.zeros(I1.shape)
    p11 = hmarray.zeros(I1.shape)
    p12 = hmarray.zeros(I1.shape)
    p21 = hmarray.zeros(I1.shape)
    p22 = hmarray.zeros(I1.shape)
    i1y, i1x = centered_gradient(I1)
    i1x = i1x.astype(np.float32).view(hmarray)
    i1y = i1y.astype(np.float32).view(hmarray)
    indices = np.indices(u1.shape).astype(np.float32)
    for w in range(num_warps):
        _f1, _f2 = build_flow_map(indices, u1, u2)
        i1w = warp(I1, _f1, _f2)
        i1wx = warp(i1x, _f1, _f2)
        i1wy = warp(i1y, _f1, _f2)
        grad = np.square(i1wx) + np.square(i1wy)
        rho_c = i1w - i1wx * u1 - i1wy * u2 - I0
        n0 = 0
        error = sys.maxint
        while n0 < n_outer and error > epsilon * epsilon * I0.size:
            # u1 = cv2.medianBlur(u1, median_filtering)
            # u2 = cv2.medianBlur(u2, median_filtering)
            n1 = 0
            while n1 < n_inner and error > epsilon * epsilon * I0.size:
                v1, v2 = py_threshold(u1, u2, rho_c, grad, i1wx, i1wy)
                div_p1 = py_divergence(p11, p12)
                div_p2 = py_divergence(p21, p22)
                u1_old = u1
                u2_old = u2
                u1 = v1 + div_p1 * theta
                u2 = v2 + div_p2 * theta
                error = np.sum(np.square(u1 - u1_old) + np.square(u2 - u2_old))
                u1x, u1y = py_forward_gradient(u1)
                u2x, u2y = py_forward_gradient(u2)
                ng1 = 1.0 + tau / theta * np.sqrt(np.square(u1x) +
                                                  np.square(u1y))
                ng2 = 1.0 + tau / theta * np.sqrt(np.square(u2x) +
                                                  np.square(u2y))
                p11 = (p11 + tau / theta * u1x) / ng1
                p12 = (p12 + tau / theta * u1y) / ng1
                p21 = (p21 + tau / theta * u2x) / ng2
                p22 = (p22 + tau / theta * u2y) / ng2
                n1 += 1
            n0 += 1
    return u1, u2


cap = cv2.VideoCapture("ir.mp4")
ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(prev)
hsv[..., 1] = 255

ret, frame = cap.read()
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
u, v = py_flow(prev_gray.view(hmarray), frame_gray.view(hmarray))
prev_gray = frame_gray
mag, ang = cv2.cartToPolar(u, v)
hsv[..., 0] = ang*180/np.pi/2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('frame2', rgb)
k = cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
