import numpy as np
import cv2
from scipy.ndimage.filters import convolve

alpha = 15.0

jacobi = np.array([
    [.0833, .1677, .0833],
    [.1677, 0, .1677],
    [.0833, .1677, .0833],
])


def hs_jacobi(im0, im1):
    u = np.zeros_like(im0)
    v = np.zeros_like(im0)
    It = im1 - im0
    Iy, Ix = np.gradient(im1)
    denom = np.square(Ix) + np.square(Iy) + alpha ** 2
    epsilon = .01 ** 2 * np.prod(im0.shape)

    for _ in range(100):
        ubar = convolve(jacobi, u)
        vbar = convolve(jacobi, v)
        t = (Ix * ubar + Iy * vbar + It) / denom
        u_new = ubar - Ix * t
        v_new = vbar - Iy * t
        err = np.square(u_new - u) + np.square(v_new - v)
        if np.sum(err) < epsilon:
            break
        u, v = u_new, v_new
    return u, v


frame0 = cv2.imread('images/frame0.png')
frame1 = cv2.imread('images/frame1.png')
frame0 = cv2.resize(frame0, (384, 288))
frame1 = cv2.resize(frame1, (384, 288))
im0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
im1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

u = hs_jacobi(im0, im1)
mag, ang = cv2.cartToPolar(u[0], u[1])
mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
ang = ang*180/np.pi/2
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
hsv[..., 0] = ang
hsv[..., 2] = mag
flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('flow', flow)
cv2.waitKey()
