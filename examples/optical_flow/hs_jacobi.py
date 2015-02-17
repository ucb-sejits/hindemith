import numpy as np
# import logging
# logging.basicConfig(level=20)

from hindemith.operations.reduce import sum
from hindemith.meta.core import meta
from hindemith.operations.map import square, copy
from solver import Solver

from stencil_code.stencil_kernel import Stencil


class Jacobi(Stencil):
    neighborhoods = [[(0, -1), (0, 1), (-1, 0), (1, 0)],
                     [(-1, -1), (-1, 1), (1, -1), (1, 1)]]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            out_grid[x] = 0.0
            for y in self.neighbors(x, 0):
                out_grid[x] += .166666667 * in_grid[y]
            for y in self.neighbors(x, 1):
                out_grid[x] += .083333333 * in_grid[y]


class Dx(Stencil):
    neighborhoods = [
        [(1, 0), (1, 1)],
        [(0, 0), (0, 1)]]

    def kernel(self, a, b, out_grid):
        for x in self.interior_points(out_grid):
            out_grid[x] = 0.0
            for y in self.neighbors(x, 0):
                out_grid[x] += .25 * (a[y] + b[y])
            for y in self.neighbors(x, 1):
                out_grid[x] -= .25 * (a[y] + b[y])


class Dy(Stencil):
    neighborhoods = [
        [(0, 1), (1, 1)],
        [(0, 0), (1, 0)]]

    def kernel(self, a, b, out_grid):
        for x in self.interior_points(out_grid):
            out_grid[x] = 0.0
            for y in self.neighbors(x, 0):
                out_grid[x] += .25 * (a[y] + b[y])
            for y in self.neighbors(x, 1):
                out_grid[x] -= .25 * (a[y] + b[y])


class Dt(Stencil):
    neighborhoods = [
        [(0, 0), (1, 0), (0, 1), (1, 1)]]

    def kernel(self, a, b, out_grid):
        for x in self.interior_points(out_grid):
            for y in self.neighbors(x, 0):
                out_grid[x] += .25 * (b[y] - a[y])

dx, dy, dt = Dx(), Dy(), Dt()

jacobi = Jacobi()


@meta
def update(u, v, Ix, Iy, It, denom):
    ubar = jacobi(u)
    vbar = jacobi(v)
    t = (Ix * ubar + Iy * vbar + It) / denom
    u_new = ubar - Ix * t
    v_new = vbar - Iy * t
    err = square(u_new - u) + square(v_new - v)
    return u_new, v_new, err

alpha = 15.0
alpha2 = alpha ** 2


@meta
def compute_denom(Ix, Iy):
    return square(Ix) + square(Iy) + alpha2


@meta
def gradient_and_denom(im0, im1):
    It = im1 - im0
    Ix = dx(im0, im1)
    Iy = dy(im0, im1)
    denom = square(Ix) + square(Iy) + alpha2
    return Ix, Iy, It, denom


class HS_Jacobi(Solver):
    def solve(self, im0, im1, u, v):
        Ix, Iy, It, denom = gradient_and_denom(im0, im1)
        epsilon = (0.04 ** 2) * np.prod(u.shape)

        for _ in range(100):
            u, v, err = update(u, v, Ix, Iy, It, denom)
            if sum(err) < epsilon:
                break
        return u, v


def main():
    import cv2
    frame0 = cv2.imread('images/frame0.png')
    frame1 = cv2.imread('images/frame1.png')
    frame0 = cv2.resize(frame0, (384, 288))
    frame1 = cv2.resize(frame1, (384, 288))
    im0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    im1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hs_jacobi = HS_Jacobi(1, .5)

    from ctree.util import Timer
    hs_jacobi(im0, im1)
    with Timer() as t:
        u = hs_jacobi(im0, im1)
    print(t.interval)
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

if __name__ == '__main__':
    import cProfile
    cProfile.run('main()')
