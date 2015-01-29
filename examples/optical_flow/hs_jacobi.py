from solver import Solver
import numpy as np
import cv2
from hindemith.types.hmarray import EltWiseArrayOp, zeros, indices
from hindemith.operations.map import SpecializedMap
from hindemith.operations.structured_grid import structured_grid
from hindemith.operations.interp import interp_linear
# import logging
# logging.basicConfig(level=20)
from ctree.util import Timer


EltWiseArrayOp.backend = 'ocl'
SpecializedMap.backend = 'ocl'


def cl_build_flow_map(xs, ys, u1, u2):
    _x = xs + u1
    _y = ys + u2
    return _x, _y


@structured_grid(border='zero')
def dx(src, output):
    for y, x in output:
        output[y, x] = src[y, x + 1] - src[y, x]


@structured_grid(border='zero')
def dy(src, output):
    for y, x in output:
        output[y, x] = src[y + 1, x] - src[y, x]


@structured_grid(border='zero')
def D(src, output):
    for y, x in output:
        output[y, x] = -float(.0025) * (
            src[y - 1, x] + src[y, x - 1] +
            src[y, x + 1] + src[y + 1, x]) + \
            float(.01) * src[y, x]


class HSJacobi(Solver):
    def solve(self, i0, i1, u1, u2):
        du = zeros(i0.shape, np.float32)
        dv = zeros(i0.shape, np.float32)

        tex_Ix = dx(i1)
        tex_Iy = dy(i1)
        ys, xs = indices(u1.shape)
        _f1, _f2 = cl_build_flow_map(xs, ys, u1, u2)
        Ix = interp_linear(tex_Ix, _f1, _f2)
        Iy = interp_linear(tex_Iy, _f1, _f2)
        It = i0 - interp_linear(i1, _f1, _f2)
        Ix2 = Ix * Ix
        IxIy = Ix * Iy
        Iy2 = Iy * Iy
        b0 = -1 * Ix * It
        b1 = -1 * Iy * It

        Dinv0 = 1 / (.01 + Ix2)
        Dinv1 = 1 / (.01 + Iy2)
        for i in range(100):
            du_resid = b0 - (D(du) + IxIy * dv)  # b-R*x(k)
            dv_resid = b1 - (IxIy * du + D(dv))  # b-R*x(k)
            du = Dinv0 * du_resid
            dv = Dinv1 * dv_resid
        return du, dv


import os
file_path = os.path.dirname(os.path.realpath(__file__))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action='store_true', help="profile execution time")
    args = parser.parse_args()
    frame0 = cv2.imread('images/frame0.png')
    frame1 = cv2.imread('images/frame1.png')
    im0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    im1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    tvl1 = HSJacobi(2, .5)
    # jit warmup
    for _ in range(5):
        tvl1(im0, im1)
    if args.profile:
        import cProfile
        cProfile.runctx('tvl1(im0, im1)', None, locals())
    else:
        with Timer() as t:
            u = tvl1(im0, im1)
        print("Specialized time: {}s".format(t.interval))
        mag, ang = cv2.cartToPolar(u[0], u[1])
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        ang = ang*180/np.pi/2
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        hsv[..., 0] = ang
        hsv[..., 2] = mag
        flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite('flow.jpg', flow)
        # cv2.imshow('flow', flow)
        # cv2.waitKey(0) & 0xff
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
