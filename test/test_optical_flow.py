import unittest
from stencil_code.stencil_grid import StencilGrid
from stencil_code.stencil_kernel import StencilKernel

from hindemith.core import fuse
from hindemith.utils import unique_name
from hindemith.operations.optical_flow.pyr_down import pyr_down
from hindemith.operations.optical_flow.pyr_up import pyr_up
from hindemith.operations.optical_flow.warp_img2D import warp_img2d
from hindemith.operations.dense_linear_algebra import Array, square, Float32

from numpy import random, float32, zeros, testing

__author__ = 'leonardtruong'

class Stencil1(StencilKernel):
    def __init__(self, backend='ocl'):
        super(Stencil1, self).__init__(backend)
        self.neighbor_definitions = [[(-2, 0)], [(-1, 0)], [(1, 0)], [(2, 0)]]

    def kernel(self, input, output):
        for x in input.interior_points():
            for y in input.neighbors(x, 0):
                output[x] = input[y] * .083333333
            for y in input.neighbors(x, 1):
                output[x] = input[y] * .666666667
            for y in input.neighbors(x, 2):
                output[x] = input[y] * -.666666667
            for y in input.neighbors(x, 3):
                output[x] = input[y] * -.083333333

stencil1 = Stencil1().kernel


class Stencil2(StencilKernel):
    def __init__(self, backend='ocl'):
        self.neighbor_definitions = [[(-1, -1), (1, -1), (1, 0), (0, 1)],
                                    [(0, -1), (-1, 0), (-1, 1), (1, 1)]]
        super(Stencil2, self).__init__(backend)

    def kernel(self, input, output):
        for x in input.interior_points():
            for y in input.neighbors(x, 0):
                output[x] = input[y] * .083333333
            for y in input.neighbors(x, 1):
                output[x] = input[y] * .333333333

stencil2 = Stencil2().kernel

def pyr_down_fn(im):
    output =  pyr_down(Array(unique_name(), im))
    return output.data

@fuse
def hs_solver_jacobi(im1_data, im2_data, u, v, zero, one, lam2, num_iter):
    du = zero * u
    dv = zero * v

    # tex_Ix0 = stencil1(im1_data)
    # tex_Iy0 = stencil1(im1_data)
    tex_Ix = stencil1(im2_data)
    tex_Iy = stencil1(im2_data)
    # tex_Ixx = stencil1(tex_Ix)
    # tex_Iyy = stencil1(tex_Iy)
    # tex_Ixy = stencil1(tex_Iy)
    Ix = warp_img2d(tex_Ix, u, v)
    Iy = warp_img2d(tex_Iy, u, v)
    It = im1_data - warp_img2d(im2_data, u, v)
    # Ixy = warp_img2d(tex_Ixy, u, v)
    # Ix2 = warp_img2d(tex_Ixx, u, v)
    # Iy2 = warp_img2d(tex_Iyy, u, v)
    # Ix2 = square(Ix)
    # IxIy = Ix * Iy
    # Iy2 = square(Ix)

    for i in range(num_iter.value):
        ubar = stencil2(du)
        vbar = stencil2(dv)
        num = Ix * ubar + Iy * vbar + It
        # den = Ix2 * Iy2 + lam2
        # den = lam2 + Ix2 * Iy2
        # du = ubar - Ix * num / 4.0
        # dv = vbar - Iy * num / 4.0
        du = ubar - Float32('a', .25) * num * Ix
        dv = vbar - Float32('b', .25) * num * Iy
    return du, dv


def py_hs_solver_jacobi(im1_data, im2_data, u, v, zero, one, lam2, num_iter):
    du = zero * u
    dv = zero * v

    # tex_Ix0 = stencil1(im1_data)
    # tex_Iy0 = stencil1(im1_data)
    tex_Ix = stencil1(im2_data)
    tex_Iy = stencil1(im2_data)
    # tex_Ixx = stencil1(tex_Ix)
    # tex_Iyy = stencil1(tex_Iy)
    # tex_Ixy = stencil1(tex_Iy)
    u = Array('u', v)
    v = Array('v', v)
    Ix = warp_img2d(Array('Ix', tex_Ix), u, v).data
    Iy = warp_img2d(Array('Iy', tex_Iy), u, v).data
    It = im1_data - warp_img2d(Array('It', im2_data), u, v).data
    # Ixy = warp_img2d(Array('Ixy', tex_Ixy), u, v).data
    # Ix2 = warp_img2d(Array('Ix2', tex_Ixx), u, v).data
    # Iy2 = warp_img2d(Array('Iy2', tex_Iyy), u, v).data
    # Ix2 = Ix * Ix
    # IxIy = Ix * Iy
    # Iy2 = Ix * Ix

    for i in range(num_iter):
        ubar = stencil2(du)
        vbar = stencil2(dv)
        num = Ix * ubar + Iy * vbar + It
        # den = Ix2 * Iy2 + lam2
        # den = lam2 + Ix2 * Iy2
        du = ubar - Ix * num / 4.0
        dv = vbar - Iy * num / 4.0
    return du, dv


@fuse
def update_uv(u, v, du, dv, two, w, h):
    u = u + du
    v = v + dv
    # u = u * two
    # v = v * two
    u = two * u
    v = two * v
    # u = median_filter(u)
    # v = median_filter(v)
    new_u = pyr_up(u)
    new_v = pyr_up(v)
    return new_u, new_v


def py_update_uv(u, v, du, dv, two, w, h):
    u = u + du
    v = v + dv
    # u = u * two
    # v = v * two
    u = two * u
    v = two * v
    # u = median_filter(u)
    # v = median_filter(v)
    new_u = pyr_up(Array('u', u)).data
    new_v = pyr_up(Array('v', v)).data
    return new_u, new_v


class TestOpticalFlow(unittest.TestCase):
    def test_optical_flow(self):
        height = 120
        width = 160
        im1 = random.rand(height, width).astype(float32)
        im2 = random.rand(height, width).astype(float32)
        u = zeros((height, width), dtype=float32)
        v = zeros((height, width), dtype=float32)
        num_pyr = 2
        # num_full = 1

        pyr1 = [im1]
        pyr2 = [im2]
        sizes = [(im1.shape[0], im1.shape[1])]

        for i in range(1, num_pyr + 1):
            pyr1.insert(0, pyr_down_fn(im=pyr1[0]))
            pyr2.insert(0, pyr_down_fn(im=pyr2[0]))
            sizes.insert(0, (pyr1[0].shape[0], pyr1[0].shape[1]))

        hm_u = zeros(sizes[0], dtype=u.dtype)
        hm_v = zeros(sizes[0], dtype=v.dtype)
        py_hm_u = zeros(sizes[0], dtype=u.dtype)
        py_hm_v = zeros(sizes[0], dtype=v.dtype)

        two = 2.0

        num_iter = 10
        alpha = .1
        lam2 = alpha ** 2
        zero = 0.0
        one = 1.0

        for i in range(0, num_pyr):
            hm_du, hm_dv = hs_solver_jacobi(
                im1_data=pyr1[i], im2_data=pyr2[i], u=hm_u, v=hm_v, zero=zero,
                one=one, lam2=lam2, num_iter=num_iter
            )
            py_hm_du, py_hm_dv = py_hs_solver_jacobi(
                im1_data=pyr1[i], im2_data=pyr2[i], u=py_hm_u, v=py_hm_v,
                zero=zero, one=one, lam2=lam2, num_iter=num_iter
            )
            hm_u, hm_v = update_uv(
                u=hm_u, v=hm_v, du=hm_du, dv=hm_dv, two=two,
                w=sizes[i + 1][0], h=sizes[i+1][1]
            )
            py_hm_u, py_hm_v = py_update_uv(
                u=py_hm_u, v=py_hm_v, du=py_hm_du, dv=py_hm_dv, two=two,
                w=sizes[i + 1][0], h=sizes[i+1][1]
            )

        testing.assert_array_almost_equal(hm_u.data, py_hm_u)
        testing.assert_array_almost_equal(hm_v.data, py_hm_v)
