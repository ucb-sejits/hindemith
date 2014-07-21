# from hindemith.hlib import *
from hindemith.operations.optical_flow.warp_img2D import warp_img2d
from optical_flow_solver import *
from stencil_code.stencil_kernel import StencilKernel
from numpy import *


@fuse
def HornSchunckJacobiSolver(im1_data, im2_data, D, Gx, Gy, u, v, zero, one, lam2, num_iter):
    du = zero * u
    dv = zero * v

    tex_Ix0 = Gx * im1_data
    tex_Iy0 = Gy * im1_data
    tex_Ix = Gx * im2_data
    tex_Iy = Gy * im2_data
    tex_Ixx = Gx * tex_Ix
    tex_Iyy = Gy * tex_Iy
    tex_Ixy = Gx * tex_Iy
    Ix = warp_img2d(tex_Ix, u, v)
    Iy = warp_img2d(tex_Iy, u, v)
    It = im1_data - warp_img2d(im2_data, u, v)
    Ixy = warp_img2d(tex_Ixy, u, v)
    Ix2 = warp_img2d(tex_Ixx, u, v)
    Iy2 = warp_img2d(tex_Iyy, u, v)
    Ix2 = Ix * Ix
    IxIy = Ix * Iy
    Iy2 = Iy * Iy

    #TODO: Chaneg num_iter.value
    for i in range(num_iter.value):
        ubar = D * du
        vbar = D * dv
        num = Ix * ubar + Iy * vbar + It
        den = Ix2 + Iy2 + lam2
        # den = (Ix*Ix) + (Iy*Iy) + lam2
        du = ubar - (Ix * num) / den
        dv = vbar - (Iy * num) / den
    return du, dv


class HornSchunckJacobi(OpticalFlowSolver):
    def __init__(self, lam, num_iterations):
        # self.num_iter = ScalarConstant(num_iterations)
        # self.lam2 = Float32(lam*lam)
        # self.zero = Float32(0.0)
        # self.one = Float32(1.0)
        self.num_iter = num_iterations
        self.lam2 = lam * lam
        self.zero = 0.0
        self.one = 1.0
        class S(StencilKernel):
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

        Sdata = array([1.0 / 12.0, 8.0 / 12.0, -8.0 / 12.0, -1.0 / 12.0], dtype=float32)
        Soffx = array([-2, -1, 1, 2], dtype=int32)
        Soffy = array([0, 0, 0, 0], dtype=int32)
        of = 1.0 / 4.0
        # Adata = array([of, of, of, of], dtype=float32)
        #Aoffx = array([0,-1,1,0], dtype=int32)
        #Aoffy = array([-1,0,0,1], dtype=int32)
        Adata = array(
            [1.0 / 12.0, 2.0 / 12.0, 1.0 / 12.0, 2.0 / 12.0, 2.0 / 12.0, 1.0 / 12.0, 2.0 / 12.0,
             1.0 / 12.0], dtype=float32)
        Aoffx = array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=int32)
        Aoffy = array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=int32)
        class A(StencilKernel):
            def kernel(self, input, output):
                for x in input.interior_points():
                    for y in input.neighbors(x, 0):
                        output[x] = input[y] * .083333333
                    for y in input.neighbors(x, 1):
                        output[x] = input[y] * .333333333
        self.A = Stencil(Adata, Aoffx, Aoffy)
        self.A.kernel = A(backend='omp').kernel
        self.A.neighbor_points = [[(-1, -1), (1, -1), (1, 0), (0, 1)],
                                  [(0, -1), (-1, 0), (-1, 1), (1, 1)]]
        self.Gx = Stencil(Sdata, Soffx, Soffy)
        self.Gx.kernel = S(backend='omp').kernel
        self.Gx.neighbor_points = [[(-2, 0)], [(-1, 0)], [(1, 0)], [(2, 0)]]
        self.Gy = Stencil(Sdata, Soffy, Soffx)
        self.Gy.kernel = S(backend='omp').kernel
        self.Gy.neighbor_points = [[(-2, 0)], [(-1, 0)], [(1, 0)], [(2, 0)]]
        self.bytes_transferred = 0
        self.theoretical_bytes_transferred = 0
        self.flops = 0

    def run(self, im1, im2, u, v):
        import os

        HM_CPU0 = os.environ.get('HM_CPU0', None)
        HM_GPU0 = os.environ.get('HM_GPU0', None)
        HM_GPU01 = os.environ.get('HM_GPU01', None)
        HM_GPU1 = os.environ.get('HM_GPU1', None)

        if HM_GPU0 or HM_GPU1 or HM_GPU01:
            defines['nd20'] = 1
            defines['nd21'] = 1
            defines['nt20'] = 128
            defines['nt21'] = 1
            defines['ne20'] = 1
            defines['ne21'] = 1
        else:
            defines['nd20'] = 1
            defines['nd21'] = 1
            defines['nt20'] = 32
            defines['nt21'] = 32
            defines['ne20'] = 1
            defines['ne21'] = 1
            # defines['nd20'] = 1
            #defines['nd21'] = 1
            #defines['nt20'] = 1
            #defines['nt21'] = 2
            #defines['ne20'] = 64
            #defines['ne21'] = 1

        self.bytes_transferred += self.num_iter * im1.shape[0] * im1.shape[1] * 4 * 11
        self.theoretical_bytes_transferred += self.num_iter * im1.shape[0] * im1.shape[1] * 4 * 7
        self.flops += self.num_iter * im1.shape[0] * im1.shape[1] * 30
        return HornSchunckJacobiSolver(im1_data=im1, im2_data=im2, D=self.A, Gx=self.Gx, Gy=self.Gy,
                                       u=u, v=v, zero=self.zero, one=self.one, lam2=self.lam2,
                                       num_iter=self.num_iter)

