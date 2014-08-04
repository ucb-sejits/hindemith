import unittest
import numpy
from numpy import testing

import ast

from ctree.frontend import get_ast

from stencil_code.stencil_grid import StencilGrid
from stencil_code.stencil_kernel import StencilKernel

from hindemith.fusion.core import fuse, BlockBuilder, get_blocks, Fuser
from hindemith.types.common import Array
from hindemith.utils import unique_name
from hindemith.operations.optical_flow.warp_img2D import warp_img2d
from hindemith.operations.dense_linear_algebra.array_op import square
from hindemith.operations.dense_linear_algebra.core import array_mul, \
    array_sub, scalar_array_mul


class TestFuser(unittest.TestCase):
    def test_is_fusable_true(self):
        def f(a, b):
            c = array_mul(a, b)
            d = array_sub(a, c)
        tree = get_ast(f)
        blocks = []
        BlockBuilder(blocks).visit(tree)
        fuser = Fuser(blocks, dict(locals(), **globals()))
        self.assertTrue(fuser._is_fusable(blocks[0], blocks[1]))

    def test_is_fusable_false(self):
        def f(a, b):
            c = array_mul(a, b)
            return c
        tree = get_ast(f)
        blocks = []
        BlockBuilder(blocks).visit(tree)
        fuser = Fuser(blocks, dict(locals(), **globals()))
        self.assertFalse(fuser._is_fusable(blocks[0], blocks[1]))

    def test_fuse(self):
        a = numpy.random.rand(100, 100).astype(numpy.float32) * 100
        b = numpy.random.rand(100, 100).astype(numpy.float32) * 100
        def f(a, b):
            c = array_mul(a, b)
            d = array_sub(a, c)
        tree = get_ast(f)
        blocks = []
        BlockBuilder(blocks).visit(tree)
        fuser = Fuser(blocks, dict(locals(), **globals()))
        actual_c, actual_d = fuser._fuse([blocks[0], blocks[1]])
        try:
            numpy.testing.assert_array_almost_equal(actual_d, a - a * b)
        except Exception as e:
            self.fail("Arrays not almost equal: {0}".format(e.message))

class TestBlockBuilder(unittest.TestCase):
    def test_simple(self):
        def f(a):
            return array_mul(a)
        tree = get_ast(f)
        blocks = []
        BlockBuilder(blocks).visit(tree)
        self.assertEqual(len(blocks), 1)
        self.assertIsInstance(blocks[0], ast.Return)

    def test_multiline(self):
        def f(a, b):
            c = array_mul(a, b)
            d = array_sub(a, c)
            e = d * c
            return e / 4
        tree = get_ast(f)
        blocks = []
        BlockBuilder(blocks).visit(tree)
        self.assertEqual(len(blocks), 4)
        self.assertIsInstance(blocks[0], ast.Assign)
        self.assertIsInstance(blocks[1], ast.Assign)
        self.assertIsInstance(blocks[2], ast.Assign)
        self.assertIsInstance(blocks[3], ast.Return)


class TestSimpleFusion(unittest.TestCase):
    @unittest.skip("Not implemented")
    def test_simple(self):
        a = numpy.random.rand(100, 100).astype(numpy.float32) * 100
        b = numpy.random.rand(100, 100).astype(numpy.float32) * 100
        def f(a, b):
            c = array_mul(a, b)
            d = scalar_array_mul(4, c)
            return d
        tree = get_ast(f)
        blocks = get_blocks(tree)
        fuser = Fuser(blocks, dict(locals(), **globals()))
        fuser.do_fusion()
        self.assertEqual(len(blocks), 2)



# class TestDecorator(unittest.TestCase):
#     def test_dec(self):
#         @fuse(locals(), globals())
#         def test_func(arg=None):
#             return arg

#         a = test_func(arg=1)
#         self.assertEqual(a, 1)

#     def test_fusion(self):
#         @fuse(locals(), globals())
#         def test_func(A=None, B=None, C=None):
#             D = array_mul(A, B)
#             E = array_sub(C, D)
#             return E

#         A = numpy.random.rand(60, 60).astype(numpy.float32)
#         B = numpy.random.rand(60, 60).astype(numpy.float32)
#         C = numpy.random.rand(60, 60).astype(numpy.float32)
#         actual = test_func(A=A, B=B, C=C)
#         expected = C - (A * B)
#         try:
#             testing.assert_array_almost_equal(actual, expected)
#         except AssertionError as e:
#             self.fail("Outputs not equal: %s" % e.message)

    # @unittest.skip("")
    # def test_hs_jacobi(self):
    #     @fuse
    #     def hs_jacobi_solver(im1_data, im2_data, u, v, zero, lam2, num_iter):
    #         du = zero * u
    #         dv = zero * v

    #         im2_data = StencilGrid(im2_data.shape, data=im2_data.data)
    #         for index, defn in enumerate(
    #                 [[(-2, 0)], [(-1, 0)], [(1, 0)], [(2, 0)]]):
    #             im2_data.set_neighborhood(index, defn)
    #         tex_Ix = stencil_a(im2_data)
    #         tex_Iy = stencil_a(im2_data)
    #         im2_data = im2.data
    #         tex_Ix = Array(unique_name(), tex_Ix.data)
    #         tex_Iy = Array(unique_name(), tex_Iy.data)
    #         Ix = warp_img2d(tex_Ix, u, v)
    #         Iy = warp_img2d(tex_Iy, u, v)
    #         It = im1_data - warp_img2d(im2_data, u, v)
    #         Ix2 = square(Ix)
    #         Iy2 = square(Iy)

    #         for i in range(num_iter.value):
    #             du = StencilGrid(du.shape, data=du.data)
    #             dv = StencilGrid(dv.shape, data=dv.data)
    #             for index, defn in enumerate(
    #                 [[(-1, -1), (1, -1), (1, 0), (0, 1)],
    #                  [(0, -1), (-1, 0), (-1, 1), (1, 1)]]):
    #                 du.set_neighborhood(index, defn)
    #                 dv.set_neighborhood(index, defn)
    #             ubar = stencil_b(du)
    #             ubar = Array(unique_name(), ubar.data)
    #             vbar = stencil_b(dv)
    #             vbar = Array(unique_name(), vbar.data)
    #             num = Ix * ubar + Iy * vbar + It
    #             den = Ix2 + Iy2
    #             den = lam2 + den
    #             du = ubar - (Ix * num) / den
    #             dv = vbar - (Iy * num) / den
    #         return du, dv

    #     def py_hs_jacobi_solver(im1_data, im2_data, u, v, zero, lam2, num_iter):
    #         du = zero * u
    #         dv = zero * v

    #         im2_data = StencilGrid(im2_data.shape, data=im2_data)
    #         for index, defn in enumerate(
    #                 [[(-2, 0)], [(-1, 0)], [(1, 0)], [(2, 0)]]):
    #             im2_data.set_neighborhood(index, defn)
    #         tex_Ix = stencil_a(im2_data).data
    #         tex_Iy = stencil_a(im2_data).data
    #         im2_data = Array(unique_name(), im2_data.data)
    #         u = Array(unique_name(), u)
    #         v = Array(unique_name(), v)
    #         Ix = warp_img2d(Array(unique_name(), tex_Ix), u, v).data
    #         Iy = warp_img2d(Array(unique_name(), tex_Iy), u, v).data
    #         It = im1_data - warp_img2d(im2_data, u, v).data
    #         Ix2 = Ix * Ix
    #         Iy2 = Iy * Iy

    #         for i in range(num_iter):
    #             du = StencilGrid(du.shape, data=du)
    #             dv = StencilGrid(dv.shape, data=dv)
    #             for index, defn in enumerate(
    #                 [[(-1, -1), (1, -1), (1, 0), (0, 1)],
    #                  [(0, -1), (-1, 0), (-1, 1), (1, 1)]]):
    #                 du.set_neighborhood(index, defn)
    #                 dv.set_neighborhood(index, defn)
    #             ubar = stencil_b(du).data
    #             vbar = stencil_b(dv).data
    #             num = Ix * ubar + Iy * vbar + It
    #             den = Ix2 + Iy2 + lam2
    #             du = ubar - (Ix * num) / den
    #             dv = vbar - (Iy * num) / den
    #         return du, dv

    #     im1 = numpy.random.rand(60, 40).astype(numpy.float32)
    #     im2 = numpy.random.rand(60, 40).astype(numpy.float32)
    #     u = numpy.zeros((60, 40), numpy.float32)
    #     v = numpy.zeros((60, 40), numpy.float32)
    #     zero = 0.0
    #     alpha = 0.1
    #     lam2 = alpha ** 2
    #     num_iter = 10
    #     du, dv = hs_jacobi_solver(
    #         im1_data=im1, im2_data=im2, u=u, v=v, zero=zero, lam2=lam2,
    #         num_iter=num_iter
    #     )

    #     py_du, py_dv = py_hs_jacobi_solver(
    #         im1_data=im1, im2_data=im2, u=u, v=v, zero=zero, lam2=lam2,
    #         num_iter=num_iter
    #     )

    #     testing.assert_array_almost_equal(du.data, py_du)
    #     testing.assert_array_almost_equal(dv.data, py_dv)



# class StencilA(StencilKernel):
#     def kernel(self, input, output):
#         for x in input.interior_points():
#             for y in input.neighbors(x, 0):
#                 output[x] = input[y] * .083333333
#             for y in input.neighbors(x, 1):
#                 output[x] = input[y] * .666666667
#             for y in input.neighbors(x, 2):
#                 output[x] = input[y] * -.666666667
#             for y in input.neighbors(x, 3):
#                 output[x] = input[y] * -.083333333


# class StencilB(StencilKernel):
#     def kernel(self, input, output):
#         for x in input.interior_points():
#             for y in input.neighbors(x, 0):
#                 output[x] = input[y] * .083333333
#             for y in input.neighbors(x, 1):
#                 output[x] = input[y] * .333333333

# stencil_a = StencilA(backend='c').kernel
# stencil_b = StencilB(backend='c').kernel
