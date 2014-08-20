import unittest
import numpy
from numpy import testing

import ast

from ctree.frontend import get_ast

from stencil_code.stencil_kernel import StencilKernel

from hindemith.fusion.core import BlockBuilder, get_blocks, Fuser, fuse
# from hindemith.types.common import Array
# from hindemith.utils import unique_name
# from hindemith.operations.optical_flow.warp_img2D import warp_img2d
# from hindemith.operations.dense_linear_algebra.array_op import square
from hindemith.operations.dense_linear_algebra.core import array_mul, \
    array_sub, scalar_array_mul, array_add, array_scalar_add

import logging

log = logging.getLogger('ctree')
log.setLevel = 10


class TestFuser(unittest.TestCase):
    def test_is_fusable_true(self):
        def f(a, b):
            c = array_mul(a, b)
            d = array_sub(a, c)
            return d

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

    def test_fuse_2(self):
        a = numpy.random.rand(100, 100).astype(numpy.float32) * 100
        b = numpy.random.rand(100, 100).astype(numpy.float32) * 100

        def f(a, b):
            c = array_mul(a, b)
            d = array_sub(a, c)
            return d

        tree = get_ast(f)
        blocks = []
        BlockBuilder(blocks).visit(tree)
        fuser = Fuser(blocks, dict(locals(), **globals()))
        fused = fuser._fuse([blocks[0], blocks[1]]).value
        actual_c, actual_d = fuser._symbol_table[fused.func.id](
            *(fuser._symbol_table[arg.id] if isinstance(arg, ast.Name) else
              arg.n for arg in fused.args)
        )
        expected_c = a * b
        expected_d = a - expected_c
        try:
            testing.assert_array_almost_equal(actual_c, expected_c)
            testing.assert_array_almost_equal(actual_d, expected_d)
        except Exception as e:
            self.fail("Arrays not almost equal: {0}".format(e))

    def test_fuse_3(self):
        a = numpy.random.rand(100, 100).astype(numpy.float32) * 100
        b = numpy.random.rand(100, 100).astype(numpy.float32) * 100

        def f(a, b):
            c = array_mul(a, b)
            d = array_sub(a, c)
            e = array_add(c, d)
            return e

        tree = get_ast(f)
        blocks = []
        BlockBuilder(blocks).visit(tree)
        fuser = Fuser(blocks, dict(locals(), **globals()))
        fused = fuser._fuse([blocks[0], blocks[1], blocks[2]]).value
        actual_c, actual_d, actual_e = fuser._symbol_table[fused.func.id](
            *(fuser._symbol_table[arg.id] if isinstance(arg, ast.Name) else
              arg.n for arg in fused.args)
        )
        expected_c = a * b
        expected_d = a - expected_c
        expected_e = expected_c + expected_d
        try:
            testing.assert_array_almost_equal(actual_c, expected_c)
            testing.assert_array_almost_equal(actual_d, expected_d)
            testing.assert_array_almost_equal(actual_e, expected_e)
        except Exception as e:
            self.fail("Arrays not almost equal: {0}".format(e))


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
    def test_simple(self):
        a = numpy.random.rand(100, 100).astype(numpy.float32) * 100
        b = numpy.random.rand(100, 100).astype(numpy.float32) * 100

        def f(a, b):
            c = array_mul(a, b)
            d = scalar_array_mul(4, c)
            return d

        orig_f = f
        tree = get_ast(f)
        blocks = get_blocks(tree)
        fuser = Fuser(blocks, dict(locals(), **globals()))
        fused_blocks = fuser.do_fusion()
        tree.body[0].body = fused_blocks
        tree = ast.fix_missing_locations(tree)
        exec(compile(tree, '', 'exec')) in fuser._symbol_table
        try:
            testing.assert_array_almost_equal(fuser._symbol_table['f'](a, b),
                                              orig_f(a, b))
        except Exception as e:
            self.fail("Arrays not almost equal: {0}".format(e))

    def test_simple2(self):
        a = numpy.random.rand(100, 100).astype(numpy.float32) * 100
        b = numpy.random.rand(100, 100).astype(numpy.float32) * 100

        def f(a, b):
            c = array_mul(a, b)
            d = scalar_array_mul(4, c)
            e = array_sub(d, b)
            return e

        orig_f = f
        tree = get_ast(f)
        blocks = get_blocks(tree)
        fuser = Fuser(blocks, dict(locals(), **globals()))
        fused_blocks = fuser.do_fusion()
        tree.body[0].body = fused_blocks
        tree = ast.fix_missing_locations(tree)
        exec(compile(tree, '', 'exec')) in fuser._symbol_table
        try:
            testing.assert_array_almost_equal(fuser._symbol_table['f'](a, b),
                                              orig_f(a, b))
        except Exception as e:
            self.fail("Arrays not almost equal: {0}".format(e))

    def test_non_fusable(self):
        a = numpy.random.rand(100, 100).astype(numpy.float32) * 100
        b = numpy.random.rand(100, 100).astype(numpy.float32) * 100

        def f(a, b):
            c = array_mul(a, b)
            d = scalar_array_mul(4, c)
            b = d * 3
            e = array_sub(d, b)
            return e

        orig_f = f
        tree = get_ast(f)
        blocks = get_blocks(tree)
        fuser = Fuser(blocks, dict(locals(), **globals()))
        fused_blocks = fuser.do_fusion()
        tree.body[0].body = fused_blocks
        tree = ast.fix_missing_locations(tree)
        exec(compile(tree, '', 'exec')) in fuser._symbol_table
        try:
            testing.assert_array_almost_equal(fuser._symbol_table['f'](a, b),
                                              orig_f(a, b))
        except Exception as e:
            self.fail("Arrays not almost equal: {0}".format(e))

    def test_fuse_with_return(self):
        a = numpy.random.rand(100, 100).astype(numpy.float32) * 100
        b = numpy.random.rand(100, 100).astype(numpy.float32) * 100

        def f(a, b):
            c = array_mul(a, b)
            d = scalar_array_mul(4, c)
            return array_sub(c, d)

        orig_f = f
        tree = get_ast(f)
        blocks = get_blocks(tree)
        fuser = Fuser(blocks, dict(locals(), **globals()))
        fused_blocks = fuser.do_fusion()
        tree.body[0].body = fused_blocks
        tree = ast.fix_missing_locations(tree)
        exec(compile(tree, '', 'exec')) in fuser._symbol_table
        try:
            testing.assert_array_almost_equal(fuser._symbol_table['f'](a, b),
                                              orig_f(a, b))
        except Exception as e:
            self.fail("Arrays not almost equal: {0}".format(e))


stdev_d = 3
stdev_s = 70
radius = 1
width = 64 + radius * 2
height = width


class Stencil(StencilKernel):
    @property
    def dim(self):
        return 2

    @property
    def ghost_depth(self):
        return 1

    def neighbors(self, pt, defn=0):
        if defn == 0:
            for x in range(-radius, radius+1):
                for y in range(-radius, radius+1):
                    yield (pt[0] - x, pt[1] - y)

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]


class TestDecorator(unittest.TestCase):
    def _check(self, actual, expected):
        try:
            testing.assert_array_almost_equal(actual, expected)
        except AssertionError as e:
            self.fail("Outputs not equal: %s" % e)

    def test_dec_no_fusion(self):
        @fuse
        def test_func(arg):
            return arg

        a = test_func(1)
        self.assertEqual(a, 1)

    def test_fusion(self):
        A = numpy.random.rand(60, 60).astype(numpy.float32)
        B = numpy.random.rand(60, 60).astype(numpy.float32)
        C = numpy.random.rand(60, 60).astype(numpy.float32)

        @fuse
        def test_func(A, B, C):
            D = array_mul(A, B)
            E = array_sub(C, D)
            return E

        actual = test_func(A, B, C)
        expected = C - (A * B)
        self._check(actual, expected)

    def test_non_fusable(self):
        A = numpy.random.rand(60, 60).astype(numpy.float32)
        B = numpy.random.rand(60, 60).astype(numpy.float32)
        C = numpy.random.rand(60, 60).astype(numpy.float32)

        @fuse
        def test_func(A, B, C):
            D = array_mul(A, B)
            E = array_sub(C, D)
            F = (C * 495) / 394
            return array_add(E, F)

        actual = test_func(A, B, C)
        expected = (C - (A * B)) + ((C * 495) / 394)
        self._check(actual, expected)

    def test_fuse_with_return(self):
        A = numpy.random.rand(60, 60).astype(numpy.float32)
        B = numpy.random.rand(60, 60).astype(numpy.float32)
        C = numpy.random.rand(60, 60).astype(numpy.float32)

        @fuse
        def test_func(A, B, C):
            D = array_mul(A, B)
            E = array_sub(C, D)
            return array_scalar_add(E, 3)

        actual = test_func(A, B, C)
        expected = (C - A * B) + 3
        self._check(actual, expected)

    def test_fusing_stencil_array_op(self):
        in_grid = numpy.random.rand(1024, 1024).astype(numpy.float32) * 100
        kernel1 = Stencil(backend='ocl')
        kernel2 = Stencil(backend='ocl')

        @fuse
        def f(in_grid):
            out = kernel1(in_grid)
            return array_scalar_add(out, 4)

        actual = f(in_grid)
        expected = kernel2(in_grid) + 4
        self._check(actual, expected)

    @unittest.skip("Not implemented yet")
    def test_fusing_stencils(self):
        in_grid = (numpy.random.rand(16, 16) * 10).astype(numpy.int32)
        kernel1 = Stencil(backend='ocl')
        kernel2 = Stencil(backend='ocl')
        py_kernel1 = Stencil(backend='python')
        py_kernel2 = Stencil(backend='python')

        @fuse
        def f(in_grid):
            a = kernel1(in_grid)
            return kernel2(a)

        actual = f(in_grid)
        expected = py_kernel2(py_kernel1(in_grid))
        self._check(actual, expected)

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

    #     def py_hs_jacobi_solver(im1_data, im2_data, u, v, zero, lam2,
    #                             num_iter):
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
