import unittest
import numpy as np
from hindemith.operations.matrix_powers import MatrixPowers

from numpy import *


# pragma: no cover
class TestMatrixPowers(unittest.TestCase):
    def test_pure_python(self):
        a = np.ones([5, 5])
        b = np.zeros([10, 5, 5])
        mp = MatrixPowers(pure_python=True)
        # mp = MatrixPowers()
        mp(a, b)
        print b
        np.testing.assert_array_almost_equal(a, b[0])

    def test_ocl(self):
        a = np.ones([5, 5])
        b = np.zeros([10, 5, 5])
        mp = MatrixPowers()
        mp(a, b)
        print b
        np.testing.assert_array_almost_equal(a, b[0])

# @hm
# def mpow_test(A,x,s):
#   for i in range(1):
#     W = mpow(A,x,s)
#
#    # p = A * x - b
#    # r = A * x - b
#
#     # for inner in range(n_outer):
#
#     # P = mpow(A, p, s)
#     # R = mpow(A, r, s)
#     # PP = tssyrk(P, s)
#     # RR = tssyrk(R, s)
#     # PR = tsgemm(P, R, s)
#
#     # x_c, r_c, p_c = cacg_coef(PP, RR, PR, s)
#     # x, p, r = cacg_update(x_c, r_c, p_c, s)
#
#     #G = tssyrk(W,s)
#     G = tsgemm(W, W, s)
#   return W, G
#
# defines['nd20'] = 1
# defines['nd21'] = 1
# defines['nt20'] = 8
# defines['nt21'] = 8
# defines['ne20'] = 2
# defines['ne21'] = 2
#
# dim = 1024
# x = Array2D((dim, dim), dtype=float32)
# x[:] = 1
# #random.seed(5)
# #x[:] = random.rand(dim,dim)[:]
#
# Adata = array([0.25, 0.25, 0.25, 0.25], dtype=float32)
# Aoffx = array([0,-1,1,0], dtype=int32)
# Aoffy = array([-1,0,0,1], dtype=int32)
# A = Stencil(Adata, Aoffx, Aoffy)
#
# import time
# for i in range(1):
#   ts = time.time()
#   res, g = mpow_test(A=A,x=x,s=ScalarConstant(8))
#   runtime = float64(time.time() - ts)
#   gbytes_transferred = float64(4 * dim * dim * 3 * 1) / float64(1024*1024*1024)
#   print 'Runtime: ' + str(runtime) + '\tGBytes/s: ' + str(gbytes_transferred / runtime)
#
# sync(res)
# sync(g)
# print g
# print res
# print res.shape
#
# # Generate real krylov subspace
# y1 = A*x
# y2 = A*y1
# y3 = A*y2
# y4 = A*y3
# y5 = A*y4
#
# import pdb; pdb.set_trace()
