from stencil_code.stencil_kernel import StencilKernel

__author__ = 'chick'

import numpy as np


class ApplyBroxPreconditioner(object):
    @staticmethod
    def kernel(stencil, diag0, diag1, off_diag, src0, src1):
        dst0 = np.empty_like(src0)
        dst1 = np.empty_like(src1)

        for i in range(src0.shape[0]):
            for j in range(src0.shape[1]):
                a = diag0[i, j] + stencil[2, 0, 0]
                d = diag1[i, j] + stencil[2, 0, 0]

                det = a * d - off_diag[i, j] * off_diag[i, j]

                if abs(det) > 1.0e-4:
                    dst0[i, j] = (src0[i, j] * d - off_diag[i, j] * src1[i, j]) / det
                    dst1[i, j] = (-off_diag[i, j] * src0[i, j] + a * src1[i, j]) / det
                else:
                    dst0[i, j] = src0[i, j]
                    dst1[i, j] = src1[i, j]

        return dst0, dst1

if __name__ == '__main__':
    shape = [6, 4]
    s0 = np.random.random(shape)
    s1 = np.random.random(shape)
    d0 = np.random.random(shape)
    d1 = np.random.random(shape)
    od = np.random.random(shape)
    s = np.random.random([3, 1, 1])

    out0, out1 = ApplyBroxPreconditioner.kernel(s, d0, d1, od, s0, s1)

    print out0
