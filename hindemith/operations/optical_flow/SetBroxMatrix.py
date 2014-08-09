import numpy as np
# pragma: no cover
class SetBroxMatrix(object):
    def __init__(self, pure_python=False):
        self.pure_python = pure_python
    def __call__(self, A, Psi, Alpha):
        if self.pure_python:
            return self.pythonFunc(A, Psi, Alpha)
    def pythonFunc(self,A ,psi,alpha):
        diags = A.shape[0]
        dim1 = A.shape[1]
        dim2 = A.shape[2]
        for x in xrange(dim1):
            for y in xrange(dim2):
                temp = 0
                val = 0
                psiVal = alpha * psi[x][y]
                if x > 0:
                    val = alpha * psi[x-1][y]
                    A[0][x-1][y] = -val
                    temp += val
                if y > 0:
                    val = alpha * psi[x][y-1]
                    A[1][x][y-1] = - val
                    temp += val
                if x < dim1 - 1:
                    A[4][x+1][y] = - psiVal
                    temp += psiVal
                if y < dim2 - 1:
                    A[3][x][y+1] = - psiVal
                    temp += psiVal
                A[2][x][y] = temp
