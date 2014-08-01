import numpy as np


def set_brox_redblack_matrix(spmat, psi, alpha):
    Adiag = np.zeros_like(psi)
    for x in range(1, psi.shape[0] - 1):
        for y in range(1, psi.shape[1] - 1):
            spmat[(x, y - 1, 0)] = -alpha * psi[(x, y - 1)]
            Adiag[(x, y)] += alpha * psi[(x, y - 1)]
            spmat[(x - 1, y, 1)] = -alpha * psi[(x - 1, y)]
            Adiag[(x, y)] += alpha * psi[(x - 1, y)]
            spmat[(x + 1, y, 2)] = -alpha * psi[(x, y)]
            Adiag[(x, y)] += alpha * psi[(x, y)]
            spmat[(x, y + 1, 3)] = -alpha * psi[(x, y)]
            Adiag[(x, y)] += alpha * psi[(x, y)]
    return Adiag
