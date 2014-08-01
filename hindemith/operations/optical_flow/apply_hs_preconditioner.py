__author__ = 'Leonard Truong'

from numpy import zeros


def apply_hs_preconditioner(D, diag0, diag1, offdiag, x0, x1):
    height, width = x0.shape
    dtype = x0.dtype
    a = zeros((height, width), dtype=dtype)
    b = zeros((height, width), dtype=dtype)
    c = zeros((height, width), dtype=dtype)
    d = zeros((height, width), dtype=dtype)
    a += diag0
    d += diag1
    b += offdiag
    c += offdiag
    a[1:, :] += D.data[2]
    d[1:, :] += D.data[2]
    a[:-1, :] += D.data[2]
    d[:-1, :] += D.data[2]
    a[:, 1:] += D.data[2]
    d[:, 1:] += D.data[2]
    a[:, :-1] += D.data[2]
    d[:, :-1] += D.data[2]

    det = (a*d) - (c*b)
    mask = abs(det) > 1e-4
    g = (x0 * d - b * x1) / det
    h = (-c * x0 + a * x1) / det
    y0 = 1.0 * x0
    y1 = 1.0 * x1
    y0[mask] = g[mask]
    y1[mask] = h[mask]

    return y0, y1
