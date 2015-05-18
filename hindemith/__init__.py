from hindemith.types import hmarray
import numpy as np


def ones(shape, dtype=np.float32):
    return np.ones(shape, dtype).view(hmarray)


def zeros(shape, dtype=np.float32):
    return np.zeros(shape, dtype).view(hmarray)


def zeros_like(arr):
    return np.zeros_like(arr).view(hmarray)


def random(shape, _range=(0, 1), dtype=np.float32):
    rand = np.random.rand(*shape).astype(dtype)
    length = _range[1] - _range[0]
    rand *= length
    rand += _range[0]
    return rand.view(hmarray)
