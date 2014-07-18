__author__ = 'leonardtruong'

from stencil_code.stencil_grid import StencilGrid
from ..operations.dense_linear_algebra.array_op import Array
from ..utils import unique_name


class Stencil(object):
    def __init__(self, data, offx, offy):
        self.data = data
        self.offx = offx
        self.offy = offy
        self.dtype = data.dtype
        self.uniqueName = unique_name()

    def __mul__(self, other):
        if isinstance(other, Array):
            input = StencilGrid(other.data.shape)
            for index, defn in enumerate(self.neighbor_points):
                input.set_neighborhood(index, defn)
            input.data = other.data
            output = StencilGrid(other.data.shape, dtype=other.data.dtype)
            self.kernel(input, output)
            return Array(unique_name(), output.data)
        print(type(other))
        raise NotImplementedError()
