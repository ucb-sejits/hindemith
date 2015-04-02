import numpy as np
import pycl as cl
from hindemith.cl import queue, context
import ctree.c.nodes as C
from ctree.types import get_c_type_from_numpy_dtype


class Matrix(object):
    def __init__(self, shape, dtype, ndarray=None):
        if ndarray is None:
            self.data = np.ndarray(shape, dtype)
            self.ocl_buf = cl.clCreateBuffer(
                context, np.prod(shape) * self.data.itemsize)
        else:
            self.data = ndarray
            buf, evt = cl.buffer_from_ndarray(queue, self.data)
            self.ocl_buf = buf
        self.shape = shape
        self.dtype = dtype
        self.host_dirty = False
        self.ocl_dirty = False
        self.register = None

    def get_element(self, name):
        if self.register is not None:
            return self.register
        else:
            return "{}[get_global_id(0)]".format(name)

    def sync(self):
        if self.host_dirty:
            _, evt = cl.buffer_to_ndarray(queue, self.ocl_buf, self.data)
            evt.wait()
            self.host_dirty = False

    def promote_to_register(self, name):
        if self.register is None:
            self.register = name
            ptr = get_c_type_from_numpy_dtype(self.data.dtype)()
            return C.SymbolRef(name, ptr)
        return None

    @staticmethod
    def rand(shape, dtype):
        data = np.random.rand(*shape).astype(dtype) * 255
        return Matrix(shape, dtype, ndarray=data)
