import numpy as np
import pycl as cl
from hindemith.cl import queue, context
import ctree.c.nodes as C
from ctree.types import get_c_type_from_numpy_dtype


class NDArray(np.ndarray):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        obj = np.ndarray.__new__(subtype, shape, dtype, buffer,
                                 offset, strides, order)
        obj.ocl_buf = cl.clCreateBuffer(
            context, np.prod(shape) * obj.itemsize)
        obj.host_dirty = False
        obj.ocl_dirty = False
        obj.register = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        buf, evt = cl.buffer_from_ndarray(queue, obj)
        self.ocl_buf = buf
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
            _, evt = cl.buffer_to_ndarray(queue, self.ocl_buf, self)
            evt.wait()
            self.host_dirty = False

    def promote_to_register(self, name):
        print(self.register)
        if self.register is None:
            self.register = name
            ptr = get_c_type_from_numpy_dtype(self.dtype)()
            return C.SymbolRef(name, ptr)
        return None

    @staticmethod
    def rand(shape, dtype):
        return np.random.rand(*shape).astype(dtype).view(NDArray)
