import numpy as np
import pycl as cl
from hindemith.cl import queue


class Vector(object):
    def __init__(self, size, dtype, ndarray=None):
        if ndarray is None:
            self.data = np.ndarray(size, dtype)
        else:
            self.data = ndarray
        self.size = size
        self.shape = (size, )
        self.dtype = dtype
        self.host_dirty = False
        self.ocl_dirty = False
        buf, evt = cl.buffer_from_ndarray(queue, self.data)
        evt.wait()
        self.ocl_buf = buf

    def get_element(self, name):
        return "{}[get_global_id(0)]".format(name)

    def sync(self):
        if self.host_dirty:
            _, evt = cl.buffer_to_ndarray(queue, self.ocl_buf, self.data)
            evt.wait()
            self.host_dirty = False

    @staticmethod
    def rand(size, dtype):
        data = np.random.rand(size).astype(dtype) * 255
        vec = Vector(size, dtype, ndarray=data)
        return vec
