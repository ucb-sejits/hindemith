import numpy as np
import pycl as cl

from hindemith.cl import context, queue


class hmarray(np.ndarray):
    """Subclass of ndarray that has an OpenCL buffer associated with it"""
    def __new__(subtype, shape, dtype=np.float32, buffer=None, offset=0,
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

        if hasattr(obj, 'ocl_buf'):
            self.ocl_buf = obj.ocl_buf
            self.host_dirty = obj.host_dirty
            self.ocl_dirty = obj.ocl_dirty
        else:
            buf, evt = cl.buffer_from_ndarray(queue, obj)
            evt.wait()
            self.ocl_buf = buf
            self.host_dirty = False
            self.ocl_dirty = False
        self.register = None

    def sync_host(self):
        cl.clFinish(queue)
        _, evt = cl.buffer_to_ndarray(queue, self.ocl_buf, self)
        evt.wait()

    def sync_ocl(self):
        cl.clFinish(queue)
        _, evt = cl.buffer_from_ndarray(queue, self, self.ocl_buf)
        evt.wait()

    @staticmethod
    def random(shape, _range=(0, 1), dtype=np.float32):
        rand = np.random.rand(*shape).astype(dtype)
        length = _range[1] - _range[0]
        rand *= length
        rand += _range[0]
        return rand.view(hmarray)

    @staticmethod
    def zeros(shape, dtype=np.float32):
        return np.zeros(shape, dtype).view(hmarray)
