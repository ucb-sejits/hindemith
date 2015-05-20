import numpy as np
import pycl as cl
import os
backend = os.getenv("HM_BACKEND", "ocl")

if backend in {"ocl", "opencl", "OCL"}:
    from hindemith.cl import context, queue


class hmarray(np.ndarray):
    """Subclass of ndarray that has an OpenCL buffer associated with it"""
    def __new__(subtype, shape, dtype=np.float32, buffer=None, offset=0,
                strides=None, order=None, info=None):
        obj = np.ndarray.__new__(subtype, shape, dtype, buffer,
                                 offset, strides, order)
        if backend in {"ocl", "opencl", "OCL"}:
            obj.ocl_buf = cl.clCreateBuffer(
                context, np.prod(shape) * obj.itemsize)
            obj.host_dirty = False
            obj.ocl_dirty = False
        obj.register = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        if backend in {"ocl", "opencl", "OCL"}:
            buf, evt = cl.buffer_from_ndarray(queue, obj)
            evt.wait()
            self.ocl_buf = buf
            self.host_dirty = False
            self.ocl_dirty = False
        self.register = None

    def sync_host(self):
        if backend in {"ocl", "opencl", "OCL"}:
            if os.environ.get("HM_BACKEND") in {'omp', 'openmp'}:
                return
            _, evt = cl.buffer_to_ndarray(queue, self.ocl_buf, self)
            evt.wait()

    def sync_ocl(self):
        if backend in {"ocl", "opencl", "OCL"}:
            _, evt = cl.buffer_from_ndarray(queue, self, self.ocl_buf)
            evt.wait()
