import numpy as np
from hindemith.linalg import add, sub, mul, div


class hmarray(np.ndarray):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        """
        ---IMPORTANT---
        Read this link before making changes,
        http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        ---------------

        The hmarray constructor accepts the normal ndarray prototype,
        but also allows marshalling an existing ndarray into the
        hmarray subclass by passing it in as the first argument.
        """
        if isinstance(shape, np.ndarray):
            # Set to subtype if marshalling an existing ndarray
            obj = np.asarray(shape).view(subtype)
        else:
            obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset,
                                     strides, order)
        obj._ocl_buf = None
        obj._host_dirty = False
        obj._ocl_dirty = False
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._ocl_buf = getattr(obj, '_ocl_buf', None)
        self._host_dirty = getattr(obj, '_host_dirty', None)
        self._ocl_dirty = getattr(obj, '_ocl_dirty', None)

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __div__(self, other):
        return div(self, other)
