from ctree.util import Timer
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from scikits.cuda.cublas import cublasCreate, cublasSaxpy, cublasDestroy

from hindemith.types import Matrix
from hindemith.core import hm

shape = 2**12, 2**12
alpha = np.float32(np.random.rand())
alpha = 1
x = np.random.rand(*shape).astype(np.float32)
y = np.random.rand(*shape).astype(np.float32)
z = np.random.rand(*shape).astype(np.float32)
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)
z_gpu = gpuarray.to_gpu(z)
h = cublasCreate()

hm_x = Matrix(shape, np.float32, ndarray=x)
hm_y = Matrix(shape, np.float32, ndarray=y)
hm_z = Matrix(shape, np.float32, ndarray=z)


@hm
def fn(x, y, z):
    z = x + y + z
    return z

trials = 20
cuda_total = 0
hm_total = 0
for i in range(trials):
    x_gpu = gpuarray.to_gpu(x)
    y_gpu = gpuarray.to_gpu(y)
    z_gpu = gpuarray.to_gpu(z)
    with Timer() as t:
        cublasSaxpy(h, x_gpu.size, alpha, x_gpu.gpudata, 1, z_gpu.gpudata, 1)
        cublasSaxpy(h, y_gpu.size, alpha, y_gpu.gpudata, 1, z_gpu.gpudata, 1)
        result = z_gpu.get()
    print("cublas time {0:.5f}s".format(t.interval))
    cuda_total += t.interval

    with Timer() as t:
        hm_z = fn(hm_x, hm_y, hm_z)
        hm_z.sync()
    print("hindemith time {0:.5f}s".format(t.interval))
    hm_total += t.interval
    np.allclose(result, hm_z.data)

cublasDestroy(h)
print("cublas avg time: {0:.5f}s".format(cuda_total / trials))
print("hindemith avg time: {0:.5f}s".format(hm_total / trials))
print("speedup (cublas / hindemith): {0:.3f}x".format(cuda_total / hm_total))
