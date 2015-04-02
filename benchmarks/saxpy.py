from ctree.util import Timer
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from scikits.cuda.cublas import cublasCreate, cublasSaxpy, cublasDestroy

from hindemith.types import Matrix
from hindemith.core import hm

shape = 2**10, 2**10
alpha = np.float32(np.random.rand())
alpha = 1
x = np.random.rand(*shape).astype(np.float32)
y = np.random.rand(*shape).astype(np.float32)
z = np.random.rand(*shape).astype(np.float32)
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)
z_gpu = gpuarray.to_gpu(z)
h = cublasCreate()


trials = 10
cuda_total = 0
hm_total = 0
for i in range(trials):
    x_gpu = gpuarray.to_gpu(x)
    y_gpu = gpuarray.to_gpu(y)
    z_gpu = gpuarray.to_gpu(z)
    with Timer() as t:
        cublasSaxpy(h, x_gpu.size, alpha, x_gpu.gpudata, 1, z_gpu.gpudata, 1)
        cublasSaxpy(h, y_gpu.size, alpha, y_gpu.gpudata, 1, z_gpu.gpudata, 1)
        z_gpu.get()
    cuda_total += t.interval * 1000
cublasDestroy(h)
print("cublas avg time: {0:.5f}ms".format(cuda_total / trials))
hm_x = Matrix(shape, np.float32, ndarray=x)
hm_y = Matrix(shape, np.float32, ndarray=y)
hm_z = Matrix(shape, np.float32, ndarray=z)


@hm
def fn(x, y, z):
    z = x + y + z
    return z

for i in range(trials):
    with Timer() as t:
        hm_z = fn(hm_x, hm_y, hm_z)
        hm_z.sync()
    hm_total += t.interval * 1000
print("hindemith avg time: {0:.5f}ms".format(hm_total / trials))

np.allclose(z_gpu.get(), hm_z.data)
