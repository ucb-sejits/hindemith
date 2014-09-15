from hindemith.fusion.core import fuse
from hindemith.fusion.core import dont_fuse_fusables

from stencil_code.stencil_kernel import StencilKernel
import numpy as np
from scipy.ndimage import convolve

from ctree.util import Timer


radius = 1
# import logging
# logging.basicConfig(level=20)

np_blur = np.array(
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
)


class Stencil(StencilKernel):
    neighbor_definition = [[
        (-1, 1),  (0, 1),  (1, 1),
        (-1, 0),  (0, 0),  (1, 0),
        (-1, -1), (0, -1), (1, -1)
    ]]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]


x = []
iterations = 1
results = [[] for _ in range(4)]
speedup = [[] for _ in range(4)]

for width in range(2**10, 2**13, 1024):
# for width in range(2**8, 2**10, 256):
    print("Running width: %d" % width)
    total0, total1, total2, total3 = 0, 0, 0, 0
    for _ in range(iterations):
        stencil1 = Stencil(backend='ocl')
        stencil2 = Stencil(backend='ocl')
        stencil3 = Stencil(backend='ocl')
        stencil4 = Stencil(backend='ocl')
        stencil5 = Stencil(backend='ocl')
        stencil6 = Stencil(backend='ocl')

        @fuse
        def fused_f(A):
            C = stencil1(A)
            return stencil2(C)

        @dont_fuse_fusables
        def unfused_f(A):
            C = stencil3(A)
            return stencil4(C)

        def naive_f(A):
            C = stencil5(A)
            return stencil6(C)

        def numpy_unfused(A):
            tmp = convolve(A, np_blur)
            return convolve(tmp, np_blur)

        A = np.random.rand(width, width).astype(np.float32) * 256

        a = fused_f(A)
        b = unfused_f(A)
        c = numpy_unfused(A)
        d = naive_f(A)
        np.testing.assert_array_almost_equal(a[2:-2, 2:-2], b[2:-2, 2:-2], decimal=2)
        np.testing.assert_array_almost_equal(a[2:-2, 2:-2], c[2:-2, 2:-2], decimal=2)
        np.testing.assert_array_almost_equal(a[2:-2, 2:-2], d[2:-2, 2:-2], decimal=2)
        # x.append(width)
        with Timer() as fused_time:
            fused_f(A)
        # results[0].append(fused_time.interval)
        total0 += fused_time.interval

        with Timer() as unfused_time:
            unfused_f(A)
        total1 += unfused_time.interval

        with Timer() as naive_time:
            naive_f(A)
        total2 += naive_time.interval

        with Timer() as numpy_time:
            numpy_unfused(A)
        total3 += numpy_time.interval

    numpy = total3
    results[0].append(numpy / total0)
    results[1].append(numpy / total1)
    results[2].append(numpy / total2)
    results[3].append(numpy / total3)
    x.append(width)


# colors = ['b', 'c', 'r', 'g']
# import matplotlib.pyplot as plt

# # r1 = plt.scatter(x, results[0], marker='x', color=colors[0])
# # r2 = plt.scatter(x, results[1], marker='x', color=colors[1])
# # r3 = plt.scatter(x, results[2], marker='x', color=colors[2])
# # # r4 = plt.scatter(x, speedup[3], marker='x', color=colors[3])

# # plt.legend((r1, r2, r3), #, r4),
# #            ('Fused Kernel', 'Unfused Kernel', 'Numpy'), #, '5 Stencils'),
# #            scatterpoints=1,
# #            loc='lower left',
# #            ncol=3,
# #            fontsize=8)

import csv
print(results)
with open('numpy_results.csv', 'wb') as f:
    writer = csv.writer(f)
    for index, size in enumerate(x):
        writer.writerow([size] + [result[index] for result in reversed(results)])

# width = .15
# fig, ax = plt.subplots()
# # x = [2 * i for i in x]
# rects1 = ax.bar([index for index, _ in enumerate(x)], results[3], width, color='c')

# rects2 = ax.bar([index + width for index, _ in enumerate(x)], results[2], width, color='g')
# rects3 = ax.bar([index + width * 2 for index, _ in enumerate(x)], results[1], width, color='b')
# rects4 = ax.bar([index + width * 3 for index, _ in enumerate(x)], results[0], width, color='m')

# ax.set_title('Speedup over Numpy Convolve Implementation')
# ax.set_ylabel('Speedup')
# ax.set_xlabel('Square Matrix Size')
# ax.set_xticks([index + width for index, _ in enumerate(x)])
# ax.set_xticklabels(x)

# # ax.legend((rects1[0], rects2[0], rects3[0]), ('Fused v1', 'Unfused', 'Numpy'), loc=2)
# ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('Numpy', 'Naive Sequential Specializers (Data Movement)', 'Unfused Kernels (No data movement)','Fused Kernels (No data movement)'), loc=2)
# plt.show()
