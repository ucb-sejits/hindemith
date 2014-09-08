from __future__ import print_function

import png

import sys
import numpy
import array
import math
import time

from stencil_code.stencil_kernel import StencilKernel

radius = 1


class BilateralKernel(StencilKernel):
    # def __init__(self, radius):
    #     self.radius = radius
    #     super(BilateralKernel, self).__init__()

    @property
    def radius(self):
        return 1

    @property
    def dim(self):
        return 3

    @property
    def ghost_depth(self):
        return self.radius

    def neighbors(self, point, neighbor_index=0):
        assert neighbor_index == 0
        for x in range(-self.radius, self.radius+1):
            for y in range(-self.radius, self.radius+1):
                yield (point[0] - x, point[1] - y, point[2])

    def kernel(self, in_img, filter_d, filter_s, out_img):
        for x in self.interior_points():
            for y in self.neighbors(x, 0):
                out_img[x] += in_img[y] * filter_d[int(distance(x, y))] *\
                    filter_s[abs(int(in_img[x] - in_img[y]))]


def gaussian(stdev, length):
    scale = 1.0 / (stdev*math.sqrt(2.0*math.pi))
    divisor = -1.0 / (2.0 * stdev * stdev)

    return numpy.array([scale * math.exp(float(x) * float(x) * divisor) for x in range(length)])


def distance(x, y):
    return math.sqrt(sum([(x[i]-y[i])**2 for i in range(0, len(x))]))


def write_image(image_array, file_name, metadata):
    m = metadata
    height = image_array.shape[0]
    width = image_array.shape[1]
    print("m {}".format(m))
    writer = png.Writer(width, height, alpha=m['alpha'], greyscale=m['greyscale'], bitdepth=m['bitdepth'],
                        interlace=m['interlace'], planes=m['planes'])
    output = array.array('B', image_array.reshape(width * height * m['planes']))
    with open(file_name, 'wb') as out_file:
        writer.write_array(out_file, output)


def main():
    width, height, raw_pixels, metadata = png.Reader('gray.png').read_flat()
    pixels = numpy.array(raw_pixels).reshape(height, width, metadata['planes']).astype(numpy.float32)
    print("image metadata {}".format(metadata))

    sigma_s = 3 if len(sys.argv) < 2 else int(sys.argv[1])
    stdev_s = 70
    radius = sigma_s * 3

    intensity = float(sum(raw_pixels))/len(raw_pixels)

    kernel = BilateralKernel(backend='ocl')

    out_grid = numpy.empty_like(pixels)

    gaussian1 = gaussian(sigma_s, radius*2)
    gaussian2 = gaussian(stdev_s, 256)

    print("gaussian1 {}".format(gaussian1))
    print("gaussian2 {}".format(gaussian2))

    kernel(pixels, gaussian1, gaussian2, out_grid)

    for x in range(0, width):
        for y in range(0,height):
            pixels[y * width + x] = out_grid.data[(x, y)]

    out_intensity = float(sum(pixels))/len(pixels)
    for i in range(0, len(pixels)):
        pixels[i] = min(255, max(0, int(pixels[i] * (intensity/out_intensity))))

    write_image(out_grid, 'bilat.png', metadata)


if __name__ == '__main__':
    main()