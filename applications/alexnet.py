from hindemith.types import NDArray
from hindemith.core import hm
import numpy as np
# from hindemith.operations.image_processing import patches_to_rows
from hindemith.operations.neural_net import Relu, Lrn, Pool, Conv, Dropout
# from hindemith.operations.ndarray import transpose, reshape, dot


num_img = 256

# Data layer
data = NDArray.rand((num_img, 3, 277, 277), np.float32) * 255
data.ocl_dirty = True
data.sync()
# Conv1
conv1_filters = NDArray.rand((96, 3, 11, 11), np.float32)
conv1_biases = NDArray((96, ), np.float32)
conv1 = NDArray((num_img, 96, 55, 55), np.float32)
# lrn1
lrn1_scale = NDArray((num_img, 96, 55, 55), np.float32)
norm1 = NDArray((num_img, 96, 55, 55), np.float32)

# pool1
pool1 = NDArray((num_img, 96, 27, 27), np.float32)
pool1_mask = NDArray((num_img, 96, 27, 27), np.float32)

# conv2
conv2_filters = NDArray.rand((256, 96, 5, 5), np.float32)
conv2_biases = NDArray((256, ), np.float32)
conv2 = NDArray((num_img, 256, 27, 27), np.float32)

# lrn2
lrn2_scale = NDArray((num_img, 256, 27, 27), np.float32)
norm2 = NDArray((num_img, 256, 27, 27), np.float32)

# pool2
pool2 = NDArray((num_img, 256, 13, 13), np.float32)
pool2_mask = NDArray((num_img, 256, 13, 13), np.float32)

# conv3
conv3_filters = NDArray.rand((384, 256, 3, 3), np.float32)
conv3_biases = NDArray((384, ), np.float32)
conv3 = NDArray((num_img, 384, 13, 13), np.float32)

# conv4
conv4_filters = NDArray.rand((384, 384, 3, 3), np.float32)
conv4_biases = NDArray((384, ), np.float32)
conv4 = NDArray((num_img, 384, 13, 13), np.float32)

# conv5
conv5_filters = NDArray.rand((256, 384, 3, 3), np.float32)
conv5_biases = NDArray((256, ), np.float32)
conv5 = NDArray((num_img, 256, 13, 13), np.float32)

# pool5
pool5 = NDArray((num_img, 256, 6, 6), np.float32)
pool5_mask = NDArray((num_img, 256, 6, 6), np.float32)

# fc6
fc6_conv_filters = NDArray.rand((4096, 256, 6, 6), np.float32)
fc6_conv_biases = NDArray((4096, ), np.float32)
fc6 = NDArray((num_img, 4096, 1, 1), np.float32)
fc6_mask = NDArray.rand((num_img, 4096, 1, 1), np.float32)

# fc7
fc7_conv_filters = NDArray.rand((4096, 4096, 1, 1), np.float32)
fc7_conv_biases = NDArray((4096, ), np.float32)
fc7 = NDArray((num_img, 4096, 1, 1), np.float32)
fc7_mask = NDArray.rand((num_img, 4096, 1, 1), np.float32)

# fc8
fc8_conv_filters = NDArray.rand((1000, 4096, 1, 1), np.float32)
fc8_conv_biases = NDArray((1000,), np.float32)
fc8 = NDArray((num_img, 1000, 1, 1), np.float32)

local_size = 5
alpha = 0.0001
beta = 0.75


@hm
def forward(data, conv1, lrn1_scale, conv1_filters, norm1, pool1,
            pool1_mask, conv2, conv2_filters, lr2n_scale, norm2,
            pool2, pool2_mask, conv3, conv3_filters, conv4,
            conv4_filters, conv5, conv5_filters, pool5, pool5_mask,
            fc6, fc6_mask, fc6_conv_filters, fc7, fc7_mask,
            fc7_conv_filters, fc8, fc8_conv_filters):
    conv1 = Conv(data, conv1_filters, kernel_size=(11, 11),
                 padding=(0, 0), stride=(4, 4))
    conv1 = Relu(conv1)
    norm1 = Lrn(conv1, lrn1_scale, alpha=alpha, beta=beta,
                local_size=local_size, k=1)
    pool1 = Pool(norm1, pool1_mask, kernel_size=(3, 3),
                 padding=(0, 0), stride=(2, 2))

    conv2 = Conv(pool1, conv2_filters, kernel_size=(5, 5),
                 padding=(2, 2), stride=(1, 1))
    conv2 = Relu(conv2)
    norm2 = Lrn(conv2, lrn2_scale, alpha=alpha, beta=beta,
                local_size=local_size, k=1)
    pool2 = Pool(norm2, pool2_mask, kernel_size=(3, 3),
                 padding=(0, 0), stride=(2, 2))

    conv3 = Conv(pool2, conv3_filters, kernel_size=(3, 3),
                 padding=(1, 1), stride=(1, 1))
    conv3 = Relu(conv3)

    conv4 = Conv(conv3, conv4_filters, kernel_size=(3, 3),
                 padding=(1, 1), stride=(1, 1))
    conv4 = Relu(conv4)

    conv5 = Conv(conv4, conv5_filters, kernel_size=(3, 3),
                 padding=(1, 1), stride=(1, 1))
    conv5 = Relu(conv5)
    pool5 = Pool(conv5, pool5_mask, kernel_size=(3, 3),
                 padding=(0, 0), stride=(2, 2))

    fc6 = Conv(pool5, fc6_conv_filters, kernel_size=(6, 6),
               padding=(0, 0), stride=(1, 1))
    fc6 = Relu(fc6)
    fc6 = Dropout(fc6, threshold=0.5, mask=fc6_mask)

    fc7 = Conv(fc6, fc7_conv_filters, kernel_size=(1, 1),
               padding=(0, 0), stride=(1, 1))
    fc7 = Relu(fc7)
    fc7 = Dropout(fc7, threshold=0.5, mask=fc7_mask)

    fc8 = Conv(fc7, fc8_conv_filters, kernel_size=(1, 1),
               padding=(0, 0), stride=(1, 1))
    return fc8

fc8 = forward(data, conv1, lrn1_scale, conv1_filters, norm1, pool1,
              pool1_mask, conv2, lrn2_scale, conv2_filters, norm2,
              pool2, pool2_mask, conv3, conv3_filters, conv4,
              conv4_filters, conv5, conv5_filters, pool5, pool5_mask,
              fc6, fc6_mask, fc6_conv_filters, fc7, fc7_mask,
              fc7_conv_filters, fc8, fc8_conv_filters)
fc8.sync()
print(fc8)

