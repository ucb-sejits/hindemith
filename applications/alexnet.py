from hindemith.types import NDArray
# from hindemith.operations.image_processing import patches_to_rows
from hindemith.operations.neural_net import relu, lrn, pool, conv, dropout
from hindemith.operations.ndarray import transpose, reshape, dot


num_img = 256

data = NDArray(num_img, 3, 277, 277)
conv1_filters = NDArray(96, 3, 11, 11)
conv1_biases = NDArray(96)
# conv1 = NDArray(num_img, 96, 55, 55)
# norm1 = NDArray(num_img, 96, 55, 55)
# pool1 = NDArray(num_img, 96, 27, 27)
conv2_filters = NDArray(256, 96, 5, 5)
conv2_biases = NDArray(256)
# conv2 = NDArray(num_img, 256, 27, 27)
# norm2 = NDArray(num_img, 256, 27, 27)
# pool2 = NDArray(num_img, 256, 13, 13)
conv3_filters = NDArray(384, 256, 3, 3)
conv3_biases = NDArray(384)
# conv3 = NDArray(num_img, 384, 13, 13)
conv4_filters = NDArray(384, 384, 3, 3)
conv4_biases = NDArray(384)
# conv4 = NDArray(num_img, 384, 13, 13)
conv5_filters = NDArray(256, 384, 3, 3)
conv5_biases = NDArray(256)
conv5 = NDArray(num_img, 256, 13, 13)
# pool5 = NDArray(num_img, 256, 6, 6)
fc6_conv_filters = NDArray(4096, 256, 6, 6)
fc6_conv_biases = NDArray(4096)
# fc6 = NDArray(num_img, 4096, 1, 1)
fc7_conv_filters = NDArray(4096, 4096, 1, 1)
fc7_conv_biases = NDArray(4096)
# fc7 = NDArray(num_img, 4096, 1, 1)
fc8_conv_filters = NDArray(1000, 4096, 1, 1)
fc8_conv_biases = NDArray(1000)
# fc8 = NDArray(num_img, 1000, 1, 1)


def forward(data, conv1_filters, conv2_filters, conv3_filters,
            conv4_filters, conv5_filters, fc6_conv_filters,
            fc7_conv_filters, fc8_conv_filters):
    # conv1
    # for i in range(data.shape[0]):
    #     row_per_patch = patches_to_rows(data[i], padding=(0, 0),
    #                                     stride=(4, 4))
    #     conv1[i] = row_per_patch * transpose(reshape(conv1_filters, 96, 363))
    # conv1
    conv1 = conv(data, conv1_filters, padding=(0, 0), stride=(4, 4))
    conv1 = relu(conv1)
    norm1 = lrn(conv1, padding=(0, 0), stride=(1, 1))
    pool1 = pool(norm1, padding=(0, 0), stride=(2, 2))

    conv2 = conv(pool1, conv2_filters, padding=(2, 2), stride=(1, 1))
    conv2 = relu(conv2)
    norm2 = lrn(conv2, padding=(0, 0), stride=(1, 1))
    pool2 = pool(norm2, padding=(0, 0), stride=(2, 2))

    conv3 = conv(pool2, conv3_filters, padding=(1, 1), stride=(1, 1))
    conv3 = relu(conv3)

    conv4 = conv(conv3, conv4_filters, padding=(1, 1), stride=(1, 1))
    conv4 = relu(conv4)

    conv5 = conv(conv4, conv5_filters, padding=(1, 1), stride=(1, 1))
    conv5 = relu(conv5)
    pool5 = pool(conv5, padding=(0, 0), stride=(2, 2))

    fc6 = conv(pool5, fc6_conv_filters, padding=(0, 0), stride=(1, 1))
    fc6 = relu(fc6)
    fc6 = dropout(fc6)

    fc7 = conv(fc6, fc7_conv_filters, padding=(0, 0), stride=(1, 1))
    fc7 = relu(fc7)
    fc7 = dropout(fc7)

    fc8 = conv(fc7, fc8_conv_filters, padding=(0, 0), stride=(1, 1))
    return fc8

if __name__ == 'main':
    forward(data, conv1_filters, conv2_filters, conv3_filters,
            conv4_filters, conv5_filters, fc6_conv_filters,
            fc7_conv_filters, fc8_conv_filters)
