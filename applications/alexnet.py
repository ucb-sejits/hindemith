from hindemith.types import NDArray
# from hindemith.operations.image_processing import patches_to_rows
from hindemith.operations.neural_net import Relu, Lrn, Pool, Conv, Dropout
# from hindemith.operations.ndarray import transpose, reshape, dot


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
    conv1 = Conv(data, conv1_filters, padding=(0, 0), stride=(4, 4))
    conv1 = Relu(conv1)
    norm1 = Lrn(conv1, padding=(0, 0), stride=(1, 1))
    pool1 = Pool(norm1, kernel_size=(3, 3), padding=(0, 0), stride=(2, 2))

    conv2 = Conv(pool1, conv2_filters, padding=(2, 2), stride=(1, 1))
    conv2 = Relu(conv2)
    norm2 = Lrn(conv2, padding=(0, 0), stride=(1, 1))
    pool2 = Pool(norm2, kernel_size=(3, 3), padding=(0, 0), stride=(2, 2))

    conv3 = Conv(pool2, conv3_filters, padding=(1, 1), stride=(1, 1))
    conv3 = Relu(conv3)

    conv4 = Conv(conv3, conv4_filters, padding=(1, 1), stride=(1, 1))
    conv4 = Relu(conv4)

    conv5 = Conv(conv4, conv5_filters, padding=(1, 1), stride=(1, 1))
    conv5 = Relu(conv5)
    pool5 = Pool(conv5, kernel_size=(3, 3), padding=(0, 0), stride=(2, 2))

    fc6 = Conv(pool5, fc6_conv_filters, padding=(0, 0), stride=(1, 1))
    fc6 = Relu(fc6)
    fc6 = Dropout(fc6)

    fc7 = Conv(fc6, fc7_conv_filters, padding=(0, 0), stride=(1, 1))
    fc7 = Relu(fc7)
    fc7 = Dropout(fc7)

    fc8 = Conv(fc7, fc8_conv_filters, padding=(0, 0), stride=(1, 1))
    return fc8

if __name__ == 'main':
    forward(data, conv1_filters, conv2_filters, conv3_filters,
            conv4_filters, conv5_filters, fc6_conv_filters,
            fc7_conv_filters, fc8_conv_filters)
