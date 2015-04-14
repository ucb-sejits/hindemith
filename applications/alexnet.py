from hindemith.types import NDArray
from hindemith.core import hm
import numpy as np
from hindemith.operations.neural_net import Relu, LrnForward, PoolForward, \
    ConvForward, Dropout, SoftMaxWithLossForward, LrnBackward, PoolBackward, \
    ConvBackward, SoftMaxWithLossBackward
import lmdb
import caffe_pb2 as pb
import random
import math

db_path = "/storage2/datasets/ilsvrc2012_train_256x256_lmdb"
env = lmdb.Environment(db_path, readonly=True, lock=False)

PHASE = "train"
crop_size = 227
num_img = 64

# Data layer
data = np.ndarray((num_img, 3, 277, 277), np.float32)
data = []
label = np.ndarray((num_img, 1), np.float32)

txn = env.begin()
cursor = txn.cursor().iternext()
datum = pb.Datum()
for i in range(num_img):
    datum.ParseFromString(next(cursor)[1])
    channels, datum_height, datum_width = datum.channels, datum.height, \
        datum.width
    # channels, datum_height, datum_width = 1, 28, 28
    height = datum_height
    width = datum_width
    if PHASE == "train":
        height = crop_size
        width = crop_size
        h_off = random.randrange(datum_height - crop_size + 1)
        w_off = random.randrange(datum_width - crop_size + 1)
    else:
        h_off = (datum_height - crop_size) / 2
        w_off = (datum_width - crop_size) / 2
    uncropped = np.fromstring(
        datum.data, dtype=np.uint8
    ).astype(np.float32).reshape(channels, datum_height, datum_width)
    for c in range(channels):
        uncropped[c] = np.fliplr(uncropped[c])
    data.append(uncropped[..., h_off:h_off + height, w_off:w_off + width])
    label[i] = datum.label

data = np.array(data)
data = data.view(NDArray)
data.sync_ocl(force=True)
data_diff = NDArray.zeros(data.shape, np.float32)
label = label.view(NDArray)
label.sync_ocl(force=True)

def init_filters(shape, std=0.01):
    """
    Initialize filters using xaviar scheme.
    Draws random sample from [-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}] where
    n is the number of filters

    :param NDArray shape: Tuple of scheme (num_filters, spatial_dim)
    :return tuple[NDArray, NDArray]: Tuple of initialized filters, zero-ed diff
    """
    num_filters = shape[0]
    # filters = NDArray.rand(shape, np.float32) * 2 - 1
    # filters *= (1.0 / math.sqrt(num_filters))
    # filters.sync_ocl(True)
    filters = NDArray.normal(0.0, std, shape, np.float32)
    diff = NDArray.zeros(shape, np.float32)
    return filters, diff

# Conv1
conv1_filters, conv1_filters_diff = init_filters((96, 3 * 11 * 11))
conv1_bias = NDArray.zeros((96, ), np.float32)
conv1_bias_diff = NDArray.zeros((96, ), np.float32)
conv1 = NDArray.zeros((num_img, 96, 55, 55), np.float32)
conv1_diff = NDArray.zeros((num_img, 96, 55, 55), np.float32)

# lrn1
lrn1_scale = NDArray.zeros((num_img, 96, 55, 55), np.float32)
norm1 = NDArray.zeros((num_img, 96, 55, 55), np.float32)
norm1_diff = NDArray.zeros((num_img, 96, 55, 55), np.float32)

# pool1
pool1 = NDArray.zeros((num_img, 96, 27, 27), np.float32)
pool1_mask = NDArray.zeros((num_img, 96, 27, 27), np.float32)
pool1_diff = NDArray.zeros((num_img, 96, 27, 27), np.float32)

# conv2
conv2_filters, conv2_filters_diff = init_filters((256, 96 * 5 * 5))
conv2_bias = NDArray((256, ), np.float32)
conv2_bias_diff = NDArray((256, ), np.float32)
conv2_bias.fill(.1)
conv2_bias.sync_ocl(True)
conv2 = NDArray.zeros((num_img, 256, 27, 27), np.float32)
conv2_diff = NDArray.zeros((num_img, 256, 27, 27), np.float32)

# lrn2
lrn2_scale = NDArray.zeros((num_img, 256, 27, 27), np.float32)
norm2 = NDArray.zeros((num_img, 256, 27, 27), np.float32)
norm2_diff = NDArray.zeros((num_img, 256, 27, 27), np.float32)

# pool2
pool2 = NDArray.zeros((num_img, 256, 13, 13), np.float32)
pool2_mask = NDArray.zeros((num_img, 256, 13, 13), np.float32)
pool2_diff = NDArray.zeros((num_img, 256, 13, 13), np.float32)

# conv3
conv3_filters, conv3_filters_diff = init_filters((384, 256 * 3 * 3))
conv3_bias = NDArray.zeros((384, ), np.float32)
conv3_bias_diff = NDArray.zeros((384, ), np.float32)
conv3 = NDArray.zeros((num_img, 384, 13, 13), np.float32)
conv3_diff = NDArray.zeros((num_img, 384, 13, 13), np.float32)

# conv4
conv4_filters, conv4_filters_diff = init_filters((384, 384 * 3 * 3))
conv4_bias = NDArray((384, ), np.float32)
conv4_bias_diff = NDArray.zeros((384, ), np.float32)
conv4_bias.fill(.1)
conv4_bias.sync_ocl(True)
conv4 = NDArray.zeros((num_img, 384, 13, 13), np.float32)
conv4_diff = NDArray.zeros((num_img, 384, 13, 13), np.float32)

# conv5
conv5_filters, conv5_filters_diff = init_filters((256, 384 * 3 * 3))
conv5_bias = NDArray((256, ), np.float32)
conv5_bias_diff = NDArray.zeros((256, ), np.float32)
conv5_bias.fill(.1)
conv5_bias.sync_ocl(True)
conv5 = NDArray.zeros((num_img, 256, 13, 13), np.float32)
conv5_diff = NDArray.zeros((num_img, 256, 13, 13), np.float32)

# pool5
pool5 = NDArray.zeros((num_img, 256, 6, 6), np.float32)
pool5_mask = NDArray.zeros((num_img, 256, 6, 6), np.float32)
pool5_diff = NDArray.zeros((num_img, 256, 6, 6), np.float32)

# fc6
fc6_conv_filters, fc6_conv_filters_diff = init_filters((4096, 256 * 6 * 6), .005)
fc6 = NDArray.zeros((num_img, 4096, 1, 1), np.float32)
fc6_bias = NDArray((4096, ), np.float32)
fc6_bias_diff = NDArray.zeros((4096, ), np.float32)
fc6_bias.fill(.1)
fc6_mask = NDArray.rand((num_img, 4096, 1, 1), np.float32)
fc6_diff = NDArray.zeros((num_img, 4096, 1, 1), np.float32)

# fc7
fc7_conv_filters, fc7_conv_filters_diff = init_filters((4096, 4096 * 1 * 1), .005)
fc7 = NDArray.zeros((num_img, 4096, 1, 1), np.float32)
fc7_bias = NDArray((4096, ), np.float32)
fc7_bias_diff = NDArray.zeros((4096, ), np.float32)
fc7_bias.fill(.1)
fc7_diff = NDArray.zeros((num_img, 4096, 1, 1), np.float32)
fc7_mask = NDArray.rand((num_img, 4096, 1, 1), np.float32)

# fc8
fc8_conv_filters, fc8_conv_filters_diff = init_filters((1000, 4096 * 1 * 1))
fc8 = NDArray.zeros((num_img, 1000, 1, 1), np.float32)
fc8_bias = NDArray.zeros((1000, ), np.float32)
fc8_bias_diff = NDArray.zeros((1000,), np.float32)
fc8_diff = NDArray.zeros((num_img, 1000, 1, 1), np.float32)


local_size = 5
alpha = 0.0001
beta = 0.75

softmax_prob = NDArray.zeros(fc8.shape, np.float32)
loss = NDArray.zeros((1,), np.float32)


@hm
def forward():
    conv1 = ConvForward(data, conv1_filters, conv1_bias, kernel_size=(11, 11),
                        padding=(0, 0), stride=(4, 4))
    conv1 = Relu(conv1)
    norm1 = LrnForward(conv1, lrn1_scale, alpha=alpha, beta=beta,
                       local_size=local_size, k=1)
    pool1 = PoolForward(norm1, pool1_mask, kernel_size=(3, 3),
                        padding=(0, 0), stride=(2, 2))

    conv2 = ConvForward(pool1, conv2_filters, conv2_bias, kernel_size=(5, 5),
                        padding=(2, 2), stride=(1, 1))
    conv2 = Relu(conv2)
    norm2 = LrnForward(conv2, lrn2_scale, alpha=alpha, beta=beta,
                       local_size=local_size, k=1)
    pool2 = PoolForward(norm2, pool2_mask, kernel_size=(3, 3),
                        padding=(0, 0), stride=(2, 2))

    conv3 = ConvForward(pool2, conv3_filters, conv3_bias, kernel_size=(3, 3),
                        padding=(1, 1), stride=(1, 1))
    conv3 = Relu(conv3)

    conv4 = ConvForward(conv3, conv4_filters, conv4_bias, kernel_size=(3, 3),
                        padding=(1, 1), stride=(1, 1))
    conv4 = Relu(conv4)

    conv5 = ConvForward(conv4, conv5_filters, conv5_bias, kernel_size=(3, 3),
                        padding=(1, 1), stride=(1, 1))
    conv5 = Relu(conv5)
    pool5 = PoolForward(conv5, pool5_mask, kernel_size=(3, 3),
                        padding=(0, 0), stride=(2, 2))

    fc6 = ConvForward(pool5, fc6_conv_filters, fc6_bias, kernel_size=(6, 6),
                      padding=(0, 0), stride=(1, 1))
    fc6 = Relu(fc6)
    fc6 = Dropout(fc6, threshold=0.5, mask=fc6_mask)

    fc7 = ConvForward(fc6, fc7_conv_filters, fc7_bias, kernel_size=(1, 1),
                      padding=(0, 0), stride=(1, 1))
    fc7 = Relu(fc7)
    fc7 = Dropout(fc7, threshold=0.5, mask=fc7_mask)

    fc8 = ConvForward(fc7, fc8_conv_filters, fc8_bias, kernel_size=(1, 1),
                      padding=(0, 0), stride=(1, 1))
    loss = SoftMaxWithLossForward(fc8, label, softmax_prob)
    return loss


@hm
def backward(loss_diff):
    # Softmax
    fc8_diff = SoftMaxWithLossBackward(loss_diff, label, softmax_prob)
    # fc8
    fc7_diff = ConvBackward(fc7, fc8_diff, fc8_conv_filters,
                            fc8_conv_filters_diff, fc8_bias_diff,
                            kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
    fc7_diff = Dropout(fc7_diff, threshold=0.5, mask=fc7_mask)
    fc7_diff = Relu(fc7_diff)

    # fc7
    fc6_diff = ConvBackward(fc6, fc7_diff, fc7_conv_filters,
                            fc7_conv_filters_diff, fc7_bias_diff,
                            kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
    fc6_diff = Dropout(fc6_diff, threshold=0.5, mask=fc6_mask)
    fc6_diff = Relu(fc6_diff)

    # fc6
    pool5_diff = ConvBackward(pool5, fc6_diff, fc6_conv_filters,
                              fc6_conv_filters_diff, fc6_bias_diff,
                              kernel_size=(6, 6), padding=(0, 0), stride=(1, 1))

    # pool5
    conv5_diff = PoolBackward(pool5_diff, pool5_mask, kernel_size=(3, 3),
                              padding=(0, 0), stride=(2, 2))
    conv5_diff = Relu(conv5_diff)

    # conv5
    conv4_diff = ConvBackward(conv4, conv5_diff, conv5_filters,
                              conv5_filters_diff, conv5_bias_diff,
                              kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
    conv4_diff = Relu(conv4_diff)

    # conv4
    conv3_diff = ConvBackward(conv3, conv4_diff, conv4_filters,
                              conv4_filters_diff, conv4_bias_diff,
                              kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
    conv3_diff = Relu(conv3_diff)

    # conv3
    pool2_diff = ConvBackward(pool2, conv3_diff, conv3_filters,
                              conv3_filters_diff, conv3_bias_diff,
                              kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
    # pool2
    norm2_diff = PoolBackward(pool2_diff, pool2_mask, kernel_size=(3, 3),
                              padding=(0, 0), stride=(2, 2))

    # lrn2
    conv2_diff = LrnBackward(conv2, norm2, norm2_diff, lrn2_scale, alpha=alpha,
                             beta=beta, local_size=local_size, k=1)
    conv2_diff = Relu(conv2_diff)
    # conv2
    pool1_diff = ConvBackward(pool1, conv2_diff, conv2_filters,
                              conv2_filters_diff, conv2_bias_diff,
                              kernel_size=(5, 5), padding=(2, 2), stride=(1, 1))

    # pool1
    norm1_diff = PoolBackward(pool1_diff, pool1_mask, kernel_size=(3, 3),
                              padding=(0, 0), stride=(2, 2))
    # lrn1
    conv1_diff = LrnBackward(conv1, norm1, norm1_diff, lrn1_scale, alpha=alpha,
                             beta=beta, local_size=local_size, k=1)
    conv1_diff = Relu(conv1_diff)
    # conv1
    data_diff = ConvBackward(data, conv1_diff, conv1_filters,
                             conv1_filters_diff, conv1_bias_diff,
                             kernel_size=(11, 11), padding=(0, 0),
                             stride=(4, 4))
    return data_diff

for outer_i in range(6):
    print("Running iter {}".format(outer_i))
    for i in range(num_img):
        datum.ParseFromString(next(cursor)[1])
        channels, datum_height, datum_width = datum.channels, datum.height, \
            datum.width
        # channels, datum_height, datum_width = 1, 28, 28
        height = datum_height
        width = datum_width
        if PHASE == "train":
            height = crop_size
            width = crop_size
            h_off = random.randrange(datum_height - crop_size + 1)
            w_off = random.randrange(datum_width - crop_size + 1)
        else:
            h_off = (datum_height - crop_size) / 2
            w_off = (datum_width - crop_size) / 2
        uncropped = np.fromstring(
            datum.data, dtype=np.uint8
        ).astype(np.float32).reshape(channels, datum_height, datum_width)
        for c in range(channels):
            uncropped[c] = np.fliplr(uncropped[c])
        data[i] = uncropped[..., h_off:h_off + height, w_off:w_off + width]
        label[i] = datum.label
    data.sync_ocl(True)
    label.sync_ocl(True)
    forward()
    softmax_prob.sync_host(True)
    print("Median of softmax", np.median(softmax_prob[0]))
    print("Max of softmax", np.max(softmax_prob[0]))
    print("Loss", loss)
    backward(loss)

    for filt, diff, bias, bias_diff in (
            (conv1_filters, conv1_filters_diff, conv1_bias, conv1_bias_diff),
            (conv2_filters, conv2_filters_diff, conv2_bias, conv2_bias_diff),
            (conv3_filters, conv3_filters_diff, conv3_bias, conv3_bias_diff),
            (conv4_filters, conv4_filters_diff, conv4_bias, conv4_bias_diff),
            (conv5_filters, conv5_filters_diff, conv5_bias, conv5_bias_diff),
            (fc6_conv_filters, fc6_conv_filters_diff, fc6_bias, fc6_bias_diff),
            (fc7_conv_filters, fc7_conv_filters_diff, fc7_bias, fc7_bias_diff),
            (fc8_conv_filters, fc8_conv_filters_diff, fc8_bias, fc8_bias_diff)):
        filt.sync_host(True)
        diff.sync_host(True)
        print("Max of diff", np.max(diff))
        filt[...] = filt - .1 * diff
        filt.sync_ocl(True)
        bias.sync_host(True)
        bias_diff.sync_host(True)
        bias[...] = bias - .1 * bias_diff
        bias.sync_ocl(True)
