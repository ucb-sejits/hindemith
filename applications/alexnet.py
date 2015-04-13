from hindemith.types import NDArray
from hindemith.core import hm
import numpy as np
# from hindemith.operations.image_processing import patches_to_rows
from hindemith.operations.neural_net import Relu, LrnForward, PoolForward, \
    ConvForward, Dropout, SoftMaxWithLossForward, LrnBackward, PoolBackward, \
    ConvBackward, SoftMaxWithLossBackward
# from hindemith.operations.ndarray import transpose, reshape, dot
import lmdb
import caffe_pb2 as pb
import random

# env = lmdb.open(
#     "/home/neubotech/denoise_caffe/examples/mnist/mnist_train_lmdb")
env = lmdb.Environment(
    "/storage2/datasets/ilsvrc2012_train_256x256_lmdb", readonly=True)

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

# Conv1
conv1_filters = NDArray.rand((96, 3 * 11 * 11), np.float32) * 2 - 1
conv1_filters_diff = NDArray.zeros((96, 3 * 11 * 11), np.float32)
conv1_biases = NDArray((96, ), np.float32)
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
conv2_filters = NDArray.rand((256, 96 * 5 * 5), np.float32) * 2 - 1
conv2_filters_diff = NDArray.zeros((256, 96 * 5 * 5), np.float32)
conv2_biases = NDArray((256, ), np.float32)
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
conv3_filters = NDArray.rand((384, 256 * 3 * 3), np.float32) * 2 - 1
conv3_filters_diff = NDArray.zeros((384, 256 * 3 * 3), np.float32)
conv3_biases = NDArray.zeros((384, ), np.float32)
conv3 = NDArray.zeros((num_img, 384, 13, 13), np.float32)
conv3_diff = NDArray.zeros((num_img, 384, 13, 13), np.float32)

# conv4
conv4_filters = NDArray.rand((384, 384 * 3 * 3), np.float32) * 2 - 1
conv4_filters_diff = NDArray((384, 384 * 3 * 3), np.float32)
conv4_biases = NDArray((384, ), np.float32)
conv4 = NDArray.zeros((num_img, 384, 13, 13), np.float32)
conv4_diff = NDArray.zeros((num_img, 384, 13, 13), np.float32)

# conv5
conv5_filters = NDArray.rand((256, 384 * 3 * 3), np.float32) * 2 - 1
conv5_filters_diff = NDArray((256, 384 * 3 * 3), np.float32)
conv5_biases = NDArray((256, ), np.float32)
conv5 = NDArray.zeros((num_img, 256, 13, 13), np.float32)
conv5_diff = NDArray.zeros((num_img, 256, 13, 13), np.float32)

# pool5
pool5 = NDArray.zeros((num_img, 256, 6, 6), np.float32)
pool5_mask = NDArray.zeros((num_img, 256, 6, 6), np.float32)
pool5_diff = NDArray.zeros((num_img, 256, 6, 6), np.float32)

# fc6
fc6_conv_filters = NDArray.rand((4096, 256 * 6 * 6), np.float32) * 2 - 1
fc6_conv_filters_diff = NDArray((4096, 256 * 6 * 6), np.float32)
fc6_conv_biases = NDArray((4096, ), np.float32)
fc6 = NDArray.zeros((num_img, 4096, 1, 1), np.float32)
fc6_mask = NDArray.rand((num_img, 4096, 1, 1), np.float32)
fc6_diff = NDArray.zeros((num_img, 4096, 1, 1), np.float32)

# fc7
fc7_conv_filters = NDArray.rand((4096, 4096 * 1 * 1), np.float32) * 2 - 1
fc7_conv_filters_diff = NDArray((4096, 4096 * 1 * 1), np.float32)
fc7_conv_biases = NDArray((4096, ), np.float32)
fc7 = NDArray.zeros((num_img, 4096, 1, 1), np.float32)
fc7_diff = NDArray.zeros((num_img, 4096, 1, 1), np.float32)
fc7_mask = NDArray.rand((num_img, 4096, 1, 1), np.float32)

# fc8
fc8_conv_filters = NDArray.rand((1000, 4096 * 1 * 1), np.float32) * 2 - 1
fc8_conv_filters_diff = NDArray((1000, 4096 * 1 * 1), np.float32)
fc8_conv_biases = NDArray((1000,), np.float32)
fc8 = NDArray.zeros((num_img, 1000, 1, 1), np.float32)
fc8_diff = NDArray.zeros((num_img, 1000, 1, 1), np.float32)


for filt in (conv1_filters, conv2_filters, conv3_filters, conv4_filters,
             conv5_filters, fc6_conv_filters, fc7_conv_filters,
             fc8_conv_filters):
    filt[:] = filt * .02
    filt.sync_ocl(True)


local_size = 5
alpha = 0.0001
beta = 0.75

softmax_prob = NDArray.zeros(fc8.shape, np.float32)
loss = NDArray.zeros((1,), np.float32)


@hm
def forward():
    conv1 = ConvForward(data, conv1_filters, kernel_size=(11, 11),
                        padding=(0, 0), stride=(4, 4))
    conv1 = Relu(conv1)
    norm1 = LrnForward(conv1, lrn1_scale, alpha=alpha, beta=beta,
                       local_size=local_size, k=1)
    pool1 = PoolForward(norm1, pool1_mask, kernel_size=(3, 3),
                        padding=(0, 0), stride=(2, 2))

    conv2 = ConvForward(pool1, conv2_filters, kernel_size=(5, 5),
                        padding=(2, 2), stride=(1, 1))
    conv2 = Relu(conv2)
    norm2 = LrnForward(conv2, lrn2_scale, alpha=alpha, beta=beta,
                       local_size=local_size, k=1)
    pool2 = PoolForward(norm2, pool2_mask, kernel_size=(3, 3),
                        padding=(0, 0), stride=(2, 2))

    conv3 = ConvForward(pool2, conv3_filters, kernel_size=(3, 3),
                        padding=(1, 1), stride=(1, 1))
    conv3 = Relu(conv3)

    conv4 = ConvForward(conv3, conv4_filters, kernel_size=(3, 3),
                        padding=(1, 1), stride=(1, 1))
    conv4 = Relu(conv4)

    conv5 = ConvForward(conv4, conv5_filters, kernel_size=(3, 3),
                        padding=(1, 1), stride=(1, 1))
    conv5 = Relu(conv5)
    pool5 = PoolForward(conv5, pool5_mask, kernel_size=(3, 3),
                        padding=(0, 0), stride=(2, 2))

    fc6 = ConvForward(pool5, fc6_conv_filters, kernel_size=(6, 6),
                      padding=(0, 0), stride=(1, 1))
    fc6 = Relu(fc6)
    fc6 = Dropout(fc6, threshold=0.5, mask=fc6_mask)

    fc7 = ConvForward(fc6, fc7_conv_filters, kernel_size=(1, 1),
                      padding=(0, 0), stride=(1, 1))
    fc7 = Relu(fc7)
    fc7 = Dropout(fc7, threshold=0.5, mask=fc7_mask)

    fc8 = ConvForward(fc7, fc8_conv_filters, kernel_size=(1, 1),
                      padding=(0, 0), stride=(1, 1))
    loss = SoftMaxWithLossForward(fc8, label, softmax_prob)
    return loss


@hm
def backward(loss_diff):
    # Softmax
    fc8_diff = SoftMaxWithLossBackward(loss_diff, label, softmax_prob)
    # fc8
    fc7_diff = ConvBackward(fc7, fc8_diff, fc8_conv_filters,
                            fc8_conv_filters_diff, kernel_size=(1, 1),
                            padding=(0, 0), stride=(1, 1))
    fc7_diff = Dropout(fc7_diff, threshold=0.5, mask=fc7_mask)
    fc7_diff = Relu(fc7_diff)

    # fc7
    fc6_diff = ConvBackward(fc6, fc7_diff, fc7_conv_filters,
                            fc7_conv_filters_diff, kernel_size=(1, 1),
                            padding=(0, 0), stride=(1, 1))
    fc6_diff = Dropout(fc6_diff, threshold=0.5, mask=fc6_mask)
    fc6_diff = Relu(fc6_diff)

    # fc6
    pool5_diff = ConvBackward(pool5, fc6_diff, fc6_conv_filters,
                              fc6_conv_filters_diff, kernel_size=(6, 6),
                              padding=(0, 0), stride=(1, 1))

    # pool5
    conv5_diff = PoolBackward(pool5_diff, pool5_mask, kernel_size=(3, 3),
                              padding=(0, 0), stride=(2, 2))
    conv5_diff = Relu(conv5_diff)

    # conv5
    conv4_diff = ConvBackward(conv4, conv5_diff, conv5_filters,
                              conv5_filters_diff, kernel_size=(3, 3),
                              padding=(1, 1), stride=(1, 1))
    conv4_diff = Relu(conv4_diff)

    # conv4
    conv3_diff = ConvBackward(conv3, conv4_diff, conv4_filters,
                              conv4_filters_diff, kernel_size=(3, 3),
                              padding=(1, 1), stride=(1, 1))
    conv3_diff = Relu(conv3_diff)

    # conv3
    pool2_diff = ConvBackward(pool2, conv3_diff, conv3_filters,
                              conv3_filters_diff, kernel_size=(3, 3),
                              padding=(1, 1), stride=(1, 1))
    # pool2
    norm2_diff = PoolBackward(pool2_diff, pool2_mask, kernel_size=(3, 3),
                              padding=(0, 0), stride=(2, 2))

    # lrn2
    conv2_diff = LrnBackward(conv2, norm2, norm2_diff, lrn2_scale, alpha=alpha,
                             beta=beta, local_size=local_size, k=1)
    conv2_diff = Relu(conv2_diff)
    # conv2
    pool1_diff = ConvBackward(pool1, conv2_diff, conv2_filters,
                              conv2_filters_diff, kernel_size=(5, 5),
                              padding=(2, 2), stride=(1, 1))

    # pool1
    norm1_diff = PoolBackward(pool1_diff, pool1_mask, kernel_size=(3, 3),
                              padding=(0, 0), stride=(2, 2))
    # lrn1
    conv1_diff = LrnBackward(conv1, norm1, norm1_diff, lrn1_scale, alpha=alpha,
                             beta=beta, local_size=local_size, k=1)
    conv1_diff = Relu(conv1_diff)
    # conv1
    data_diff = ConvBackward(data, conv1_diff, conv1_filters,
                             conv1_filters_diff, kernel_size=(11, 11),
                             padding=(0, 0), stride=(4, 4))
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
    print("Loss", loss)
    backward(loss)

    for filt, diff in ((conv1_filters, conv1_filters_diff),
                       (conv2_filters, conv2_filters_diff),
                       (conv3_filters, conv3_filters_diff),
                       (conv4_filters, conv4_filters_diff),
                       (conv5_filters, conv5_filters_diff),
                       (fc6_conv_filters, fc6_conv_filters_diff),
                       (fc7_conv_filters, fc7_conv_filters_diff),
                       (fc8_conv_filters, fc8_conv_filters_diff)):
        filt.sync_host(True)
        diff.sync_host(True)
        print("Max of diff", np.max(diff))
        filt = filt - .1 * diff
        diff.fill(0)
        filt.sync_ocl(True)
        diff.sync_ocl(True)
