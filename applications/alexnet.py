from hindemith.types import hmarray
import numpy as np

import lmdb
import caffe_pb2 as pb
import random
import math

from layers import ConvLayer, PoolingLayer, InnerProductLayer, ReluLayer, \
    SoftmaxLayer, LrnLayer

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
data = data.view(hmarray)
data.sync_ocl()
data_diff = hmarray.zeros(data.shape, np.float32)
label = label.view(hmarray)
label.sync_ocl()

# Conv1
conv1_layer = ConvLayer(96, 11, stride=4)
conv1, conv1_diff = conv1_layer.set_up(data, data_diff)

# relu1
relu1_layer = ReluLayer(conv1, conv1_diff)

# lrn1
norm1_layer = LrnLayer()
norm1, norm1_diff = norm1_layer.set_up(conv1, conv1_diff)

# pool1
pool1_layer = PoolingLayer(3, stride=2)
pool1, pool1_diff = pool1_layer.set_up(norm1, norm1_diff)

# conv2
conv2_layer = ConvLayer(256, 5, stride=2)
conv2, conv2_diff = conv2_layer.set_up(pool1, pool1_diff)

# relu2
relu2_layer = ReluLayer(conv2, conv2_diff)

# lrn2
norm2_layer = LrnLayer()
norm2, norm2_diff = norm2_layer.set_up(conv2, conv2_diff)

# pool2
pool2_layer = PoolingLayer(3, stride=2)
pool2, pool2_diff = pool2_layer.set_up(norm2, norm2_diff)

# conv3
conv3_layer = ConvLayer(384, 3, padding=1)
conv3, conv3_diff = conv3_layer.set_up(pool2, pool2_diff)

# relu3
relu3_layer = ReluLayer(conv3, conv3_diff)

# conv4
conv4_layer = ConvLayer(384, 3, padding=1)
conv4, conv4_diff = conv4_layer.set_up(conv3, conv3_diff)

# relu4
relu4_layer = ReluLayer(conv4, conv4_diff)

# conv5
conv5_layer = ConvLayer(256, 3, padding=1)
conv5, conv5_diff = conv5_layer.set_up(conv4, conv4_diff)

# relu5
relu5_layer = ReluLayer(conv5, conv5_diff)

# pool5
pool5_layer = PoolingLayer(3, stride=2)
pool5, pool5_diff = pool5_layer.set_up(conv5, conv5_diff)

# fc6
fc6_layer = InnerProductLayer(4906)
fc6, fc6_diff = fc6_layer.set_up(pool5, pool5_diff)

# relu6
relu6_layer = ReluLayer(fc6, fc6_diff)

# fc7
fc7_layer = InnerProductLayer(4096)
fc7, fc7_diff = fc7_layer.set_up(fc6, fc6_diff)

# relu7
relu7_layer = ReluLayer(fc7, fc7_diff)

# fc8
fc8_layer = InnerProductLayer(1000)
fc8, fc8_diff = fc8_layer.set_up(fc7, fc7_diff)

# softmax
softmax_layer = SoftmaxLayer()
prob = softmax_layer.set_up(fc8, label)

def forward_all():
    conv1_layer.forward()
    relu1_layer.forward()
    norm1_layer.forward()
    pool1_layer.forward()

    conv2_layer.forward()
    relu2_layer.forward()
    norm2_layer.forward()
    pool2_layer.forward()

    conv3_layer.forward()
    relu3_layer.forward()

    conv4_layer.forward()
    relu4_layer.forward()

    conv5_layer.forward()
    relu5_layer.forward()
    pool5_layer.forward()

    fc6_layer.forward()
    relu6_layer.forward()

    fc7_layer.forward()
    relu7_layer.forward()

    fc8_layer.forward()
    softmax_layer.forward()


for outer_i in range(1):
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
    data.sync_ocl()
    label.sync_ocl()
    forward_all()
    prob.sync_host()
    print("Prediction", np.argmax(prob[0]))
    print("label", label[0])

exit(1)
