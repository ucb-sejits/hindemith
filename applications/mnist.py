from hindemith.types import hmarray
from layers import ConvLayer, ReluLayer, InnerProductLayer, \
    SoftmaxWithLossLayer, PoolingLayer
from hindemith.clibs.clBLAS import sgemm, sgemv
import caffe_pb2 as pb
import numpy as np
import lmdb

n = 0.01


db_path = "data/mnist_train_lmdb_clean"
env = lmdb.Environment(db_path, readonly=True, lock=False)

PHASE = "train"
batch_size = 64
scale = 1.0 / 256.0

data = hmarray((batch_size, 1, 28, 28))
label = hmarray((batch_size, 1))

txn = env.begin()
cursor = txn.cursor().iternext()
datum = pb.Datum()

for i in range(batch_size):
    datum.ParseFromString(next(cursor)[1])
    unscaled = np.fromstring(
        datum.data, dtype=np.uint8).astype(np.float32).reshape(1, 28, 28)
    data[i] = unscaled * scale
    label[i] = datum.label

data.sync_ocl()
data_diff = hmarray.zeros(data.shape)
label.sync_ocl()

conv1_layer = ConvLayer(20, 5)
conv1, conv1_diff = conv1_layer.set_up(data, data_diff)

pool1_layer = PoolingLayer(2, 2)
pool1, pool1_diff = pool1_layer.set_up(conv1, conv1_diff)

conv2_layer = ConvLayer(50, 5)
conv2, conv2_diff = conv2_layer.set_up(pool1, pool1_diff)

pool2_layer = PoolingLayer(2, 2)
pool2, pool2_diff = pool2_layer.set_up(conv2, conv2_diff)

ip1_layer = InnerProductLayer(500)
ip1, ip1_diff = ip1_layer.set_up(pool2, pool2_diff)

relu1 = ReluLayer(ip1, ip1_diff)

ip2_layer = InnerProductLayer(10)
ip2, ip2_diff = ip2_layer.set_up(ip1, ip1_diff)

loss_layer = SoftmaxWithLossLayer()
loss = loss_layer.set_up(ip2, ip2_diff, label)


def forward_all():
    conv1_layer.forward()
    pool1_layer.forward()
    conv2_layer.forward()
    pool2_layer.forward()
    ip1_layer.forward()
    relu1.forward()
    ip2_layer.forward()
    loss_layer.forward()


def backward_all():
    loss_layer.backward()
    ip2_layer.backward()
    relu1.backward()
    ip1_layer.backward()
    pool2_layer.backward()
    conv2_layer.backward()
    pool1_layer.backward()
    conv1_layer.backward()

def update_weights():
    ip2_layer.update_weights()
    ip1_layer.update_weights()
    conv2_layer.update_weights()
    conv1_layer.update_weights()

for i in range(40):
    # for i in range(batch_size):
    #     datum.ParseFromString(next(cursor)[1])
    #     unscaled = np.fromstring(
    #         datum.data, dtype=np.uint8).astype(np.float32).reshape(1, 28, 28)
    #     data[i] = unscaled * scale
    #     label[i] = datum.label
    forward_all()
    # loss *= .01
    loss_layer.prob.sync_host()
    count = 0
    for p, l in zip(loss_layer.prob, loss_layer.label):
        if np.argmax(p) == int(l[0]):
            count += 1
    print(float(count) / batch_size)

    backward_all()
    update_weights()
exit(1)
