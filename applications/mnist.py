from hindemith.types import hmarray
from hindemith.core import compose
from hindemith.operations import ConvForward, PoolForward, \
    PoolBackward, ConvBackward, SoftmaxWithLossForward, \
    SoftmaxWithLossBackward, Relu
from hindemith.clibs.clBLAS import sgemm, sgemv
import caffe_pb2 as pb
import numpy as np
import lmdb

n = 0.01


class ConvLayer(object):
    def __init__(self, num_output, kernel_size, stride=1, padding=0,
                 bias_filler='constant', bias_value=0):
        self.num_output = num_output
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if bias_filler == 'constant':
            self.bias = hmarray.zeros((num_output,))
            self.bias_diff = hmarray.zeros((num_output,))
            if bias_value != 0:
                self.bias.fill(bias_value)
                self.bias.sync_ocl()

        @compose
        def hm_backward(bottom_diff, bottom, top_diff, weights, weights_diff,
                        bias_diff, kernel_size, padding, stride):
            bottom_diff, weights_diff, bias_diff = \
                ConvBackward(bottom, top_diff, weights,
                             kernel_size=(kernel_size, kernel_size),
                             stride=(stride, stride), padding=(padding, padding))
            return bottom_diff, weights_diff, bias_diff
        self.hm_backward = hm_backward

        @compose
        def hm_forward(top, bottom, weights, bias, kernel_size, padding,
                       stride):
            top = ConvForward(bottom, weights, bias,
                              kernel_size=(kernel_size, kernel_size),
                              padding=(padding, padding),
                              stride=(stride, stride))
            return top
        self.hm_forward = hm_forward

    def set_up(self, bottom, bottom_diff):
        self.bottom = bottom
        self.bottom_diff = bottom_diff
        num, channels, height, width = bottom.shape

        weights_shape = (self.num_output, channels * self.kernel_size *
                         self.kernel_size)
        scale = 1.0 / np.sqrt(self.num_output)
        self.weights = hmarray.random(weights_shape, _range=(-scale, scale))
        self.weights_diff = hmarray.zeros(weights_shape)

        height_out = (height + 2 * self.padding - self.kernel_size) // \
            self.stride + 1
        width_out = (width + 2 * self.padding - self.kernel_size) // \
            self.stride + 1
        top_shape = (num, self.num_output, height_out, width_out)
        self.top = hmarray.zeros(top_shape)
        self.top_diff = hmarray.zeros(top_shape)
        return self.top, self.top_diff

    def forward(self):
        self.hm_forward(self.top, self.bottom, self.weights, self.bias,
                        self.kernel_size, self.padding, self.stride)

    def backward(self):
        self.hm_backward(self.bottom_diff, self.bottom, self.top_diff,
                         self.weights, self.weights_diff, self.bias_diff,
                         self.kernel_size, self.padding, self.stride)

    def update_weights(self):
        for buf, diff in [(self.weights, self.weights_diff), (self.bias, self.bias_diff)]:
            diff.sync_host()
            buf -= .01 * diff
            buf.sync_ocl()

class PoolingLayer(object):
    def __init__(self, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        @compose
        def hm_forward(top, bottom, mask, kernel_size, padding, stride):
            top, mask = PoolForward(bottom,
                                    kernel_size=(kernel_size, kernel_size),
                                    padding=(padding, padding),
                                    stride=(stride, stride))
            return top, mask

        self.hm_forward = hm_forward

        @compose
        def hm_backward(bottom_diff, top_diff, mask, kernel_size, padding, stride):
            bottom_diff = PoolBackward(top_diff, mask,
                                       kernel_size=(kernel_size, kernel_size),
                                       padding=(padding, padding),
                                       stride=(stride, stride))
            return bottom_diff

        self.hm_backward = hm_backward

    def set_up(self, bottom, bottom_diff):
        self.bottom = bottom
        self.bottom_diff = bottom_diff
        num, channels, height, width = bottom.shape
        pooled_height = ((height + 2 * self.padding - self.kernel_size) //
                         self.stride) + 1
        pooled_width = ((width + 2 * self.padding - self.kernel_size) //
                        self.stride) + 1
        top_shape = num, channels, pooled_height, pooled_width
        self.mask = hmarray.zeros(top_shape)
        self.top = hmarray.zeros(top_shape)
        self.top_diff = hmarray.zeros(top_shape)
        return self.top, self.top_diff

    def forward(self):
        self.hm_forward(self.top, self.bottom, self.mask, self.kernel_size,
                        self.padding, self.stride)

    def backward(self):
        self.hm_backward(self.bottom_diff, self.top_diff, self.mask,
                         self.kernel_size, self.padding, self.stride)


class InnerProductLayer(object):
    def __init__(self, num_output):
        self.num_output = num_output

    def set_up(self, bottom, bottom_diff):
        self.bottom, self.bottom_diff = bottom, bottom_diff
        N = self.num_output
        K = np.prod(bottom.shape[1:])
        scale = 1.0 / np.sqrt(self.num_output)
        self.weights = hmarray.random((N, K), _range=(-scale, scale))
        self.weights_diff = hmarray.zeros((N, K))
        self.bias = hmarray.zeros((self.num_output, ))
        self.bias_diff = hmarray.zeros((self.num_output, ))
        self.bias_multiplier = hmarray((1, self.bottom.shape[0]))
        self.bias_multiplier.fill(1)
        self.bias_multiplier.sync_ocl()
        top_shape = (bottom.shape[0], N)
        self.top = hmarray.zeros(top_shape)
        self.top_diff = hmarray.zeros(top_shape)
        return self.top, self.top_diff

    def forward(self):
        N = self.num_output
        K = np.prod(self.bottom.shape[1:])
        M = self.bottom.shape[0]
        self.top.fill(0)
        sgemm(False, True, 1.0, self.bottom, 0, K, self.weights, 0, K, 0.0,
              self.top, 0, N, M, N, K)
        sgemm(False, False, 1.0, self.bias_multiplier, 0, 1, self.bias, 0, N,
              1.0, self.top, 0, N, M, N, 1)

    def backward(self):
        N = self.num_output
        K = np.prod(self.bottom.shape[1:])
        M = self.bottom.shape[0]
        sgemm(True, False, 1.0, self.top_diff, 0, N, self.bottom, 0, K, 0.0,
              self.weights_diff, 0, K, N, K, M)
        sgemv(True, M, N, 1.0, self.top_diff, 0, N, self.bias_multiplier, 0, 1,
              0.0, self.bias_diff, 0, 1)
        sgemm(False, False, 1.0, self.top_diff, 0, N, self.weights, 0,
              K, 0.0, self.bottom_diff, 0, K, M, K, N)

    def update_weights(self):
        for buf, diff in [(self.weights, self.weights_diff), (self.bias, self.bias_diff)]:
            diff.sync_host()
            buf -= .01 * diff
            buf.sync_ocl()


class ReluLayer(object):
    def __init__(self, bottom, bottom_diff):
        self.bottom, self.bottom_diff = bottom, bottom_diff

        @compose
        def hm_relu(bottom):
            bottom = Relu(bottom)
            return bottom

        self.hm_relu = hm_relu

    def forward(self):
        self.hm_relu(self.bottom)

    def backward(self):
        self.hm_relu(self.bottom_diff)


class SoftmaxWithLossLayer(object):
    def __init__(self):
        @compose
        def hm_forward(loss, bottom, label, prob):
            loss = SoftmaxWithLossForward(bottom, label, prob)
            return loss

        self.hm_forward = hm_forward

        @compose
        def hm_backward(bottom_diff, loss, label, prob):
            bottom_diff = SoftmaxWithLossBackward(loss, label, prob)
            return bottom_diff

        self.hm_backward = hm_backward

    def set_up(self, bottom, bottom_diff, label):
        self.bottom, self.bottom_diff, self.label = bottom, bottom_diff, label
        self.top = hmarray.zeros((1, ))
        self.prob = hmarray.zeros(bottom.shape)
        return self.top

    def forward(self):
        self.hm_forward(self.top, self.bottom, self.label, self.prob)

    def backward(self):
        self.hm_backward(self.bottom_diff, self.top, self.label, self.prob)


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
