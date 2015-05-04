import os

os.environ["HM_BACKEND"] = "omp"
from hindemith.types import hmarray                      # noqa
from hindemith.operations.conv import ConvForward        # noqa
from hindemith.operations.relu import ReluForward        # noqa
from hindemith.operations.pool import PoolForward        # noqa
from hindemith.operations.lrn import LrnForward          # noqa
from hindemith.operations.softmax import SoftmaxForward  # noqa
from hindemith.core import compose                       # noqa
# import caffe                                             # noqa
import numpy as np                                       # noqa
import time                                              # noqa
import hmcaffe.proto.caffe_pb2 as pb                     # noqa
from google.protobuf import text_format

prototxt = "models/alexnet-ng/deploy.prototxt"
caffemodel = "models/alexnet-ng/alexnet-ng.caffemodel"


class CaffeNet(object):
    def __init__(self, prototxt, caffemodel):
        self.params = {}
        proto_param = pb.NetParameter()
        with open(prototxt, "rb") as _file:
            text_format.Merge(_file.read(), proto_param)
        net_param = pb.NetParameter()
        with open(caffemodel, "rb") as _file:
            net_param.ParseFromString(_file.read())

        for layer in net_param.layers:
            if len(layer.blobs):
                self.params[layer.name] = [
                    np.array(blob) for blob in layer.blobs
                ]

caffe_net = CaffeNet(prototxt, caffemodel)

import IPython  # noqa
IPython.embed()
exit()

conv1_filters = caffe_net.params['conv1'][0].data.view(hmarray)
conv1_bias = caffe_net.params['conv1'][1].data.view(hmarray)
conv1 = hmarray.zeros(caffe_net.blobs['conv1'].data.shape)

norm1 = hmarray.zeros(caffe_net.blobs['norm1'].data.shape)
norm1_scale = hmarray.zeros(norm1.shape)

pool1 = hmarray.zeros(caffe_net.blobs['pool1'].data.shape)
pool1_mask = hmarray.zeros(pool1.shape)

conv2_filters = caffe_net.params['conv2'][0].data.view(hmarray)
conv2_bias = caffe_net.params['conv2'][1].data.view(hmarray)
conv2 = hmarray.zeros(caffe_net.blobs['conv2'].data.shape)

norm2 = hmarray.zeros(caffe_net.blobs['norm2'].data.shape)
norm2_scale = hmarray.zeros(norm2.shape)

pool2 = hmarray.zeros(caffe_net.blobs['pool2'].data.shape)
pool2_mask = hmarray.zeros(pool2.shape)

conv3_filters = caffe_net.params['conv3'][0].data.view(hmarray)
conv3_bias = caffe_net.params['conv3'][1].data.view(hmarray)
conv3 = hmarray.zeros(caffe_net.blobs['conv3'].data.shape)

conv4_filters = caffe_net.params['conv4'][0].data.view(hmarray)
conv4_bias = caffe_net.params['conv4'][1].data.view(hmarray)
conv4 = hmarray.zeros(caffe_net.blobs['conv4'].data.shape)

conv5_filters = caffe_net.params['conv5'][0].data.view(hmarray)
conv5_bias = caffe_net.params['conv5'][1].data.view(hmarray)
conv5 = hmarray.zeros(caffe_net.blobs['conv5'].data.shape)

pool5 = hmarray.zeros(caffe_net.blobs['pool5'].data.shape)
pool5_mask = hmarray.zeros(pool5.shape)

fc6_filters = caffe_net.params['fc6'][0].data.view(hmarray)
fc6_bias = caffe_net.params['fc6'][1].data.view(hmarray)
fc6_bias_multiplier = hmarray((1, pool5.shape[0]))
fc6_bias_multiplier.fill(1)
fc6 = hmarray.zeros(caffe_net.blobs['fc6'].data.shape)

fc7_filters = caffe_net.params['fc7'][0].data.view(hmarray)
fc7_bias = caffe_net.params['fc7'][1].data.view(hmarray)
fc7_bias_multiplier = hmarray((1, fc6.shape[0]))
fc7_bias_multiplier.fill(1)
fc7 = hmarray.zeros(caffe_net.blobs['fc7'].data.shape)

fc8_filters = caffe_net.params['fc8'][0].data.view(hmarray)
fc8_bias = caffe_net.params['fc8'][1].data.view(hmarray)
fc8_bias_multiplier = hmarray((1, fc7.shape[0]))
fc8_bias_multiplier.fill(1)
fc8 = hmarray.zeros(caffe_net.blobs['fc8'].data.shape)

prob = hmarray.zeros(caffe_net.blobs['prob'].data.shape)

local_size = 5
alpha = 0.0001
beta = 0.75


@compose
def forward(data):
    global fc6, fc7, fc8
    conv1 = ConvForward(data, conv1_filters, conv1_bias,
                        kernel_size=(11, 11), padding=(0, 0),
                        stride=(4, 4))
    conv1 = ReluForward(conv1)
    norm1, norm1_scale = LrnForward(conv1, alpha=alpha, beta=beta,
                                    local_size=local_size, k=1)
    pool1, pool1_mask = PoolForward(norm1, kernel_size=(3, 3),
                                    padding=(0, 0), stride=(2, 2))

    conv2 = ConvForward(pool1, conv2_filters, conv2_bias,
                        kernel_size=(5, 5), padding=(2, 2),
                        stride=(1, 1))
    conv2 = ReluForward(conv2)
    norm2, norm2_scale = LrnForward(conv2, alpha=alpha, beta=beta,
                                    local_size=local_size, k=1)
    pool2, pool2_mask = PoolForward(norm2, kernel_size=(3, 3),
                                    padding=(0, 0), stride=(2, 2))

    conv3 = ConvForward(pool2, conv3_filters, conv3_bias,
                        kernel_size=(3, 3), padding=(1, 1),
                        stride=(1, 1))
    conv3 = ReluForward(conv3)

    conv4 = ConvForward(conv3, conv4_filters, conv4_bias,
                        kernel_size=(3, 3), padding=(1, 1),
                        stride=(1, 1))
    conv4 = ReluForward(conv4)

    conv5 = ConvForward(conv4, conv5_filters, conv5_bias,
                        kernel_size=(3, 3), padding=(1, 1),
                        stride=(1, 1))
    conv5 = ReluForward(conv5)
    pool5, pool5_mask = PoolForward(conv5, kernel_size=(3, 3),
                                    padding=(0, 0), stride=(2, 2))

    np.dot(pool5.reshape(pool5.shape[0], fc6_filters.shape[1]), fc6_filters.T,
           fc6)
    fc6 += fc6_bias

    fc6 = ReluForward(fc6)

    np.dot(fc6, fc7_filters.T, fc7)
    fc7 += fc7_bias
    fc7 = ReluForward(fc7)

    np.dot(fc7, fc8_filters.T, fc8)
    fc8 += fc8_bias
    prob = SoftmaxForward(fc8)
    return prob

im = caffe.io.load_image('data/cat.jpg')
transformer = caffe.io.Transformer(
    {'data': caffe_net.blobs['data'].data.shape})
transformer.set_mean(
    'data', np.load('models/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_raw_scale('data', 255.0)


def get_data():
    # data = np.asarray([
    #     transformer.preprocess('data', im),
    # ]).view(hmarray)
    data = hmarray.random((10, 3, 227, 227), _range=(0, 255))

    # data *= hmarray.random((5, 3, 227, 227), _range=(0, 2))
    # data -= hmarray.random((5, 3, 227, 227), _range=(-20, +20))
    return data

num_trials = 3
hm_time = 0
caffe_time = 0

# warmup and test correctness
for _ in range(2):
    data = get_data()
    forward(data)
    caffe_net.forward_all(data=data)

    for blob_name in caffe_net.blobs.keys():
        blob = globals()[blob_name]
        print("Checking blob {}".format(blob_name))
        caffe_blob = caffe_net.blobs[blob_name].data
        np.testing.assert_array_almost_equal(blob, caffe_blob, decimal=1)
    caffe_prob = caffe_net.blobs['prob'].data
    np.testing.assert_array_almost_equal(prob, caffe_prob, decimal=4)

for i in range(num_trials):
    data = get_data()
    start = time.clock()
    forward(data)
    hm_time += time.clock() - start
    start = time.clock()
    caffe_net.forward_all(data=data)
    caffe_time += time.clock() - start
print("Hindemith AVG        : {}".format(hm_time / num_trials))
print("Caffe AVG            : {}".format(caffe_time / num_trials))
print("Speedup (CAFFE / HM) : {}".format(caffe_time / hm_time))
print "SUCCESS"
