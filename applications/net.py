"""
python net.py --prototxt="models/alexnet-ng/deploy.prototxt" \
    --caffemodel="models/alexnet-ng/alexnet-ng.caffemodel" --phase='TEST'
"""
import argparse
import caffe_pb2 as pb
import caffe
from google.protobuf import text_format
from layers import ConvLayer, ReluLayer, PoolingLayer, InnerProductLayer, \
    SoftmaxLayer, LrnLayer, DataLayer, DropoutLayer, AccuracyLayer, \
    SoftmaxWithLossLayer
import numpy as np
from hindemith.types import hmarray
import time
import lmdb
import random


parser = argparse.ArgumentParser()
parser.add_argument(
    '--prototxt',
    help="path to prototxt using Caffe's format for network description",
    default="models/lenet/deploy.prototxt")
parser.add_argument(
    '--caffemodel',
    help="path to .caffemodel to use for network initialization",
    default='models/lenet/lenet_iter_5000.caffemodel')
parser.add_argument(
    '--phase',
    help="TRAIN or TEST",
    default='TEST')
args = parser.parse_args()
file_path = args.prototxt


class Net(object):
    layer_map = {
        pb.V1LayerParameter.CONVOLUTION: ConvLayer,
        "Convolution": ConvLayer,
        pb.V1LayerParameter.RELU: ReluLayer,
        "ReLU": ReluLayer,
        pb.V1LayerParameter.POOLING: PoolingLayer,
        "Pooling": PoolingLayer,
        pb.V1LayerParameter.SOFTMAX: SoftmaxLayer,
        "Softmax": SoftmaxLayer,
        pb.V1LayerParameter.SOFTMAX_LOSS: SoftmaxWithLossLayer,
        pb.V1LayerParameter.INNER_PRODUCT: InnerProductLayer,
        "InnerProduct": InnerProductLayer,
        pb.V1LayerParameter.LRN: LrnLayer,
        "LRN": LrnLayer,
        pb.V1LayerParameter.DROPOUT: DropoutLayer,
        pb.V1LayerParameter.ACCURACY: AccuracyLayer,
        pb.V1LayerParameter.DATA: DataLayer,
        "Data": DataLayer
    }

    def __init__(self, prototxt, params=None, phase='TEST'):
        self.blobs = {}
        self.layers = []

        self.net_param = pb.NetParameter()
        with open(prototxt, "rb") as f:
            text_format.Merge(f.read(), self.net_param)

        if phase == 'TEST':
            self.blobs['data'] = hmarray(self.net_param.input_dim, np.float32)
            self.blobs['data_diff'] = \
                hmarray(self.net_param.input_dim, np.float32)
        if len(self.net_param.layer) > 0:
            layer_params = self.net_param.layer
        else:
            layer_params = self.net_param.layers
        # Initialize layers
        for layer_param in layer_params:
            if len(layer_param.include) > 0 and \
                    layer_param.include[0].phase != getattr(caffe, phase):
                continue
            print("Initializing layer {}".format(layer_param.name))
            # Skip dropout layers for test
            if layer_param.type in (pb.V1LayerParameter.DROPOUT, "Dropout") \
                    and phase == 'TEST':
                continue
            layer_constructor = self.layer_map[layer_param.type]
            if layer_param.name in params:
                layer = layer_constructor(
                    layer_param, phase,
                    params[layer_param.name])
            else:
                layer = layer_constructor(layer_param, phase)
            self.layers.append(layer)
            bottom = []
            for blob in layer_param.bottom:
                bottom.append(self.blobs[blob])
                bottom.append(self.blobs["{}_diff".format(blob)])
            tops = layer.set_up(*bottom)
            for top, top_name in zip(tops, layer_param.top):
                self.blobs[top_name] = top[0]
                self.blobs["{}_diff".format(top_name)] = top[1]

    def forward_all(self, **kwargs):
        for key, value in kwargs.iteritems():
            self.blobs[key][...] = value
            self.blobs[key].sync_ocl()
        for layer in self.layers:
            layer.forward()

caffe_net = caffe.Net(args.prototxt, args.caffemodel,
                      getattr(caffe, args.phase))
net = Net(args.prototxt, caffe_net.params, args.phase)

if args.phase == 'TEST':
    im = caffe.io.load_image('data/cat.jpg')
    transformer = caffe.io.Transformer(
        {'data': caffe_net.blobs['data'].data.shape})
    transformer.set_mean(
        'data', np.load('models/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)
    data = np.asarray([transformer.preprocess('data', im)]).view(hmarray)
    data.sync_ocl()

    print("HM forward")
    start = time.clock()
    net.forward_all(data=data)
    end = time.clock()
    print("Time:", end - start)
    print("Done")
    print("Caffe forward")
    start = time.clock()
    out = caffe_net.forward_all(data=data)
    end = time.clock()
    print("Time:", end - start)
    print("Done")
else:
    net.forward_all()
    caffe_net.forward()

for blob_name in net.blobs.keys():
    if "_diff" in blob_name:
        continue
    print "Checking blob ", blob_name
    blob = net.blobs[blob_name]
    blob.sync_host()
    np.testing.assert_array_almost_equal(
        blob, caffe_net.blobs[blob_name].data, decimal=3)
print("SUCCESS")
