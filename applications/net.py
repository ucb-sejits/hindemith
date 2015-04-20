"""
python net.py --prototxt="models/alexnet-ng/deploy.prototxt" --caffemodel="models/alexnet-ng/alexnet-ng.caffemodel" --phase='TEST'
"""
import argparse
import caffe_pb2 as pb
import caffe
from google.protobuf import text_format
from layers import ConvLayer, ReluLayer, PoolingLayer, InnerProductLayer, \
    SoftmaxLayer, LrnLayer, DataLayer
import numpy as np
from hindemith.types import hmarray
import time


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

# txn = env.begin()
# cursor = txn.cursor().iternext()
# datum = pb.Datum()
# for i in range(num_img):
#     datum.ParseFromString(next(cursor)[1])
#     channels, datum_height, datum_width = datum.channels, datum.height, \
#         datum.width
#     height = datum_height
#     width = datum_width
#     height = crop_size
#     width = crop_size
#     if PHASE == "train":
#         h_off = random.randrange(datum_height - crop_size + 1)
#         w_off = random.randrange(datum_width - crop_size + 1)
#     else:
#         h_off = (datum_height - crop_size) / 2
#         w_off = (datum_width - crop_size) / 2
#     uncropped = np.fromstring(
#         datum.data, dtype=np.uint8
#     ).astype(np.float32).reshape(channels, datum_height, datum_width)
#     for c in range(channels):
#         uncropped[c] = np.fliplr(uncropped[c])
#     # data[i] = uncropped[..., h_off:h_off + height, w_off:w_off + width]
#     data.append(uncropped[..., h_off:h_off + height, w_off:w_off + width])
#     label[i] = datum.label


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
        pb.V1LayerParameter.SOFTMAX_LOSS: SoftmaxLayer,
        pb.V1LayerParameter.INNER_PRODUCT: InnerProductLayer,
        "InnerProduct": InnerProductLayer,
        pb.V1LayerParameter.LRN: LrnLayer,
        "LRN": LrnLayer,
        "Data": DataLayer
    }

    def __init__(self, prototxt, caffemodel=None, phase='TEST'):
        self.blobs = {}
        self.layers = []

        net_param = pb.NetParameter()
        with open(prototxt, "rb") as f:
            text_format.Merge(f.read(), net_param)

        if phase == 'TEST':
            self.caffe_net = caffe.Net(prototxt, caffemodel, caffe.TEST)
            self.blobs['data'] = hmarray(net_param.input_dim, np.float32)
            if len(net_param.layer) > 0:
                layer_params = net_param.layer
            else:
                layer_params = net_param.layers
            # Initialize layers
            for layer_param in layer_params:
                # Skip dropout layers for test
                if layer_param.type in (pb.V1LayerParameter.DROPOUT,
                                        "Dropout"):
                    continue
                layer_constructor = self.layer_map[layer_param.type]
                if layer_param.name in self.caffe_net.params:
                    layer = layer_constructor(
                        layer_param, self.caffe_net.params[layer_param.name])
                else:
                    layer = layer_constructor(layer_param)
                self.layers.append(layer)
                bottom = []
                for blob in layer_param.bottom:
                    bottom.append(self.blobs[blob])
                tops = layer.set_up(*bottom)
                for top, top_name in zip(tops, layer_param.top):
                    self.blobs[top_name] = top
        elif phase == 'TRAIN':
            layer_params = net_param.layer
            for param in layer_params:
                print("Setting up layer {}".format(param.name))
                constructor = self.layer_map[param.type]
                constructor(param)
            print(layer_params)
        else:
            raise RuntimeError("Unsupported phase {}".format(phase))

    def forward_all(self, data=None):
        if data is not None:
            self.blobs['data'][...] = data
            self.blobs['data'].sync_ocl()
        for layer in self.layers:
            layer.forward()

net = Net(args.prototxt, args.caffemodel, args.phase)

im = caffe.io.load_image('data/cat.jpg')
transformer = caffe.io.Transformer(
    {'data': net.caffe_net.blobs['data'].data.shape})
transformer.set_mean(
    'data', np.load('models/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_raw_scale('data', 255.0)
data = np.asarray([transformer.preprocess('data', im)]).view(hmarray)
data.sync_ocl()

print("HM forward")
start = time.clock()
net.forward_all(data)
end = time.clock()
print("Time:", end - start)
print("Done")
print("Caffe forward")
start = time.clock()
out = net.caffe_net.forward_all(data=data)
end = time.clock()
print("Time:", end - start)
print("Done")

for blob_name in net.blobs.keys():
    print "Checking blob ", blob_name
    blob = net.blobs[blob_name]
    blob.sync_host()
    np.testing.assert_array_almost_equal(
        blob, net.caffe_net.blobs[blob_name].data, decimal=3)
print("SUCCESS")
