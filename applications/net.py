import caffe_pb2 as pb
from google.protobuf import text_format
from layers import ConvLayer, ReluLayer, PoolingLayer, InnerProductLayer, \
    SoftmaxLayer, LrnLayer
import lmdb
import numpy as np
import random
import math
from hindemith.types import hmarray


def caffemodel_to_net(file_path):
    net = pb.NetParameter()
    with open(file_path, "rb") as f:
        net.ParseFromString(f.read())
    return net

def prototxt_to_net(file_path):
    net = pb.NetParameter()
    with open(file_path, "rb") as f:
        text_format.Merge(f.read(), net)
    return net


file_path = "models/lenet/deploy.prototxt"

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
    "LRN": LrnLayer
}

print("Reading in prototxt")
net = prototxt_to_net(file_path)

print("Initialing data")
# db_path = "/storage2/datasets/ilsvrc2012_train_256x256_lmdb"
# env = lmdb.Environment(db_path, readonly=True, lock=False)
db_path = "data/mnist_train_lmdb_clean"
env = lmdb.Environment(db_path, readonly=True, lock=False)

PHASE = "train"
batch_size = 10
scale = 1.0 / 256.0
data = hmarray((batch_size, 1, 28, 28))
txn = env.begin()
cursor = txn.cursor().iternext()
datum = pb.Datum()

for i in range(batch_size):
    datum.ParseFromString(next(cursor)[1])
    unscaled = np.fromstring(
        datum.data, dtype=np.uint8).astype(np.float32).reshape(1, 28, 28)
    data[i] = unscaled * scale
    # label[i] = datum.label
data.sync_ocl()
# crop_size = 227
# num_img = 64

# Data layer
# data = np.ndarray((num_img, 3, 277, 277), np.float32)
# data = []
# label = hmarray((num_img, 1))

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

# data = np.array(data).view(hmarray)
# label.sync_ocl()

# blobs = {
#     "data": data,
#     "label": label
# }

import caffe
caffe_net = caffe.Net('models/lenet/deploy.prototxt',
                      'models/lenet/lenet_iter_5000.caffemodel',
                      caffe.TEST)
# im = caffe.io.load_image('data/cat.jpg')
# transformer = caffe.io.Transformer({'data': caffe_net.blobs['data'].data.shape})
# transformer.set_mean('data', np.load('models/ilsvrc_2012_mean.npy').mean(1).mean(1))
# transformer.set_transpose('data', (2,0,1))
# transformer.set_channel_swap('data', (2,1,0))
# transformer.set_raw_scale('data', 255.0)

blobs = {
    "data": data.view(hmarray)
}

out = caffe_net.forward_all(data=data)

layers = []
print("Initializing layers and blobs")
for layer_param in net.layer:  # Skip data layer
    print("Setting up layer {}".format(layer_param.name))
    if layer_param.type in (pb.V1LayerParameter.DROPOUT, "Dropout"):
        # Skip dropout layers
        continue
    if layer_param.name in caffe_net.params:
        layer = layer_map[layer_param.type](layer_param,
                                            caffe_net.params[layer_param.name])
    else:
        layer = layer_map[layer_param.type](layer_param)
    layers.append(layer)
    bottom = []
    for blob in layer_param.bottom:
        bottom.append(blobs[blob])
    tops = layer.set_up(*bottom)
    for top, top_name in zip(tops, layer_param.top):
        blobs[top_name] = top

for layer in layers:
    layer.forward()
blobs['prob'].sync_host()
for i in range(batch_size):
    np.testing.assert_array_almost_equal(blobs['prob'][i],
                                         caffe_net.blobs['prob'].data[i])
print("SUCCESS")

# blobs['conv1'].sync_host()
# np.testing.assert_array_almost_equal(blobs['conv1'][0],
#                                      caffe_net.blobs['conv1'].data[0],
#                                      decimal=3)
