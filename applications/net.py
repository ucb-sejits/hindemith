import caffe_pb2 as pb
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


file_path = "models/bvlc_alexnet.caffemodel"

layer_map = {
    pb.V1LayerParameter.CONVOLUTION: ConvLayer,
    pb.V1LayerParameter.RELU: ReluLayer,
    pb.V1LayerParameter.POOLING: PoolingLayer,
    pb.V1LayerParameter.SOFTMAX: SoftmaxLayer,
    pb.V1LayerParameter.SOFTMAX_LOSS: SoftmaxLayer,
    pb.V1LayerParameter.INNER_PRODUCT: InnerProductLayer,
    pb.V1LayerParameter.LRN: LrnLayer
}

print("Reading in caffemodel")
net = caffemodel_to_net(file_path)

print("Initialing data")
db_path = "/storage2/datasets/ilsvrc2012_train_256x256_lmdb"
env = lmdb.Environment(db_path, readonly=True, lock=False)

PHASE = "train"
crop_size = 227
num_img = 64

# Data layer
data = np.ndarray((num_img, 3, 277, 277), np.float32)
data = []
label = hmarray((num_img, 1))

txn = env.begin()
cursor = txn.cursor().iternext()
datum = pb.Datum()
for i in range(num_img):
    datum.ParseFromString(next(cursor)[1])
    channels, datum_height, datum_width = datum.channels, datum.height, \
        datum.width
    height = datum_height
    width = datum_width
    height = crop_size
    width = crop_size
    if PHASE == "train":
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
    # data[i] = uncropped[..., h_off:h_off + height, w_off:w_off + width]
    data.append(uncropped[..., h_off:h_off + height, w_off:w_off + width])
    label[i] = datum.label

data = np.array(data).view(hmarray)
label.sync_ocl()

blobs = {
    "data": data,
    "label": label
}

layers = []
print("Initializing layers and blobs")
for layer_param in net.layers[1:]:  # Skip data layer
    print("Setting up layer {}".format(layer_param.name))
    if layer_param.type == pb.V1LayerParameter.DROPOUT:
        # Skip dropout layers
        continue
    layer = layer_map[layer_param.type](layer_param)
    layers.append(layer)
    bottom = []
    for blob in layer_param.bottom:
        bottom.append(blobs[blob])
    tops = layer.set_up(*bottom)
    for top, top_name in zip(tops, layer_param.top):
        blobs[top_name] = top
print(layers)
print(blobs)

