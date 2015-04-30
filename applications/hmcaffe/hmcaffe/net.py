"""
python hmcaffe/net.py --prototxt="models/alexnet-ng/deploy.prototxt" \
    --caffemodel="models/alexnet-ng/alexnet-ng.caffemodel" --phase='TEST'
python hmcaffe/net.py --prototxt="models/alexnet-ng/trainval.prototxt" \
    --caffemodel="models/alexnet-ng/alexnet-ng.caffemodel" --phase='TRAIN'
python hmcaffe/net.py \
    --prototxt="models/alexnet-ng/trainval-nodropout.prototxt" \
    --caffemodel="models/alexnet-ng/alexnet-ng.caffemodel" --phase='TRAIN'
"""
import hmcaffe.proto.caffe_pb2 as pb

import argparse
import caffe
from google.protobuf import text_format
from layers import ConvLayer, ReluLayer, PoolingLayer, InnerProductLayer, \
    SoftmaxLayer, LrnLayer, DataLayer, DropoutLayer, AccuracyLayer, \
    SoftmaxWithLossLayer, ConcatLayer
import numpy as np
from hindemith.types import hmarray
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("hmcaffe")


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
        "Concat": ConcatLayer,
        pb.V1LayerParameter.DROPOUT: DropoutLayer,
        pb.V1LayerParameter.ACCURACY: AccuracyLayer,
        pb.V1LayerParameter.DATA: DataLayer,
        "Data": DataLayer
    }

    def __init__(self, net_param, params=None):
        self.blobs = {}
        self.layers = []

        self.net_param = net_param

        if self.net_param.state.phase == pb.TEST:
            if len(self.net_param.input_dim) > 0:
                self.blobs['data'] = hmarray.zeros(self.net_param.input_dim)
                self.blobs['data_diff'] = None
        if len(self.net_param.layer) > 0:
            layer_params = self.net_param.layer
        else:
            layer_params = self.net_param.layers
        # Initialize layers
        for layer_param in layer_params:
            if len(layer_param.include) > 0 and \
                    layer_param.include[0].phase != net_param.state.phase:
                continue
            log.info("Initializing layer %s", layer_param.name)
            log.debug("  Bottom : %s", ", ".join(layer_param.bottom))
            log.debug("  Top    : %s", ", ".join(layer_param.top))
            # Skip dropout layers for test
            if layer_param.type in (pb.V1LayerParameter.DROPOUT, "Dropout") \
                    and net_param.state.phase == pb.TEST:
                continue
            layer_constructor = self.layer_map[layer_param.type]
            if params is not None and layer_param.name in params:
                layer = layer_constructor(
                    layer_param, net_param.state.phase,
                    params[layer_param.name])
            else:
                layer = layer_constructor(layer_param, net_param.state.phase)
            self.layers.append(layer)
            bottom = []
            for blob in layer_param.bottom:
                bottom.append(self.blobs[blob])
                bottom.append(self.blobs["{}_diff".format(blob)])
            tops = layer.set_up(*bottom)
            for top, top_name in zip(tops, layer_param.top):
                self.blobs[top_name] = top[0]
                self.blobs["{}_diff".format(top_name)] = top[1]

    def forward(self, _from=0, to=None):
        if to is None:
            to = len(self.layers)
        for layer in self.layers[_from:to]:
            layer.forward()

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def forward_all(self, **kwargs):
        for key, value in kwargs.iteritems():
            self.blobs[key][...] = value
            self.blobs[key].sync_ocl()
        for layer in self.layers:
            # start = time.time()
            layer.forward()
            # print("{} forward {}".format(layer.layer_param.name, time.time() - start))

    def forward_backward_all(self, **kwargs):
        for key, value in kwargs.iteritems():
            self.blobs[key][...] = value
            self.blobs[key].sync_ocl()
        for layer in self.layers:
            layer.forward()
        for layer in reversed(self.layers):
            layer.backward()

    def forward_backward(self):
        loss = 0
        for layer in self.layers:
            loss += layer.forward_with_loss()
        for layer in reversed(self.layers):
            layer.backward()
        return loss

    def update_params(self, rate, weight_decay, momentum):
        for layer in self.layers:
            layer.update_params(rate, weight_decay, momentum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Example usage:
python net.py --prototxt="models/alexnet-ng/deploy.prototxt" \
--caffemodel="models/alexnet-ng/alexnet-ng.caffemodel" --phase='TEST'"""
    )
    parser.add_argument(
        '--prototxt',
        help="path to prototxt using Caffe's format for network description",
        required=True
    )
    parser.add_argument(
        '--caffemodel',
        help="path to .caffemodel to use for network initialization",
        required=True
    )
    parser.add_argument(
        '--phase',
        help="TRAIN or TEST",
        default='TEST'
    )
    args = parser.parse_args()

    caffe.set_mode_gpu()
    caffe_net = caffe.Net(args.prototxt, args.caffemodel,
                          getattr(caffe, args.phase))
    net_param = pb.NetParameter()
    with open(args.prototxt, "rb") as f:
        text_format.Merge(f.read(), net_param)
    if args.phase == 'TRAIN':
        net_param.state.phase = pb.TRAIN
    net = Net(net_param, caffe_net.params)

    if args.phase == 'TRAIN':
        caffe_net.forward()
        # Skip Data Layer forward and initialize with Caffe's batch
        net.layers[0].data[:] = caffe_net.blobs['data'].data
        net.layers[0].data.sync_ocl()
        net.layers[0].label[:] = caffe_net.blobs['label'].data
        net.layers[0].label.sync_ocl()
        net.forward(1)  # skip first (data) layer
        caffe_net.blobs['loss'].diff[...] = 100.0
        caffe_net.backward()
        net.backward()
    else:
        if net_param.name == "AlexNet":
            im = caffe.io.load_image('data/cat.jpg')
            transformer = caffe.io.Transformer(
                {'data': caffe_net.blobs['data'].data.shape})
            transformer.set_mean(
                'data', np.load('models/ilsvrc_2012_mean.npy').mean(1).mean(1))
            transformer.set_transpose('data', (2, 0, 1))
            transformer.set_channel_swap('data', (2, 1, 0))
            transformer.set_raw_scale('data', 255.0)
            data = np.asarray(
                [transformer.preprocess('data', im)]).view(hmarray)
            data.sync_ocl()
        net.forward_all(data=data)
        caffe_net.forward_all(data=data)

    failed = False
    for blob_name in net.blobs.keys():
        blob = net.blobs[blob_name]
        if blob is None:
            continue
        blob.sync_host()
        print("Checking blob " + blob_name)
        try:
            if "_diff" in blob_name:
                caffe_blob = caffe_net.blobs[blob_name[:-5]].diff
                # scale = np.max(caffe_blob)
                # if scale > 0:
                #     caffe_blob /= scale
                #     blob /= scale
                if args.phase == 'TEST':
                    continue
            else:
                caffe_blob = caffe_net.blobs[blob_name].data
            np.testing.assert_array_almost_equal(blob, caffe_blob,
                                                 decimal=4)
        except AssertionError as e:
            print(e)
            failed = True

    if failed:
        print("FAILED")
        from IPython import embed
        embed()
    else:
        print("SUCCESS")
