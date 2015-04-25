from hmcaffe.layers.base_layer import Layer
import lmdb
import random
import numpy as np
import hmcaffe.proto.caffe_pb2 as pb
from hindemith.types import hmarray


class DataLayer(Layer):
    def __init__(self, layer_param, phase):
        self.phase = phase
        db_path = layer_param.data_param.source
        env = lmdb.Environment(db_path, readonly=True, lock=False)

        self.batch_size = layer_param.data_param.batch_size
        self.scale = layer_param.transform_param.scale
        self.crop_size = layer_param.transform_param.crop_size
        txn = env.begin()
        self.cursor = txn.cursor().iternext()
        if layer_param.transform_param.HasField("mean_file"):
            blob_proto = pb.BlobProto()
            with open(layer_param.transform_param.mean_file, "rb") as f:
                blob_proto.ParseFromString(f.read())
                # FIXME: Assuming float32
                self.mean = np.array(
                    blob_proto.data._values).astype(np.float32).view(hmarray)
        else:
            self.mean = None

    def set_up(self):
        datum = pb.Datum()
        datum.ParseFromString(next(self.cursor)[1])
        self.mean = self.mean.reshape(datum.channels, datum.height, datum.width)
        height, width = datum.height, datum.width
        if self.crop_size:
            height, width = self.crop_size, self.crop_size
        self.data = hmarray((self.batch_size, datum.channels, height, width))
        self.label = hmarray((self.batch_size, ))
        return [(self.data, None), (self.label, None)]

    def forward(self):
        datum = pb.Datum()
        crop_size = self.crop_size
        for i in range(self.batch_size):
            datum.ParseFromString(next(self.cursor)[1])
            channels, datum_height, datum_width = datum.channels, \
                datum.height, datum.width
            height = datum_height
            width = datum_width
            height = crop_size
            width = crop_size
            h_off = random.randrange(datum_height - crop_size + 1)
            w_off = random.randrange(datum_width - crop_size + 1)
            # h_off = (datum_height - crop_size) / 2
            # w_off = (datum_width - crop_size) / 2
            uncropped = np.fromstring(
                datum.data, dtype=np.uint8
            ).astype(np.float32).reshape(channels, datum_height, datum_width)
            if self.mean is not None:
                uncropped = (uncropped - self.mean) * self.scale
            for c in range(channels):
                uncropped[c] = np.fliplr(uncropped[c])
            self.data[i] = uncropped[
                ..., h_off:h_off + height, w_off:w_off + width]
            self.label[i] = datum.label
        self.data.sync_ocl()
        self.label.sync_ocl()

    def backward(self):
        pass
