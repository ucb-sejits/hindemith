import lmdb
import random
import numpy as np
import caffe_pb2 as pb
from hindemith.types import hmarray


class DataLayer(object):
    def __init__(self, layer_param, phase):
        self.phase = phase
        db_path = layer_param.data_param.source
        env = lmdb.Environment(db_path, readonly=True, lock=False)

        self.batch_size = layer_param.data_param.batch_size
        self.scale = layer_param.transform_param.scale
        self.crop_size = layer_param.transform_param.crop_size
        txn = env.begin()
        self.cursor = txn.cursor().iternext()

    def set_up(self):
        datum = pb.Datum()
        datum.ParseFromString(next(self.cursor)[1])
        height, width = datum.height, datum.width
        if self.crop_size:
            height, width = self.crop_size, self.crop_size
        self.data = hmarray((self.batch_size, datum.channels, height, width))
        self.data_diff = hmarray((self.batch_size, datum.channels, height,
                                  width))
        self.label = hmarray((self.batch_size, ))
        return [(self.data, self.data_diff), (self.label, None)]

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
            for c in range(channels):
                uncropped[c] = np.fliplr(uncropped[c])
            self.data[i] = uncropped[
                ..., h_off:h_off + height, w_off:w_off + width]
            self.label[i] = datum.label
        self.data.sync_ocl()
        self.label.sync_ocl()
