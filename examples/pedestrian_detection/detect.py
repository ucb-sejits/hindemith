#!/usr/bin/env python
"""
detector.py is an out-of-the-box windowed detector
callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
Note that this model was trained for image classification and not detection,
and finetuning for detection can be expected to improve results.

The selective_search_ijcv_with_python code required for the selective search
proposal mode is available at
    https://github.com/sergeyk/selective_search_ijcv_with_python

TODO:
- batch up image filenames as well: don't want to load all of them into memory
- come up with a batching scheme that preserved order / keeps a unique ID
"""
import numpy as np
import pandas as pd
import os

import caffe
import cv2

COORD_COLS = ['ymin', 'xmin', 'ymax', 'xmax']

with open('det_synset_words.txt') as f:
    labels_df = pd.DataFrame([
        {
            'synset_id': l.strip().split(' ')[0],
            'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
        }
        for l in f.readlines()
    ])
labels_df.sort('synset_id')

model_def = os.path.join(
    "models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt")
pretrained_model = os.path.join("models/bvlc_reference_rcnn_ilsvrc13/" +
                                "bvlc_reference_rcnn_ilsvrc13.caffemodel")
mean_file = 'models/ilsvrc_2012_mean.npy'

mean = None
mean = np.load(mean_file)

# Make detector.
detector = caffe.Detector(model_def, pretrained_model, gpu=True, mean=mean,
                          input_scale=None, raw_scale=255.0,
                          channel_swap=[2, 1, 0], context_pad=16)


def detect_windows(self, windows, image):
    # Extract windows.
    window_inputs = []
    for window in windows:
        window_inputs.append(self.crop(image, window))

    # Run through the net (warping windows to input dimensions).
    caffe_in = np.zeros((len(window_inputs), window_inputs[0].shape[2])
                        + self.blobs[self.inputs[0]].data.shape[2:],
                        dtype=np.float32)
    for ix, window_in in enumerate(window_inputs):
        caffe_in[ix] = self.preprocess(self.inputs[0], window_in)
    out = self.forward_all(**{self.inputs[0]: caffe_in})
    predictions = out[self.outputs[0]].squeeze(axis=(2, 3))

    # Package predictions with images and windows.
    detections = []
    ix = 0
    for window in windows:
        detections.append({
            'window': window,
            'prediction': predictions[ix],
            'filename': 'a'
        })
        ix += 1
    return detections


def run(images_windows, image):
    # pycaffe_dir = os.path.dirname(__file__)

    # print(images_windows)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) * 1./255
    # img = cv2.normalize(image, None, 0., 1., cv2.NORM_MINMAX, cv2.CV_64FC1)
    detections = detect_windows(detector, images_windows, img)

    # Collect into dataframe with labeled fields.
    df = pd.DataFrame(detections)
    df.set_index('filename', inplace=True)
    df[COORD_COLS] = pd.DataFrame(
        data=np.vstack(df['window']), index=df.index, columns=COORD_COLS)
    del(df['window'])
    predictions_df = pd.DataFrame(np.vstack(df.prediction.values),
                                  columns=labels_df['name'])
    max_s = predictions_df.max(0)
    max_s.sort(ascending=False)
    for index in range(len(df)):
        det = df.iloc[index]
        # print(labels_df['name'][np.argmax(det['prediction'])])
        # print(det['prediction'][124])
        # print(labels_df['name'][124])
        if labels_df['name'][np.argmax(det['prediction'])] == 'person':
            cv2.rectangle(image, (det['xmin'], det['ymax']),
                          (det["xmax"], det['ymin']), (0, 255, 0), 2)
