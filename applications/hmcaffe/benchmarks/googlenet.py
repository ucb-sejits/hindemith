from hindemith.types import hmarray
from hindemith.operations.conv import ConvForward
from hindemith.operations.relu import ReluForward
from hindemith.operations.pool import PoolForward, AvePoolForward
from hindemith.operations.lrn import LrnForward
from hindemith.operations.softmax import SoftmaxForward
from hindemith.operations.concat import ConcatForward
from hindemith.operations.inner_product import InnerProductForward
from hindemith.core import compose
from hindemith.cl import queue
import pycl as cl
import caffe
import numpy as np
import time

prototxt = "benchmarks/googlenet.prototxt"
caffemodel = "models/googlenet/bvlc_googlenet.caffemodel"

caffe.set_mode_gpu()
caffe.set_device(1)
caffe_net = caffe.Net(prototxt, caffemodel, caffe.TEST)


conv1_7x7_s2_filters = caffe_net.params['conv1/7x7_s2'][0].data.view(hmarray)
conv1_7x7_s2_bias = caffe_net.params['conv1/7x7_s2'][1].data.view(hmarray)
conv1_7x7_s2 = hmarray.zeros(caffe_net.blobs['conv1/7x7_s2'].data.shape)

pool1_3x3_s2 = hmarray.zeros(caffe_net.blobs['pool1/3x3_s2'].data.shape)
pool1_3x3_s2_mask = hmarray.zeros(pool1_3x3_s2.shape)

pool1_norm1 = hmarray.zeros(caffe_net.blobs['pool1/norm1'].data.shape)
pool1_norm1_scale = hmarray.zeros(pool1_norm1.shape)

conv2_3x3_reduce_filters = caffe_net.params['conv2/3x3_reduce'][0].data.view(hmarray)
conv2_3x3_reduce_bias = caffe_net.params['conv2/3x3_reduce'][1].data.view(hmarray)
conv2_3x3_reduce = hmarray.zeros(caffe_net.blobs['conv2/3x3_reduce'].data.shape)

conv2_3x3_filters = caffe_net.params['conv2/3x3'][0].data.view(hmarray)
conv2_3x3_bias = caffe_net.params['conv2/3x3'][1].data.view(hmarray)
conv2_3x3 = hmarray.zeros(caffe_net.blobs['conv2/3x3'].data.shape)

conv2_norm2 = hmarray.zeros(caffe_net.blobs['conv2/norm2'].data.shape)
conv2_norm2_scale = hmarray.zeros(conv2_norm2.shape)

pool2_3x3_s2 = hmarray.zeros(caffe_net.blobs['pool2/3x3_s2'].data.shape)
pool2_3x3_s2_mask = hmarray.zeros(pool2_3x3_s2.shape)

inception_3a_1x1_filters = caffe_net.params['inception_3a/1x1'][0].data.view(hmarray)
inception_3a_1x1_bias = caffe_net.params['inception_3a/1x1'][1].data.view(hmarray)
inception_3a_1x1 = hmarray.zeros(caffe_net.blobs['inception_3a/1x1'].data.shape)

inception_3a_3x3_reduce_filters = caffe_net.params['inception_3a/3x3_reduce'][0].data.view(hmarray)
inception_3a_3x3_reduce_bias = caffe_net.params['inception_3a/3x3_reduce'][1].data.view(hmarray)
inception_3a_3x3_reduce = hmarray.zeros(caffe_net.blobs['inception_3a/3x3_reduce'].data.shape)

inception_3a_3x3_filters = caffe_net.params['inception_3a/3x3'][0].data.view(hmarray)
inception_3a_3x3_bias = caffe_net.params['inception_3a/3x3'][1].data.view(hmarray)
inception_3a_3x3 = hmarray.zeros(caffe_net.blobs['inception_3a/3x3'].data.shape)

inception_3a_5x5_reduce_filters = caffe_net.params['inception_3a/5x5_reduce'][0].data.view(hmarray)
inception_3a_5x5_reduce_bias = caffe_net.params['inception_3a/5x5_reduce'][1].data.view(hmarray)
inception_3a_5x5_reduce = hmarray.zeros(caffe_net.blobs['inception_3a/5x5_reduce'].data.shape)

inception_3a_5x5_filters = caffe_net.params['inception_3a/5x5'][0].data.view(hmarray)
inception_3a_5x5_bias = caffe_net.params['inception_3a/5x5'][1].data.view(hmarray)
inception_3a_5x5 = hmarray.zeros(caffe_net.blobs['inception_3a/5x5'].data.shape)

inception_3a_pool = hmarray.zeros(caffe_net.blobs['inception_3a/pool'].data.shape)
inception_3a_pool_mask = hmarray.zeros(inception_3a_pool.shape)

inception_3a_pool_proj_filters = caffe_net.params['inception_3a/pool_proj'][0].data.view(hmarray)
inception_3a_pool_proj_bias = caffe_net.params['inception_3a/pool_proj'][1].data.view(hmarray)
inception_3a_pool_proj = hmarray.zeros(caffe_net.blobs['inception_3a/pool_proj'].data.shape)

inception_3a_output = hmarray.zeros(caffe_net.blobs['inception_3a/output'].data.shape)

inception_3b_1x1_filters = caffe_net.params['inception_3b/1x1'][0].data.view(hmarray)
inception_3b_1x1_bias = caffe_net.params['inception_3b/1x1'][1].data.view(hmarray)
inception_3b_1x1 = hmarray.zeros(caffe_net.blobs['inception_3b/1x1'].data.shape)

inception_3b_3x3_reduce_filters = caffe_net.params['inception_3b/3x3_reduce'][0].data.view(hmarray)
inception_3b_3x3_reduce_bias = caffe_net.params['inception_3b/3x3_reduce'][1].data.view(hmarray)
inception_3b_3x3_reduce = hmarray.zeros(caffe_net.blobs['inception_3b/3x3_reduce'].data.shape)

inception_3b_3x3_filters = caffe_net.params['inception_3b/3x3'][0].data.view(hmarray)
inception_3b_3x3_bias = caffe_net.params['inception_3b/3x3'][1].data.view(hmarray)
inception_3b_3x3 = hmarray.zeros(caffe_net.blobs['inception_3b/3x3'].data.shape)

inception_3b_5x5_reduce_filters = caffe_net.params['inception_3b/5x5_reduce'][0].data.view(hmarray)
inception_3b_5x5_reduce_bias = caffe_net.params['inception_3b/5x5_reduce'][1].data.view(hmarray)
inception_3b_5x5_reduce = hmarray.zeros(caffe_net.blobs['inception_3b/5x5_reduce'].data.shape)

inception_3b_5x5_filters = caffe_net.params['inception_3b/5x5'][0].data.view(hmarray)
inception_3b_5x5_bias = caffe_net.params['inception_3b/5x5'][1].data.view(hmarray)
inception_3b_5x5 = hmarray.zeros(caffe_net.blobs['inception_3b/5x5'].data.shape)

inception_3b_pool = hmarray.zeros(caffe_net.blobs['inception_3b/pool'].data.shape)
inception_3b_pool_mask = hmarray.zeros(inception_3b_pool.shape)

inception_3b_pool_proj_filters = caffe_net.params['inception_3b/pool_proj'][0].data.view(hmarray)
inception_3b_pool_proj_bias = caffe_net.params['inception_3b/pool_proj'][1].data.view(hmarray)
inception_3b_pool_proj = hmarray.zeros(caffe_net.blobs['inception_3b/pool_proj'].data.shape)

inception_3b_output = hmarray.zeros(caffe_net.blobs['inception_3b/output'].data.shape)

pool3_3x3_s2 = hmarray.zeros(caffe_net.blobs['pool3/3x3_s2'].data.shape)
pool3_3x3_s2_mask = hmarray.zeros(pool3_3x3_s2.shape)

inception_4a_1x1_filters = caffe_net.params['inception_4a/1x1'][0].data.view(hmarray)
inception_4a_1x1_bias = caffe_net.params['inception_4a/1x1'][1].data.view(hmarray)
inception_4a_1x1 = hmarray.zeros(caffe_net.blobs['inception_4a/1x1'].data.shape)

inception_4a_3x3_reduce_filters = caffe_net.params['inception_4a/3x3_reduce'][0].data.view(hmarray)
inception_4a_3x3_reduce_bias = caffe_net.params['inception_4a/3x3_reduce'][1].data.view(hmarray)
inception_4a_3x3_reduce = hmarray.zeros(caffe_net.blobs['inception_4a/3x3_reduce'].data.shape)

inception_4a_3x3_filters = caffe_net.params['inception_4a/3x3'][0].data.view(hmarray)
inception_4a_3x3_bias = caffe_net.params['inception_4a/3x3'][1].data.view(hmarray)
inception_4a_3x3 = hmarray.zeros(caffe_net.blobs['inception_4a/3x3'].data.shape)

inception_4a_5x5_reduce_filters = caffe_net.params['inception_4a/5x5_reduce'][0].data.view(hmarray)
inception_4a_5x5_reduce_bias = caffe_net.params['inception_4a/5x5_reduce'][1].data.view(hmarray)
inception_4a_5x5_reduce = hmarray.zeros(caffe_net.blobs['inception_4a/5x5_reduce'].data.shape)

inception_4a_5x5_filters = caffe_net.params['inception_4a/5x5'][0].data.view(hmarray)
inception_4a_5x5_bias = caffe_net.params['inception_4a/5x5'][1].data.view(hmarray)
inception_4a_5x5 = hmarray.zeros(caffe_net.blobs['inception_4a/5x5'].data.shape)

inception_4a_pool = hmarray.zeros(caffe_net.blobs['inception_4a/pool'].data.shape)
inception_4a_pool_mask = hmarray.zeros(inception_4a_pool.shape)

inception_4a_pool_proj_filters = caffe_net.params['inception_4a/pool_proj'][0].data.view(hmarray)
inception_4a_pool_proj_bias = caffe_net.params['inception_4a/pool_proj'][1].data.view(hmarray)
inception_4a_pool_proj = hmarray.zeros(caffe_net.blobs['inception_4a/pool_proj'].data.shape)

inception_4a_output = hmarray.zeros(caffe_net.blobs['inception_4a/output'].data.shape)

inception_4b_1x1_filters = caffe_net.params['inception_4b/1x1'][0].data.view(hmarray)
inception_4b_1x1_bias = caffe_net.params['inception_4b/1x1'][1].data.view(hmarray)
inception_4b_1x1 = hmarray.zeros(caffe_net.blobs['inception_4b/1x1'].data.shape)

inception_4b_3x3_reduce_filters = caffe_net.params['inception_4b/3x3_reduce'][0].data.view(hmarray)
inception_4b_3x3_reduce_bias = caffe_net.params['inception_4b/3x3_reduce'][1].data.view(hmarray)
inception_4b_3x3_reduce = hmarray.zeros(caffe_net.blobs['inception_4b/3x3_reduce'].data.shape)

inception_4b_3x3_filters = caffe_net.params['inception_4b/3x3'][0].data.view(hmarray)
inception_4b_3x3_bias = caffe_net.params['inception_4b/3x3'][1].data.view(hmarray)
inception_4b_3x3 = hmarray.zeros(caffe_net.blobs['inception_4b/3x3'].data.shape)

inception_4b_5x5_reduce_filters = caffe_net.params['inception_4b/5x5_reduce'][0].data.view(hmarray)
inception_4b_5x5_reduce_bias = caffe_net.params['inception_4b/5x5_reduce'][1].data.view(hmarray)
inception_4b_5x5_reduce = hmarray.zeros(caffe_net.blobs['inception_4b/5x5_reduce'].data.shape)

inception_4b_5x5_filters = caffe_net.params['inception_4b/5x5'][0].data.view(hmarray)
inception_4b_5x5_bias = caffe_net.params['inception_4b/5x5'][1].data.view(hmarray)
inception_4b_5x5 = hmarray.zeros(caffe_net.blobs['inception_4b/5x5'].data.shape)

inception_4b_pool = hmarray.zeros(caffe_net.blobs['inception_4b/pool'].data.shape)
inception_4b_pool_mask = hmarray.zeros(inception_4b_pool.shape)

inception_4b_pool_proj_filters = caffe_net.params['inception_4b/pool_proj'][0].data.view(hmarray)
inception_4b_pool_proj_bias = caffe_net.params['inception_4b/pool_proj'][1].data.view(hmarray)
inception_4b_pool_proj = hmarray.zeros(caffe_net.blobs['inception_4b/pool_proj'].data.shape)

inception_4b_output = hmarray.zeros(caffe_net.blobs['inception_4b/output'].data.shape)

inception_4c_1x1_filters = caffe_net.params['inception_4c/1x1'][0].data.view(hmarray)
inception_4c_1x1_bias = caffe_net.params['inception_4c/1x1'][1].data.view(hmarray)
inception_4c_1x1 = hmarray.zeros(caffe_net.blobs['inception_4c/1x1'].data.shape)

inception_4c_3x3_reduce_filters = caffe_net.params['inception_4c/3x3_reduce'][0].data.view(hmarray)
inception_4c_3x3_reduce_bias = caffe_net.params['inception_4c/3x3_reduce'][1].data.view(hmarray)
inception_4c_3x3_reduce = hmarray.zeros(caffe_net.blobs['inception_4c/3x3_reduce'].data.shape)

inception_4c_3x3_filters = caffe_net.params['inception_4c/3x3'][0].data.view(hmarray)
inception_4c_3x3_bias = caffe_net.params['inception_4c/3x3'][1].data.view(hmarray)
inception_4c_3x3 = hmarray.zeros(caffe_net.blobs['inception_4c/3x3'].data.shape)

inception_4c_5x5_reduce_filters = caffe_net.params['inception_4c/5x5_reduce'][0].data.view(hmarray)
inception_4c_5x5_reduce_bias = caffe_net.params['inception_4c/5x5_reduce'][1].data.view(hmarray)
inception_4c_5x5_reduce = hmarray.zeros(caffe_net.blobs['inception_4c/5x5_reduce'].data.shape)

inception_4c_5x5_filters = caffe_net.params['inception_4c/5x5'][0].data.view(hmarray)
inception_4c_5x5_bias = caffe_net.params['inception_4c/5x5'][1].data.view(hmarray)
inception_4c_5x5 = hmarray.zeros(caffe_net.blobs['inception_4c/5x5'].data.shape)

inception_4c_pool = hmarray.zeros(caffe_net.blobs['inception_4c/pool'].data.shape)
inception_4c_pool_mask = hmarray.zeros(inception_4c_pool.shape)

inception_4c_pool_proj_filters = caffe_net.params['inception_4c/pool_proj'][0].data.view(hmarray)
inception_4c_pool_proj_bias = caffe_net.params['inception_4c/pool_proj'][1].data.view(hmarray)
inception_4c_pool_proj = hmarray.zeros(caffe_net.blobs['inception_4c/pool_proj'].data.shape)

inception_4c_output = hmarray.zeros(caffe_net.blobs['inception_4c/output'].data.shape)

inception_4d_1x1_filters = caffe_net.params['inception_4d/1x1'][0].data.view(hmarray)
inception_4d_1x1_bias = caffe_net.params['inception_4d/1x1'][1].data.view(hmarray)
inception_4d_1x1 = hmarray.zeros(caffe_net.blobs['inception_4d/1x1'].data.shape)

inception_4d_3x3_reduce_filters = caffe_net.params['inception_4d/3x3_reduce'][0].data.view(hmarray)
inception_4d_3x3_reduce_bias = caffe_net.params['inception_4d/3x3_reduce'][1].data.view(hmarray)
inception_4d_3x3_reduce = hmarray.zeros(caffe_net.blobs['inception_4d/3x3_reduce'].data.shape)

inception_4d_3x3_filters = caffe_net.params['inception_4d/3x3'][0].data.view(hmarray)
inception_4d_3x3_bias = caffe_net.params['inception_4d/3x3'][1].data.view(hmarray)
inception_4d_3x3 = hmarray.zeros(caffe_net.blobs['inception_4d/3x3'].data.shape)

inception_4d_5x5_reduce_filters = caffe_net.params['inception_4d/5x5_reduce'][0].data.view(hmarray)
inception_4d_5x5_reduce_bias = caffe_net.params['inception_4d/5x5_reduce'][1].data.view(hmarray)
inception_4d_5x5_reduce = hmarray.zeros(caffe_net.blobs['inception_4d/5x5_reduce'].data.shape)

inception_4d_5x5_filters = caffe_net.params['inception_4d/5x5'][0].data.view(hmarray)
inception_4d_5x5_bias = caffe_net.params['inception_4d/5x5'][1].data.view(hmarray)
inception_4d_5x5 = hmarray.zeros(caffe_net.blobs['inception_4d/5x5'].data.shape)

inception_4d_pool = hmarray.zeros(caffe_net.blobs['inception_4d/pool'].data.shape)
inception_4d_pool_mask = hmarray.zeros(inception_4d_pool.shape)

inception_4d_pool_proj_filters = caffe_net.params['inception_4d/pool_proj'][0].data.view(hmarray)
inception_4d_pool_proj_bias = caffe_net.params['inception_4d/pool_proj'][1].data.view(hmarray)
inception_4d_pool_proj = hmarray.zeros(caffe_net.blobs['inception_4d/pool_proj'].data.shape)

inception_4d_output = hmarray.zeros(caffe_net.blobs['inception_4d/output'].data.shape)

inception_4e_1x1_filters = caffe_net.params['inception_4e/1x1'][0].data.view(hmarray)
inception_4e_1x1_bias = caffe_net.params['inception_4e/1x1'][1].data.view(hmarray)
inception_4e_1x1 = hmarray.zeros(caffe_net.blobs['inception_4e/1x1'].data.shape)

inception_4e_3x3_reduce_filters = caffe_net.params['inception_4e/3x3_reduce'][0].data.view(hmarray)
inception_4e_3x3_reduce_bias = caffe_net.params['inception_4e/3x3_reduce'][1].data.view(hmarray)
inception_4e_3x3_reduce = hmarray.zeros(caffe_net.blobs['inception_4e/3x3_reduce'].data.shape)

inception_4e_3x3_filters = caffe_net.params['inception_4e/3x3'][0].data.view(hmarray)
inception_4e_3x3_bias = caffe_net.params['inception_4e/3x3'][1].data.view(hmarray)
inception_4e_3x3 = hmarray.zeros(caffe_net.blobs['inception_4e/3x3'].data.shape)

inception_4e_5x5_reduce_filters = caffe_net.params['inception_4e/5x5_reduce'][0].data.view(hmarray)
inception_4e_5x5_reduce_bias = caffe_net.params['inception_4e/5x5_reduce'][1].data.view(hmarray)
inception_4e_5x5_reduce = hmarray.zeros(caffe_net.blobs['inception_4e/5x5_reduce'].data.shape)

inception_4e_5x5_filters = caffe_net.params['inception_4e/5x5'][0].data.view(hmarray)
inception_4e_5x5_bias = caffe_net.params['inception_4e/5x5'][1].data.view(hmarray)
inception_4e_5x5 = hmarray.zeros(caffe_net.blobs['inception_4e/5x5'].data.shape)

inception_4e_pool = hmarray.zeros(caffe_net.blobs['inception_4e/pool'].data.shape)
inception_4e_pool_mask = hmarray.zeros(inception_4e_pool.shape)

inception_4e_pool_proj_filters = caffe_net.params['inception_4e/pool_proj'][0].data.view(hmarray)
inception_4e_pool_proj_bias = caffe_net.params['inception_4e/pool_proj'][1].data.view(hmarray)
inception_4e_pool_proj = hmarray.zeros(caffe_net.blobs['inception_4e/pool_proj'].data.shape)

inception_4e_output = hmarray.zeros(caffe_net.blobs['inception_4e/output'].data.shape)

pool4_3x3_s2 = hmarray.zeros(caffe_net.blobs['pool4/3x3_s2'].data.shape)
pool4_3x3_s2_mask = hmarray.zeros(pool4_3x3_s2.shape)

inception_5a_1x1_filters = caffe_net.params['inception_5a/1x1'][0].data.view(hmarray)
inception_5a_1x1_bias = caffe_net.params['inception_5a/1x1'][1].data.view(hmarray)
inception_5a_1x1 = hmarray.zeros(caffe_net.blobs['inception_5a/1x1'].data.shape)

inception_5a_3x3_reduce_filters = caffe_net.params['inception_5a/3x3_reduce'][0].data.view(hmarray)
inception_5a_3x3_reduce_bias = caffe_net.params['inception_5a/3x3_reduce'][1].data.view(hmarray)
inception_5a_3x3_reduce = hmarray.zeros(caffe_net.blobs['inception_5a/3x3_reduce'].data.shape)

inception_5a_3x3_filters = caffe_net.params['inception_5a/3x3'][0].data.view(hmarray)
inception_5a_3x3_bias = caffe_net.params['inception_5a/3x3'][1].data.view(hmarray)
inception_5a_3x3 = hmarray.zeros(caffe_net.blobs['inception_5a/3x3'].data.shape)

inception_5a_5x5_reduce_filters = caffe_net.params['inception_5a/5x5_reduce'][0].data.view(hmarray)
inception_5a_5x5_reduce_bias = caffe_net.params['inception_5a/5x5_reduce'][1].data.view(hmarray)
inception_5a_5x5_reduce = hmarray.zeros(caffe_net.blobs['inception_5a/5x5_reduce'].data.shape)

inception_5a_5x5_filters = caffe_net.params['inception_5a/5x5'][0].data.view(hmarray)
inception_5a_5x5_bias = caffe_net.params['inception_5a/5x5'][1].data.view(hmarray)
inception_5a_5x5 = hmarray.zeros(caffe_net.blobs['inception_5a/5x5'].data.shape)

inception_5a_pool = hmarray.zeros(caffe_net.blobs['inception_5a/pool'].data.shape)
inception_5a_pool_mask = hmarray.zeros(inception_5a_pool.shape)

inception_5a_pool_proj_filters = caffe_net.params['inception_5a/pool_proj'][0].data.view(hmarray)
inception_5a_pool_proj_bias = caffe_net.params['inception_5a/pool_proj'][1].data.view(hmarray)
inception_5a_pool_proj = hmarray.zeros(caffe_net.blobs['inception_5a/pool_proj'].data.shape)

inception_5a_output = hmarray.zeros(caffe_net.blobs['inception_5a/output'].data.shape)

inception_5b_1x1_filters = caffe_net.params['inception_5b/1x1'][0].data.view(hmarray)
inception_5b_1x1_bias = caffe_net.params['inception_5b/1x1'][1].data.view(hmarray)
inception_5b_1x1 = hmarray.zeros(caffe_net.blobs['inception_5b/1x1'].data.shape)

inception_5b_3x3_reduce_filters = caffe_net.params['inception_5b/3x3_reduce'][0].data.view(hmarray)
inception_5b_3x3_reduce_bias = caffe_net.params['inception_5b/3x3_reduce'][1].data.view(hmarray)
inception_5b_3x3_reduce = hmarray.zeros(caffe_net.blobs['inception_5b/3x3_reduce'].data.shape)

inception_5b_3x3_filters = caffe_net.params['inception_5b/3x3'][0].data.view(hmarray)
inception_5b_3x3_bias = caffe_net.params['inception_5b/3x3'][1].data.view(hmarray)
inception_5b_3x3 = hmarray.zeros(caffe_net.blobs['inception_5b/3x3'].data.shape)

inception_5b_5x5_reduce_filters = caffe_net.params['inception_5b/5x5_reduce'][0].data.view(hmarray)
inception_5b_5x5_reduce_bias = caffe_net.params['inception_5b/5x5_reduce'][1].data.view(hmarray)
inception_5b_5x5_reduce = hmarray.zeros(caffe_net.blobs['inception_5b/5x5_reduce'].data.shape)

inception_5b_5x5_filters = caffe_net.params['inception_5b/5x5'][0].data.view(hmarray)
inception_5b_5x5_bias = caffe_net.params['inception_5b/5x5'][1].data.view(hmarray)
inception_5b_5x5 = hmarray.zeros(caffe_net.blobs['inception_5b/5x5'].data.shape)

inception_5b_pool = hmarray.zeros(caffe_net.blobs['inception_5b/pool'].data.shape)
inception_5b_pool_mask = hmarray.zeros(inception_5b_pool.shape)

inception_5b_pool_proj_filters = caffe_net.params['inception_5b/pool_proj'][0].data.view(hmarray)
inception_5b_pool_proj_bias = caffe_net.params['inception_5b/pool_proj'][1].data.view(hmarray)
inception_5b_pool_proj = hmarray.zeros(caffe_net.blobs['inception_5b/pool_proj'].data.shape)

inception_5b_output = hmarray.zeros(caffe_net.blobs['inception_5b/output'].data.shape)

pool5_7x7_s1 = hmarray.zeros(caffe_net.blobs['pool5/7x7_s1'].data.shape)

loss3_classifier_filters = caffe_net.params['loss3/classifier'][0].data.view(hmarray)
loss3_classifier_bias = caffe_net.params['loss3/classifier'][1].data.view(hmarray)
loss3_classifier = hmarray.zeros(caffe_net.blobs['loss3/classifier'].data.shape)

prob = hmarray.zeros(caffe_net.blobs['prob'].data.shape)


@compose
def forward(data):

    conv1_7x7_s2 = ConvForward(data, conv1_7x7_s2_filters, conv1_7x7_s2_bias, kernel_size=(7, 7), padding=(3, 3), stride=(2, 2))

    conv1_7x7_s2 = ReluForward(conv1_7x7_s2)

    pool1_3x3_s2, pool1_3x3_s2_mask = PoolForward(conv1_7x7_s2, kernel_size=(3, 3), padding=(0, 0), stride=(2, 2))

    pool1_norm1, pool1_norm1_scale = LrnForward(pool1_3x3_s2, alpha=0.0001, beta=0.75, local_size=5, k=1)

    conv2_3x3_reduce = ConvForward(pool1_norm1, conv2_3x3_reduce_filters, conv2_3x3_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    conv2_3x3_reduce = ReluForward(conv2_3x3_reduce)

    conv2_3x3 = ConvForward(conv2_3x3_reduce, conv2_3x3_filters, conv2_3x3_bias, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    conv2_3x3 = ReluForward(conv2_3x3)

    conv2_norm2, conv2_norm2_scale = LrnForward(conv2_3x3, alpha=0.0001, beta=0.75, local_size=5, k=1)

    pool2_3x3_s2, pool2_3x3_s2_mask = PoolForward(conv2_norm2, kernel_size=(3, 3), padding=(0, 0), stride=(2, 2))

    inception_3a_1x1 = ConvForward(pool2_3x3_s2, inception_3a_1x1_filters, inception_3a_1x1_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_3a_1x1 = ReluForward(inception_3a_1x1)

    inception_3a_3x3_reduce = ConvForward(pool2_3x3_s2, inception_3a_3x3_reduce_filters, inception_3a_3x3_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_3a_3x3_reduce = ReluForward(inception_3a_3x3_reduce)

    inception_3a_3x3 = ConvForward(inception_3a_3x3_reduce, inception_3a_3x3_filters, inception_3a_3x3_bias, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_3a_3x3 = ReluForward(inception_3a_3x3)

    inception_3a_5x5_reduce = ConvForward(pool2_3x3_s2, inception_3a_5x5_reduce_filters, inception_3a_5x5_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_3a_5x5_reduce = ReluForward(inception_3a_5x5_reduce)

    inception_3a_5x5 = ConvForward(inception_3a_5x5_reduce, inception_3a_5x5_filters, inception_3a_5x5_bias, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1))

    inception_3a_5x5 = ReluForward(inception_3a_5x5)

    inception_3a_pool, inception_3a_pool_mask = PoolForward(pool2_3x3_s2, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_3a_pool_proj = ConvForward(inception_3a_pool, inception_3a_pool_proj_filters, inception_3a_pool_proj_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_3a_pool_proj = ReluForward(inception_3a_pool_proj)

    inception_3a_output = ConcatForward(inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj)

    inception_3b_1x1 = ConvForward(inception_3a_output, inception_3b_1x1_filters, inception_3b_1x1_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_3b_1x1 = ReluForward(inception_3b_1x1)

    inception_3b_3x3_reduce = ConvForward(inception_3a_output, inception_3b_3x3_reduce_filters, inception_3b_3x3_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_3b_3x3_reduce = ReluForward(inception_3b_3x3_reduce)

    inception_3b_3x3 = ConvForward(inception_3b_3x3_reduce, inception_3b_3x3_filters, inception_3b_3x3_bias, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_3b_3x3 = ReluForward(inception_3b_3x3)

    inception_3b_5x5_reduce = ConvForward(inception_3a_output, inception_3b_5x5_reduce_filters, inception_3b_5x5_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_3b_5x5_reduce = ReluForward(inception_3b_5x5_reduce)

    inception_3b_5x5 = ConvForward(inception_3b_5x5_reduce, inception_3b_5x5_filters, inception_3b_5x5_bias, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1))

    inception_3b_5x5 = ReluForward(inception_3b_5x5)

    inception_3b_pool, inception_3b_pool_mask = PoolForward(inception_3a_output, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_3b_pool_proj = ConvForward(inception_3b_pool, inception_3b_pool_proj_filters, inception_3b_pool_proj_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_3b_pool_proj = ReluForward(inception_3b_pool_proj)

    inception_3b_output = ConcatForward(inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj)

    pool3_3x3_s2, pool3_3x3_s2_mask = PoolForward(inception_3b_output, kernel_size=(3, 3), padding=(0, 0), stride=(2, 2))

    inception_4a_1x1 = ConvForward(pool3_3x3_s2, inception_4a_1x1_filters, inception_4a_1x1_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4a_1x1 = ReluForward(inception_4a_1x1)

    inception_4a_3x3_reduce = ConvForward(pool3_3x3_s2, inception_4a_3x3_reduce_filters, inception_4a_3x3_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4a_3x3_reduce = ReluForward(inception_4a_3x3_reduce)

    inception_4a_3x3 = ConvForward(inception_4a_3x3_reduce, inception_4a_3x3_filters, inception_4a_3x3_bias, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_4a_3x3 = ReluForward(inception_4a_3x3)

    inception_4a_5x5_reduce = ConvForward(pool3_3x3_s2, inception_4a_5x5_reduce_filters, inception_4a_5x5_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4a_5x5_reduce = ReluForward(inception_4a_5x5_reduce)

    inception_4a_5x5 = ConvForward(inception_4a_5x5_reduce, inception_4a_5x5_filters, inception_4a_5x5_bias, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1))

    inception_4a_5x5 = ReluForward(inception_4a_5x5)

    inception_4a_pool, inception_4a_pool_mask = PoolForward(pool3_3x3_s2, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_4a_pool_proj = ConvForward(inception_4a_pool, inception_4a_pool_proj_filters, inception_4a_pool_proj_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4a_pool_proj = ReluForward(inception_4a_pool_proj)

    inception_4a_output = ConcatForward(inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj)

    inception_4b_1x1 = ConvForward(inception_4a_output, inception_4b_1x1_filters, inception_4b_1x1_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4b_1x1 = ReluForward(inception_4b_1x1)

    inception_4b_3x3_reduce = ConvForward(inception_4a_output, inception_4b_3x3_reduce_filters, inception_4b_3x3_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4b_3x3_reduce = ReluForward(inception_4b_3x3_reduce)

    inception_4b_3x3 = ConvForward(inception_4b_3x3_reduce, inception_4b_3x3_filters, inception_4b_3x3_bias, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_4b_3x3 = ReluForward(inception_4b_3x3)

    inception_4b_5x5_reduce = ConvForward(inception_4a_output, inception_4b_5x5_reduce_filters, inception_4b_5x5_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4b_5x5_reduce = ReluForward(inception_4b_5x5_reduce)

    inception_4b_5x5 = ConvForward(inception_4b_5x5_reduce, inception_4b_5x5_filters, inception_4b_5x5_bias, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1))

    inception_4b_5x5 = ReluForward(inception_4b_5x5)

    inception_4b_pool, inception_4b_pool_mask = PoolForward(inception_4a_output, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_4b_pool_proj = ConvForward(inception_4b_pool, inception_4b_pool_proj_filters, inception_4b_pool_proj_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4b_pool_proj = ReluForward(inception_4b_pool_proj)

    inception_4b_output = ConcatForward(inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj)

    inception_4c_1x1 = ConvForward(inception_4b_output, inception_4c_1x1_filters, inception_4c_1x1_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4c_1x1 = ReluForward(inception_4c_1x1)

    inception_4c_3x3_reduce = ConvForward(inception_4b_output, inception_4c_3x3_reduce_filters, inception_4c_3x3_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4c_3x3_reduce = ReluForward(inception_4c_3x3_reduce)

    inception_4c_3x3 = ConvForward(inception_4c_3x3_reduce, inception_4c_3x3_filters, inception_4c_3x3_bias, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_4c_3x3 = ReluForward(inception_4c_3x3)

    inception_4c_5x5_reduce = ConvForward(inception_4b_output, inception_4c_5x5_reduce_filters, inception_4c_5x5_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4c_5x5_reduce = ReluForward(inception_4c_5x5_reduce)

    inception_4c_5x5 = ConvForward(inception_4c_5x5_reduce, inception_4c_5x5_filters, inception_4c_5x5_bias, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1))

    inception_4c_5x5 = ReluForward(inception_4c_5x5)

    inception_4c_pool, inception_4c_pool_mask = PoolForward(inception_4b_output, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_4c_pool_proj = ConvForward(inception_4c_pool, inception_4c_pool_proj_filters, inception_4c_pool_proj_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4c_pool_proj = ReluForward(inception_4c_pool_proj)

    inception_4c_output = ConcatForward(inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj)

    inception_4d_1x1 = ConvForward(inception_4c_output, inception_4d_1x1_filters, inception_4d_1x1_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4d_1x1 = ReluForward(inception_4d_1x1)

    inception_4d_3x3_reduce = ConvForward(inception_4c_output, inception_4d_3x3_reduce_filters, inception_4d_3x3_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4d_3x3_reduce = ReluForward(inception_4d_3x3_reduce)

    inception_4d_3x3 = ConvForward(inception_4d_3x3_reduce, inception_4d_3x3_filters, inception_4d_3x3_bias, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_4d_3x3 = ReluForward(inception_4d_3x3)

    inception_4d_5x5_reduce = ConvForward(inception_4c_output, inception_4d_5x5_reduce_filters, inception_4d_5x5_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4d_5x5_reduce = ReluForward(inception_4d_5x5_reduce)

    inception_4d_5x5 = ConvForward(inception_4d_5x5_reduce, inception_4d_5x5_filters, inception_4d_5x5_bias, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1))

    inception_4d_5x5 = ReluForward(inception_4d_5x5)

    inception_4d_pool, inception_4d_pool_mask = PoolForward(inception_4c_output, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_4d_pool_proj = ConvForward(inception_4d_pool, inception_4d_pool_proj_filters, inception_4d_pool_proj_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4d_pool_proj = ReluForward(inception_4d_pool_proj)

    inception_4d_output = ConcatForward(inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj)

    inception_4e_1x1 = ConvForward(inception_4d_output, inception_4e_1x1_filters, inception_4e_1x1_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4e_1x1 = ReluForward(inception_4e_1x1)

    inception_4e_3x3_reduce = ConvForward(inception_4d_output, inception_4e_3x3_reduce_filters, inception_4e_3x3_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4e_3x3_reduce = ReluForward(inception_4e_3x3_reduce)

    inception_4e_3x3 = ConvForward(inception_4e_3x3_reduce, inception_4e_3x3_filters, inception_4e_3x3_bias, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_4e_3x3 = ReluForward(inception_4e_3x3)

    inception_4e_5x5_reduce = ConvForward(inception_4d_output, inception_4e_5x5_reduce_filters, inception_4e_5x5_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4e_5x5_reduce = ReluForward(inception_4e_5x5_reduce)

    inception_4e_5x5 = ConvForward(inception_4e_5x5_reduce, inception_4e_5x5_filters, inception_4e_5x5_bias, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1))

    inception_4e_5x5 = ReluForward(inception_4e_5x5)

    inception_4e_pool, inception_4e_pool_mask = PoolForward(inception_4d_output, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_4e_pool_proj = ConvForward(inception_4e_pool, inception_4e_pool_proj_filters, inception_4e_pool_proj_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_4e_pool_proj = ReluForward(inception_4e_pool_proj)

    inception_4e_output = ConcatForward(inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj)

    pool4_3x3_s2, pool4_3x3_s2_mask = PoolForward(inception_4e_output, kernel_size=(3, 3), padding=(0, 0), stride=(2, 2))

    inception_5a_1x1 = ConvForward(pool4_3x3_s2, inception_5a_1x1_filters, inception_5a_1x1_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_5a_1x1 = ReluForward(inception_5a_1x1)

    inception_5a_3x3_reduce = ConvForward(pool4_3x3_s2, inception_5a_3x3_reduce_filters, inception_5a_3x3_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_5a_3x3_reduce = ReluForward(inception_5a_3x3_reduce)

    inception_5a_3x3 = ConvForward(inception_5a_3x3_reduce, inception_5a_3x3_filters, inception_5a_3x3_bias, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_5a_3x3 = ReluForward(inception_5a_3x3)

    inception_5a_5x5_reduce = ConvForward(pool4_3x3_s2, inception_5a_5x5_reduce_filters, inception_5a_5x5_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_5a_5x5_reduce = ReluForward(inception_5a_5x5_reduce)

    inception_5a_5x5 = ConvForward(inception_5a_5x5_reduce, inception_5a_5x5_filters, inception_5a_5x5_bias, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1))

    inception_5a_5x5 = ReluForward(inception_5a_5x5)

    inception_5a_pool, inception_5a_pool_mask = PoolForward(pool4_3x3_s2, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_5a_pool_proj = ConvForward(inception_5a_pool, inception_5a_pool_proj_filters, inception_5a_pool_proj_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_5a_pool_proj = ReluForward(inception_5a_pool_proj)

    inception_5a_output = ConcatForward(inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj)

    inception_5b_1x1 = ConvForward(inception_5a_output, inception_5b_1x1_filters, inception_5b_1x1_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_5b_1x1 = ReluForward(inception_5b_1x1)

    inception_5b_3x3_reduce = ConvForward(inception_5a_output, inception_5b_3x3_reduce_filters, inception_5b_3x3_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_5b_3x3_reduce = ReluForward(inception_5b_3x3_reduce)

    inception_5b_3x3 = ConvForward(inception_5b_3x3_reduce, inception_5b_3x3_filters, inception_5b_3x3_bias, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_5b_3x3 = ReluForward(inception_5b_3x3)

    inception_5b_5x5_reduce = ConvForward(inception_5a_output, inception_5b_5x5_reduce_filters, inception_5b_5x5_reduce_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_5b_5x5_reduce = ReluForward(inception_5b_5x5_reduce)

    inception_5b_5x5 = ConvForward(inception_5b_5x5_reduce, inception_5b_5x5_filters, inception_5b_5x5_bias, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1))

    inception_5b_5x5 = ReluForward(inception_5b_5x5)

    inception_5b_pool, inception_5b_pool_mask = PoolForward(inception_5a_output, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    inception_5b_pool_proj = ConvForward(inception_5b_pool, inception_5b_pool_proj_filters, inception_5b_pool_proj_bias, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    inception_5b_pool_proj = ReluForward(inception_5b_pool_proj)

    inception_5b_output = ConcatForward(inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj)

    pool5_7x7_s1= AvePoolForward(inception_5b_output, kernel_size=(7, 7), padding=(0, 0), stride=(1, 1))

    loss3_classifier = InnerProductForward(pool5_7x7_s1, loss3_classifier_filters, loss3_classifier_bias)

    prob = SoftmaxForward(loss3_classifier)
    return prob


def get_data():
    data = hmarray.random((32, 3, 224, 224), _range=(0, 255))
    data.sync_ocl()
    return data

num_trials = 10
hm_time = 0
caffe_time = 0

# warmup
for _ in range(2):
    data = get_data()
    forward(data)
    caffe_net.forward_all(data=data)

data = get_data()
cl.clFinish(queue)
for i in range(num_trials):
    start = time.clock()
    forward(data)
    hm_time += time.clock() - start
    start = time.clock()
    caffe_net.forward_all(data=data)
    caffe_time += time.clock() - start


import hmcaffe.proto.caffe_pb2 as pb
from google.protobuf import text_format

net_param = pb.NetParameter()

with open(prototxt, "rb") as f:
    text_format.Merge(f.read(), net_param)
    # for blob_name in sorted(caffe_net.blobs.keys()):
for layer in net_param.layer:
    for blob_name in layer.top:
        blob = globals()[blob_name.replace('/', '_')]
        blob.sync_host()
        print("Checking blob {}".format(blob_name))
        caffe_blob = caffe_net.blobs[blob_name].data
        np.testing.assert_array_almost_equal(blob, caffe_blob, decimal=1)
caffe_prob = caffe_net.blobs['prob'].data
prob.sync_host()
np.testing.assert_array_almost_equal(prob, caffe_prob, decimal=3)
print(np.argmax(prob))
print(np.argmax(caffe_net.blobs['prob'].data))
print("Hindemith AVG        : {}".format(hm_time / num_trials))
print("Caffe AVG            : {}".format(caffe_time / num_trials))
print("Speedup (CAFFE / HM) : {}".format(caffe_time / hm_time))
print "SUCCESS"
