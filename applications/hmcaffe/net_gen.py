from string import Template
import hmcaffe.proto.caffe_pb2 as pb
from google.protobuf import text_format


def gen_conv_buffers(layer_param):
    name = layer_param.name
    return Template("""
${name}_filters = caffe_net.params['${name}'][0].data.view(hmarray)
${name}_bias = caffe_net.params['${name}'][1].data.view(hmarray)
${name} = hmarray.zeros(caffe_net.blobs['{$name}'].data.shape)
    """).substitute(name=name)


def gen_conv_forward(layer_param):
    name = layer_param.name
    top = layer_param.top[0]
    bottom = layer_param.bottom[0]
    conv_param = layer_param.convolution_param
    return Template("""
    $top = ConvForward($bottom, ${name}_filters, ${name}_bias, kernel_size=(${kernel_size}, ${kernel_size}), padding=(${padding}, ${padding}), stride=(${stride}, ${stride}))
    """).substitute(name=name, top=top, bottom=bottom,
                    kernel_size=conv_param.kernel_size,
                    padding=conv_param.pad,
                    stride=conv_param.stride)


def gen_lrn_buffers(layer_param):
    name = layer_param.name
    return Template("""
$name = hmarray.zeros(caffe_net.blobs['${name}'].data.shape)
${name}_scale = hmarray.zeros(${name}.shape)
    """).substitute(name=name)


def gen_lrn_forward(layer_param):
    name = layer_param.name
    top = layer_param.top[0]
    bottom = layer_param.bottom[0]
    lrn_param = layer_param.lrn_param
    return Template("""
    ${top}, ${name}_scale = LrnForward(${bottom}, alpha=${alpha}, beta=${beta}, local_size=${local_size}, k=1)
    """).substitute(name=name, top=top, bottom=bottom,
                    alpha=lrn_param.alpha, beta=lrn_param.beta,
                    local_size=lrn_param.local_size)
    

def gen_pool_buffers(layer_param):
    name = layer_param.name
    return Template("""
$name = hmarray.zeros(caffe_net.blobs['${name}'].data.shape)
${name}_mask = hmarray.zeros(${name}.shape)
    """).substitute(name=name)
    

def gen_pool_forward(layer_param):
    name = layer_param.name
    top = layer_param.top[0]
    bottom = layer_param.bottom[0]
    pool_param = layer_param.pooling_param
    return Template("""
    ${top}, ${name}_mask = PoolForward(${bottom}, kernel_size=(${kernel_size}, ${kernel_size}), padding=(${padding}, ${padding}), stride=(${stride}, ${stride}))
    """).substitute(top=top, bottom=bottom, name=name,
                    kernel_size=pool_param.kernel_size,
                    padding=pool_param.pad,
                    stride=pool_param.stride)


def gen_fc_buffers(layer_param):
    name = layer_param.name
    bottom = layer_param.bottom[0]
    return Template("""
${name}_filters = caffe_net.params['${name}'][0].data.view(hmarray)
${name}_bias = caffe_net.params['${name}'][1].data.view(hmarray)
${name}_bias_multiplier = hmarray((1, ${bottom}.shape[0]))
${name}_bias_multiplier.fill(1)
${name}_bias_multiplier.sync_ocl()
$name = hmarray.zeros(caffe_net.blobs['${name}'].data.shape)
    """).substitute(name=name, bottom=bottom)


def gen_fc_forward(layer_param):
    name = layer_param.name
    bottom = layer_param.bottom[0]
    top = layer_param.top[0]
    return Template("""
    N = ${top}.shape[1]
    K = np.prod(${bottom}.shape[1:])
    M = ${bottom}.shape[0]
    sgemm(False, True, 1.0, ${bottom}, 0, K, ${name}_filters, 0, K, 0.0,
          ${top}, 0, N, M, N, K)
    sgemm(False, False, 1.0, ${name}_bias_multiplier, 0, 1, ${name}_bias, 0, N,
          1.0, ${top}, 0, N, M, N, 1)
    """).substitute(name=name, bottom=bottom, top=top)


def gen_softmax_buffers(layer_param):
    top = layer_param.top[0]
    return Template("""
    $top = hmarray.zeros(caffe_net.blobs['${top}'].data.shape)
    """).substitute(top=top)


def gen_softmax_forward(layer_param):
    top = layer_param.top[0]
    bottom = layer_param.bottom[0]
    return Template("""
    $top = SoftmaxForward(${bottom})
    """).substitute(top=top, bottom=bottom)


def gen_relu_forward(layer_param):
    top = layer_param.top[0]
    bottom = layer_param.bottom[0]
    return Template("""
    $top = ReluForward(${bottom})
    """).substitute(top=top, bottom=bottom)


prototxt = "models/alexnet-ng/deploy.prototxt"

net_param = pb.NetParameter()

with open(prototxt, "rb") as f:
    text_format.Merge(f.read(), net_param)

if len(net_param.layer) > 0:
    layer_params = net_param.layer
else:
    layer_params = net_param.layers

layer_get_buf_map = {
    pb.V1LayerParameter.CONVOLUTION: gen_conv_buffers,
    pb.V1LayerParameter.POOLING: gen_pool_buffers,
    pb.V1LayerParameter.SOFTMAX: gen_softmax_buffers,
    pb.V1LayerParameter.INNER_PRODUCT: gen_fc_buffers,
    pb.V1LayerParameter.LRN: gen_lrn_buffers,
}

layer_forward_map = {
    pb.V1LayerParameter.CONVOLUTION: gen_conv_forward,
    pb.V1LayerParameter.POOLING: gen_pool_forward,
    pb.V1LayerParameter.SOFTMAX: gen_softmax_forward,
    pb.V1LayerParameter.INNER_PRODUCT: gen_fc_forward,
    pb.V1LayerParameter.LRN: gen_lrn_forward,
    pb.V1LayerParameter.RELU: gen_relu_forward,
}

output = Template("""
from hindemith.types import hmarray
from hindemith.operations.conv import ConvForward
from hindemith.operations.relu import ReluForward
from hindemith.operations.pool import PoolForward
from hindemith.operations.lrn import LrnForward
from hindemith.operations.softmax import SoftmaxForward
from hindemith.core import compose
from hindemith.cl import queue
from hindemith.clibs.clblas import sgemm
import pycl as cl
import caffe
import numpy as np

prototxt = "${prototxt}"
caffemodel = "models/alexnet-ng/alexnet-ng.caffemodel"

caffe.set_mode_gpu()
caffe.set_device(2)
caffe_net = caffe.Net(prototxt, caffemodel, caffe.TEST)

""").substitute(prototxt=prototxt)
for layer_param in layer_params:
    if layer_param.type in layer_get_buf_map:
        output += layer_get_buf_map[layer_param.type](layer_param)
output += """
def forward(data):
"""
for layer_param in layer_params:
    if layer_param.type in layer_forward_map:
        output += layer_forward_map[layer_param.type](layer_param)

print(output)
