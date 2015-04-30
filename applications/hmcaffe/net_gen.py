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

def gen_lrn_buffers(layer_param):
    name = layer_param.name
    return Template("""
$name = hmarray.zeros(caffe_net.blobs['${name}'].data.shape)
${name}_scale = hmarray.zeros(${name}.shape)
    """).substitute(name=name)
    
def gen_pool_buffers(layer_param):
    name = layer_param.name
    return Template("""
$name = hmarray.zeros(caffe_net.blobs['${name}'].data.shape)
${name}_mask = hmarray.zeros(${name}.shape)
    """).substitute(name=name)

def gen_fc_buffers(layer_param):
    name = layer_param.name
    bottom = layer_param.bottom
    return Template("""
${name}_filters = caffe_net.params['${name}'][0].data.view(hmarray)
${name}_bias = caffe_net.params['${name}'][1].data.view(hmarray)
${name}_bias_multiplier = hmarray((1, ${bottom}.shape[0]))
${name}_bias_multiplier.fill(1)
${name}_bias_multiplier.sync_ocl()
$name = hmarray.zeros(caffe_net.blobs['${name}'].data.shape)
    """).substitute(name=name, bottom=bottom)

def gen_softmax_buffers(layer_param):
    top = layer_param.top[0]
    return Template("""
$top = hmarray.zeros(caffe_net.blobs['${top}'].data.shape)
    """).substitute(top=top)

prototxt = "models/alexnet-ng/deploy.prototxt"

net_param = pb.NetParameter()

with open(prototxt, "rb") as f:
    text_format.Merge(f.read(), net_param)

if len(net_param.layer) > 0:
    layer_params = net_param.layer
else:
    layer_params = net_param.layers

layer_map = {
    pb.V1LayerParameter.CONVOLUTION: gen_conv_buffers,
    pb.V1LayerParameter.POOLING: gen_pool_buffers,
    pb.V1LayerParameter.SOFTMAX: gen_softmax_buffers,
    pb.V1LayerParameter.INNER_PRODUCT: gen_fc_buffers,
    pb.V1LayerParameter.LRN: gen_lrn_buffers,
}
output = ""
for layer_param in layer_params:
    if layer_param.type in layer_map:
        output += layer_map[layer_param.type](layer_param)

print(output)
