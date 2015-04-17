import caffe_pb2 as pb


def caffemodel_to_net(file_path):
    net = pb.NetParameter()
    with open(file_path, "rb") as f:
        net.ParseFromString(f.read())
    return net


