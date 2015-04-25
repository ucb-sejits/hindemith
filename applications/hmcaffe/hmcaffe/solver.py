from net import Net
import hmcaffe.proto.caffe_pb2 as pb
from google.protobuf import text_format
import logging
log = logging.getLogger("hmcaffe")


class Solver(object):

    """Docstring for Solver. """

    def __init__(self, param):
        """TODO: to be defined1. """
        self.param = param
        self.init_train_net(param)

    def init_train_net(self, param):
        net_param = pb.NetParameter()
        with open(param.net, "rb") as f:
            text_format.Merge(f.read(), net_param)

        net_state = pb.NetState()
        net_state.phase = pb.TRAIN
        # net_state.MergeFrom(net_param.state)
        # net_state.MergeFrom(param.train_state)
        net_param.state.CopyFrom(net_state)
        self.train_net = Net(net_param)

    def step(self, iters):
        avg_loss = self.param.average_loss
        losses = []
        smoothed_loss = 0
        for i in range(iters):
            loss = self.train_net.forward_backward()
            if len(losses) < avg_loss:
                losses.append(loss)
                size = len(losses)
                smoothed_loss = (smoothed_loss * (size - 1) + loss) / size
            else:
                idx = (i - 0) % avg_loss
                smoothed_loss += (loss - losses[idx]) / avg_loss
            log.info("Iteration %d, loss %f", i, smoothed_loss)
            self.compute_update_value(i)
            # self.train_net.update()

    def compute_update_value(self, i):
        current_step = i / 100000.0
        base_lr = .01
        gamma = .1
        rate = base_lr * pow(gamma, current_step)
        weight_decay = .0005
        momentum = 0.9
        self.train_net.update_params(rate, weight_decay, momentum)



def main():
    param = pb.SolverParameter()
    with open("./models/alexnet-ng/solver.prototxt", "rb") as f:
        text_format.Merge(f.read(), param)
    solver = Solver(param)
    solver.step(10)


if __name__ == '__main__':
    main()
