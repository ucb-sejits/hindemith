import numpy as np


class Layer(object):
    def update_params(self, rate, weight_decay, momentum):
        pass

    def forward_with_loss(self):
        self.forward()
        loss = 0
        if hasattr(self, 'top') and hasattr(self, 'top_diff'):
            self.top.sync_host()
            self.top_diff.sync_host()
            for i in range(self.top.shape[0]):
                loss += np.sum(self.top[i] * self.top_diff[i])
        return loss
