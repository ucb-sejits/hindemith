import numpy as np


class hmarray(np.ndarray):
    def get_element(self, name):
        return "{}[get_global_id(0)]".format(name, self.shape[1])
