from hindemith.operations.core import DeviceLevel
from hindemith.types import hmarray
import ast
import numpy as np
import pycl as cl
import os

backend = os.getenv("HM_BACKEND", "ocl")
if backend in {"ocl", "opencl", "OCL"}:
    import pycl as cl
    from hindemith.clibs.clblas import sgemm

class InnerProductForward(DeviceLevel):
    if backend in {"ocl", "opencl", "OCL"}:
        @classmethod
        def get_launcher(cls, sources, sinks, keywords, symbol_table):
            bottom = symbol_table[sources[0].name]
            top = symbol_table[sinks[0].name]
            bias_multiplier = hmarray((1, bottom.shape[0]))
            bias_multiplier.fill(1)
            bias_multiplier.sync_ocl()
            N = top.shape[1]
            K = np.prod(bottom.shape[1:])
            M = bottom.shape[0]
            class InnerProductLauncher(object):
                def __init__(self, sources, sinks):
                    self.sources = sources
                    self.sinks = sinks

                def compile(self):
                    pass

                def launch(self, symbol_table, wait_for):
                    bottom = symbol_table[sources[0].name]
                    weights = symbol_table[sources[1].name]
                    bias = symbol_table[sources[2].name]
                    top = symbol_table[sinks[0].name]
                    evt = sgemm(False, True, 1.0, bottom, 0, K, weights,
                                0, K, 0.0, top, 0, N, M, N, K, wait_for=wait_for)
                    evt = sgemm(False, False, 1.0, bias_multiplier, 0,
                                1, bias, 0, N, 1.0, top, 0, N, M, N,
                                1, wait_for=evt)
                    return [evt]
            return InnerProductLauncher(sources, sinks)
