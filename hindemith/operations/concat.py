from hindemith.operations.core import DeviceLevel
import pycl as cl
from string import Template
import numpy as np
import os
import ast

backend = os.getenv("HM_BACKEND", "ocl")

if backend in {"ocl", "opencl", "OCL"}:
    from hindemith.cl import context, queues
    class ConcatForward(DeviceLevel):
        @classmethod
        def get_launcher(cls, sources, sinks, keyword, symbol_table):
            bottoms = sources
            concat_kern = Template("""
    __kernel void concat(global const float* bottom,
                         global float* top, int top_offset, int bot_offset) {
        int index = get_global_id(0);
        top[index + top_offset] = bottom[index + bot_offset];
    }
            """).substitute()
            program = cl.clCreateProgramWithSource(context, concat_kern).build()
            kernel = program['concat']
            kernel.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_int, cl.cl_int)

            class Launcher():
                def __init__(self, sources, sinks):
                    self.sources = sources
                    self.sinks = sinks

                def compile(self):
                    pass

                def launch(self, symbol_table, wait_for):
                    top = symbol_table[sinks[0].name]
                    bots = [symbol_table[b.name] for b in bottoms]
                    evts = []
                    concat_off = 0
                    for i in range(len(bottoms)):
                        count = np.prod(bots[i].shape[1:])
                        for n in range(bots[i].shape[0]):
                            top_offset = n * np.prod(top.shape[1:]) + concat_off * np.prod(top.shape[2:])
                            evt = kernel(
                                bots[i].ocl_buf, top.ocl_buf, top_offset, n * count).on(
                                    queues[n % len(queues)], (count, ), wait_for=wait_for)
                            evts.append(evt)
                        concat_off += bots[i].shape[1]
                    return evts


            return Launcher(sources, sinks)

elif backend in {"omp", "openmp"}:
    class ConcatForward(DeviceLevel):
        @classmethod
        def get_launcher(cls, sources, sinks, keyword, symbol_table):
            bottoms = sources
            class Launcher():
                def __init__(self, sources, sinks):
                    self.sources = sources
                    self.sinks = sinks

                def compile(self):
                    pass

                def launch(self, symbol_table, wait_for):
                    top = symbol_table[sinks[0].name]
                    bots = [symbol_table[b.name] for b in bottoms]
                    top_offset = 0
                    for bot in bots:
                        top[:top.shape[0], top_offset:top_offset + bot.shape[1], ...] = bot
                        top_offset += bot.shape[1]


            return Launcher(sources, sinks)
