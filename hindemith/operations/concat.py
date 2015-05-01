from hindemith.operations.core import DeviceLevel
from hindemith.cl import context, queue
import pycl as cl
from string import Template
import numpy as np
import os


backend = os.getenv("HM_BACKEND", "ocl")


if backend in {"ocl", "opencl", "OCL"}:
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
                def compile(self):
                    pass

                def launch(self, symbol_table):
                    top = symbol_table[sinks[0]]
                    bots = [symbol_table[b] for b in bottoms]
                    for n in range(top.shape[0]):
                        top_offset = n * np.prod(top.shape[1:])
                        for i in range(len(bottoms)):
                            bot_offset = n * np.prod(bots[i].shape[1:])
                            count = np.prod(bots[i].shape[1:])
                            kernel(bots[i].ocl_buf, top.ocl_buf,
                                   top_offset, bot_offset).on(queue, (count, ))
                            top_offset += np.prod(bots[i].shape[1:])


            return Launcher()

elif backend in {"omp", "openmp"}:
    class ConcatForward(DeviceLevel):
        @classmethod
        def get_launcher(cls, sources, sinks, keyword, symbol_table):
            bottoms = sources
            class Launcher():
                def compile(self):
                    pass

                def launch(self, symbol_table):
                    top = symbol_table[sinks[0]]
                    bots = [symbol_table[b] for b in bottoms]
                    top_offset = 0
                    for bot in bots:
                        top[:top.shape[0], top_offset:top_offset + bot.shape[1], ...] = bot
                        top_offset += bot.shape[1]


            return Launcher()
