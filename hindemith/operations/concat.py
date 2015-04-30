from hindemith.operations.core import DeviceLevel
from hindemith.cl import context, queue
import pycl as cl
from string import Template
import numpy as np


class ConcatForward(DeviceLevel):
    @classmethod
    def get_launcher(cls, sources, sinks, keyword, symbol_table):
        bottoms = sources
        count = np.prod(symbol_table[bottoms[0]].shape[1:])
        concat_kern = Template("""
__kernel void concat(global const float* bottom,
                     global float* top, int top_offset, int bot_offset) {
  if (get_global_id(0) < $count) {
    int index = get_global_id(0);
    top[index + top_offset] = bottom[index + bot_offset];
  }
}
        """).substitute(count=count)
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
                        bot_offset = n * np.prod(bots[0].shape[1:])
                        kernel(symbol_table[bottoms[i]].ocl_buf, top.ocl_buf,
                               top_offset, bot_offset).on(queue, (count, ))
                        top_offset += np.prod(bots[0].shape[1:])

                    
        return Launcher()

