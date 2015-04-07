import pycl as cl


try:
    devices = cl.clGetDeviceIDs(device_type=cl.CL_DEVICE_TYPE_GPU)
except cl.DeviceNotFoundError:
    devices = cl.clGetDeviceIDs()
context = cl.clCreateContext([devices[-1]])
queue = cl.clCreateCommandQueue(context)


class Kernel(object):
    def __init__(self, kernel, inputs, outputs, global_size):
        self.kernel = kernel
        self.inputs = inputs
        self.outputs = outputs
        self.global_size = global_size

    def launch(self, env):
        bufs = []
        for arr in self.inputs:
            env[arr].sync()
            bufs.append(env[arr].ocl_buf)
        for arr in self.outputs:
            env[arr].sync()
            bufs.append(env[arr].ocl_buf)
            env[arr].host_dirty = True
        self.kernel(*bufs).on(queue, self.global_size)


class ElementLevelKernel(Kernel):
    def __init__(self, global_size):
        self.body = ""
        self.global_size = global_size
        self.sources = set()
        self.sinks = set()
        self.compiled = None

    def launch(self, env):
        if not self.compiled:
            self.sources = list(self.sources - self.sinks)
            self.sinks = list(self.sinks)
            sources = ", ".join(
                "__global const float* {}".format(name)
                for name in self.sources
            )
            sinks = ", ".join(
                "__global float* {}".format(name)
                for name in self.sinks
            )
            kernel = """
__kernel void func({params}) {{
  if (get_global_id(0) < {size}) {{
    {body}
  }}
}}""".format(params=sources + ", " + sinks, body=self.body,
             size=self.global_size[0])
            print(kernel)
            self.compiled = cl.clCreateProgramWithSource(
                context, kernel
            ).build()['func']
            self.compiled.argtypes = tuple(
                cl.cl_mem for _ in self.sources + self.sinks)
        bufs = []
        for arr in self.sources:
            env[arr].sync()
            bufs.append(env[arr].ocl_buf)
        for arr in self.sinks:
            env[arr].sync()
            bufs.append(env[arr].ocl_buf)
            env[arr].host_dirty = True
        global_size = self.global_size
        padded = ()
        for s in global_size:
            if s % 32:
                padded += ((s + 31) & ~0x20,)
            else:
                padded += (s, )
        self.compiled(*bufs).on(queue, global_size)
