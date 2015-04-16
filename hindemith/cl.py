from string import Template
import pycl as cl


try:
    devices = cl.clGetDeviceIDs(device_type=cl.CL_DEVICE_TYPE_GPU)
except cl.DeviceNotFoundError:
    devices = cl.clGetDeviceIDs()
context = cl.clCreateContext(devices)
queue = cl.clCreateCommandQueue(context)


class Kernel(object):
    def __init__(self, launch_parameters):
        self.launch_parameters = launch_parameters
        self.body = ""
        self.sources = set()
        self.sinks = set()
        self.kernel = None

    def append_body(self, string):
        self.body += string

    def compile(self):
        if self.kernel is None:
            sources = set(src.id for src in self.sources)
            sinks = set(src.id for src in self.sinks)
            params = sources | sinks
            self.params = list(params)
            params_str = ", ".join(
                "global float* {}".format(p) for p in self.params)
            kernel = Template("""
    __kernel void fn($params) {
    if (get_global_id(0) < $num_work_items) {
        $body
    }
    }
    """).substitute(params=params_str, body=self.body,
                    num_work_items=self.launch_parameters[0])
            kernel = cl.clCreateProgramWithSource(context, kernel).build()['fn']
            kernel.argtypes = tuple(cl.cl_mem for _ in self.params)
            self.kernel = kernel

    def launch(self, symbol_table):
        args = [symbol_table[p].ocl_buf for p in self.params]
        global_size = self.launch_parameters[0]
        if global_size % 32:
            padded = (global_size + 31) & ~0x20
        else:
            padded = global_size
        self.kernel(*args).on(queue, (padded,))
