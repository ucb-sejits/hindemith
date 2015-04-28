from string import Template
import pycl as cl


try:
    devices = cl.clGetDeviceIDs(device_type=cl.CL_DEVICE_TYPE_GPU)
except cl.DeviceNotFoundError:
    devices = cl.clGetDeviceIDs()
context = cl.clCreateContext([devices[-1]])
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
            params = []
            for param in self.params:
                if param in sinks:
                    str = "global float* {}".format(param)
                else:
                    str = "global const float* {}".format(param)
                params.append(str)
            params_str = ", ".join(params)
            kernel = Template("""
__kernel void fn($params) {
    if (get_global_id(0) < $num_work_items) {
        $body
    }
}
    """).substitute(params=params_str, body=self.body,
                    num_work_items=self.launch_parameters[0])
            print(kernel)
            kernel = cl.clCreateProgramWithSource(
                context, kernel).build()['fn']
            kernel.argtypes = tuple(cl.cl_mem for _ in self.params)
            self.kernel = kernel

    def launch(self, symbol_table):
        args = [symbol_table[p].ocl_buf for p in self.params]
        for sink in self.sinks:
            if sink.id == 'conv2':
                for p in self.params:
                    symbol_table[p].sync_host()
                    print(symbol_table[p])
        global_size = self.launch_parameters[0]
        if global_size % 16:
            padded = (global_size + 15) & (~15)
        else:
            padded = global_size
        self.kernel(*args).on(queue, (padded,))
