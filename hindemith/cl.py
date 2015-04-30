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
        self.body += string + "\n"

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
    int index = get_global_id(0);
    if (index < $num_work_items) {
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
        args = []
        for param in self.params:
            val = symbol_table[param]
            if hasattr(val, 'ocl_buf'):
                args.append(val.ocl_buf)
        global_size = self.launch_parameters[0]
        local_size = 16
        if global_size % local_size:
            padded = (global_size + (local_size - 1)) & (~(local_size - 1))
        else:
            padded = global_size
        self.kernel(*args).on(queue, (padded,))
