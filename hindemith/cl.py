from string import Template
import pycl as cl
import os
import tempfile
import subprocess
import ctypes as ct
import numpy as np
import random
import os
import sys


backend = os.getenv("HM_BACKEND", "ocl")

count = 0
def get_unique_kernel_name():
    global count
    count += 1
    return "fn{}".format(count)


if backend in {"ocl", "opencl", "OCL"}:
    try:
        # platforms = cl.clGetPlatformIDs()
        # devices = cl.clGetDeviceIDs(platforms[1])
        devices = cl.clGetDeviceIDs(device_type=cl.CL_DEVICE_TYPE_GPU)
    except cl.DeviceNotFoundError:
        devices = cl.clGetDeviceIDs()
    context = cl.clCreateContext(devices[-1:])
    if os.environ.get("TRAVIS"):
        queues = [cl.clCreateCommandQueue(context)]
    else:
        queues = [
            cl.clCreateCommandQueue(
                context
            ) for _ in range(8)
        ]
        # queues = [
        #     cl.clCreateCommandQueue(
        #         context,
        #         properties=cl.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
        #     ) for _ in range(10)
        # ]
    queue = queues[0]

hm_dir = os.path.join(tempfile.gettempdir(), "hindemith")

if not os.path.exists(hm_dir):
    os.mkdir(hm_dir)
unique_file_id = -1


def hm_compile_and_load(_file):
    print(_file)
    file_path = os.path.join(hm_dir, "temp_file.c")
    with open(file_path, 'w') as f:
        f.write(_file)
    global unique_file_id
    unique_file_id += 1
    so_name = "compiled{}.so".format(unique_file_id)
    so_path = os.path.join(hm_dir, so_name)
    flags = "-shared -std=gnu99 -fPIC -fopenmp"
    compiler = os.environ.get("CC", "CC")
    compile_cmd = "{} {} -o {} {}".format(compiler, flags, so_path, file_path)
    subprocess.check_call(compile_cmd, shell=True)
    lib = ct.cdll.LoadLibrary(so_path)
    return lib


if backend in {"ocl", "opencl", "OCL"}:
    class Kernel(object):
        def __init__(self, launch_parameters):
            self.launch_parameters = launch_parameters
            self.body = ""
            self.sources = set()
            self.sinks = set()
            self.kernel = None
            self.kernel_str = None

        def append_body(self, string):
            lines = [l.lstrip() for l in string.splitlines()]
            self.body += "\t\t\t" + "\n\t\t\t".join(lines) + "\n"

        def compile(self):
            if self.kernel is None:
                params = set(self.sources) | set(self.sinks)
                seen_decls = set()
                seen_params = set()
                decls = []
                filtered = set()
                for param in params:
                    if param.level == 'register':
                        if param.name not in seen_decls:
                            seen_decls.add(param.name)
                            decls.append('float {}'.format(param.name))
                    else:
                        if param.name not in seen_params:
                            seen_params.add(param.name)
                            filtered.add(param)
                self.params = list(filtered)
                self.params.sort(key=lambda x: x.name)
                params = []
                sinks = set(sink.name for sink in self.sinks)
                for param in self.params:
                    if param.name in sinks:
                        str = "global float* {}".format(param.name)
                    else:
                        str = "global const float* {}".format(param.name)
                    params.append(str)
                params_str = ", ".join(params)
                decls = ";\n\t\t\t".join(decls) + ";\n"
                kernel_name = get_unique_kernel_name()
                kernel = Template("""
    __kernel void $name($params) {
        int index = get_global_id(0);
        if (index < $num_work_items) {
            $decls
$body
        }
    }
        """).substitute(name=kernel_name, params=params_str, body=self.body, decls=decls,
                        num_work_items=self.launch_parameters[0])
                # print([p.name for p in self.params])
                print(kernel)
                self.kernel_str = kernel
                kernel = cl.clCreateProgramWithSource(
                    context, kernel).build()[kernel_name]
                kernel.argtypes = tuple(cl.cl_mem for _ in self.params)
                self.kernel = kernel

        def launch(self, symbol_table, wait_for=None):
            args = []
            for param in self.params:
                val = symbol_table[param.name]
                if hasattr(val, 'ocl_buf'):
                    args.append(val.ocl_buf)
            global_size = self.launch_parameters[0]
            local_size = 32
            if global_size % local_size:
                padded = (global_size + (local_size - 1)) & (~(local_size - 1))
            else:
                padded = global_size
            evt = self.kernel(*args).on(queues[random.randint(0, len(queues) - 1)],
                                        (padded,), wait_for=wait_for)
            return [evt]
elif backend in {"omp", "openmp"}:

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
                params = set(self.sources) | set(self.sinks)
                seen_decls = set()
                seen_params = set()
                decls = []
                filtered = set()
                for param in params:
                    if param.level == 'register':
                        if param.name not in seen_decls:
                            seen_decls.add(param.name)
                            decls.append('float {}'.format(param.name))
                    else:
                        if param.name not in seen_params:
                            seen_params.add(param.name)
                            filtered.add(param)
                self.params = list(filtered)
                params = []
                for param in self.params:
                    _str = "float* {}".format(param.name)
                    params.append(_str)
                params_str = ", ".join(params)
                decls = ";\n\t\t\t".join(decls) + ";\n"
                kernel = Template("""
     #include <math.h>
     #include <float.h>
     #define max(a,b) ((a) > (b) ? a : b)
     #define min(a,b) ((a) < (b) ? a : b)
     void fn($params) {
        #pragma omp parallel for
        for (int index = 0; index < $num_work_items; index++) {
            $decls
            $body
        }
    }
        """).substitute(params=params_str, body=self.body, decls=decls,
                        num_work_items=self.launch_parameters[0])
                lib = hm_compile_and_load(kernel)
                self.func = lib.fn
                self.func.restype = None

        def launch(self, symbol_table, wait_for):
            args = []
            for param in self.params:
                val = symbol_table[param.name]
                args.append(val)
            self.func.argtypes = (
                np.ctypeslib.ndpointer(p.dtype, p.ndim, p.shape) for p in args)
            self.func(*args)
else:
    raise NotImplementedError(
        "Hindemith has not implemented a backend called " + backend)
