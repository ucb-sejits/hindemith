import pycl as cl


try:
    devices = cl.clGetDeviceIDs(device_type=cl.CL_DEVICE_TYPE_GPU)
except cl.DeviceNotFoundError:
    devices = cl.clGetDeviceIDs()
context = cl.clCreateContext([devices[-1]])
queue = cl.clCreateCommandQueue(context)
