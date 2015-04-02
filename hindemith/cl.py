import pycl as cl


devices = cl.clGetDeviceIDs(device_type=cl.CL_DEVICE_TYPE_GPU)
context = cl.clCreateContext([devices[-1]])
queue = cl.clCreateCommandQueue(context)
