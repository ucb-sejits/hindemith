from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
import pycl as cl
from ctree.ocl import get_context_and_queue_from_devices

class OclConcreteSpecializedFunction(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        self._c_function = self._compile(entry_name, proj, entry_type)
        devices = cl.clGetDeviceIDs()
        self.context, self.queue = get_context_and_queue_from_devices(
            [devices[-1]])

    def finalize(self, kernel):
        self.kernel = kernel
        return self
