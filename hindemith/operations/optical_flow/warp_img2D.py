from _ctypes import sizeof
from ctypes import CFUNCTYPE, c_void_p
from numpy import zeros_like
import numpy as np
from ctree.nodes import Project
from hindemith.utils import unique_name, clamp, UnsupportedBackendError
from hindemith.types.common import Array
from pycl import clCreateCommandQueue, cl_mem, clGetDeviceIDs, clCreateContext, \
    buffer_from_ndarray, clEnqueueNDRangeKernel, buffer_to_ndarray, \
    clCreateProgramWithSource, clWaitForEvents
from ctree.c.nodes import SymbolRef, Constant, CFile
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction

__author__ = 'leonardtruong'


class WarpImg2DConcreteOcl(ConcreteSpecializedFunction):
    def __init__(self):
        self.device = clGetDeviceIDs()[-1]
        self.context = clCreateContext([self.device])
        self.queue = clCreateCommandQueue(self.context)

    def finalize(self, kernel, global_size):
        self.kernel = kernel
        self.kernel.argtypes = (cl_mem, cl_mem, cl_mem, cl_mem)
        self.global_size = global_size
        return self

    def __call__(self, input, u, v):
        output = zeros_like(input.data)
        events = []
        in_buf, in_evt = buffer_from_ndarray(self.queue, input.data, blocking=False)
        events.append(in_evt)
        self.kernel.setarg(0, in_buf, sizeof(cl_mem))

        u_buf, u_evt = buffer_from_ndarray(self.queue, u.data, blocking=False)
        events.append(u_evt)
        self.kernel.setarg(1, u_buf, sizeof(cl_mem))

        v_buf, v_evt = buffer_from_ndarray(self.queue, v.data, blocking=False)
        events.append(v_evt)
        self.kernel.setarg(2, v_buf, sizeof(cl_mem))

        out_buf, out_evt = buffer_from_ndarray(self.queue, output, blocking=False)
        events.append(out_evt)
        self.kernel.setarg(3, out_buf, sizeof(cl_mem))
        clWaitForEvents(*events)
        evt = clEnqueueNDRangeKernel(self.queue, self.kernel, self.global_size)
        evt.wait()
        _, evt = buffer_to_ndarray(self.queue, out_buf, output)
        evt.wait()
        return Array(unique_name(), output)


class WarpImg2DLazyOcl(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        return tuple((arg.dtype, arg.shape) for arg in args)

    def transform(self, tree, program_config):
        arg_cfg = program_config[0]

        body = StringTemplate("""
            void __kernel warp_img2D(__global const $type* input, __global const $type* u,
                                 __global const $type* v, __global $type* output) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int my_x = (int) u[x + $len_x * y];
                int my_y = (int) v[x + $len_x * y];
                float xfrac = u[x + $len_x * y] - my_x;
                float yfrac = v[x + $len_x * y] - my_y;
                if (u[x + $len_x * y] < 0.0) {
                    my_x --;
                    xfrac = 1.0 + xfrac;
                }
                if (v[x + $len_x * y] < 0.0) {
                    my_y --;
                    yfrac = 1.0 + yfrac;
                }
                $type tmp = 0.0;
                if ((x + my_x >= 0) && (x + my_x + 1 < $len_x) &&
                    (y + my_y >= 0) && (y + my_y + 1 <  $len_y)) {
                    tmp += input[(x + my_x) + $len_x * (y + my_y)] * (1.0 - xfrac) * (1.0 - yfrac);
                    tmp += input[(x + my_x + 1) + $len_x * (y + my_y)] * (xfrac) * (1.0 - yfrac);
                    tmp += input[(x + my_x) + $len_x * (y + my_y + 1)] * (1.0 - xfrac) * (yfrac);
                    tmp += input[(x + my_x + 1) + $len_x * (y + my_y + 1)] * (xfrac) * (yfrac);
                } else {
                    tmp = input[clamp(x + my_x, 0, $len_x - 1) +
                                $len_x * clamp(y + my_y, 0, $len_y - 1)];
                }
                output[x + $len_x * y] = tmp;
            }
        """,
                              {
                                  'type': SymbolRef('float'),
                                  'len_x': Constant(arg_cfg[0][1][0]),
                                  'len_y': Constant(arg_cfg[0][1][1])
                              }
        )
        fn = WarpImg2DConcreteOcl()
        kernel = OclFile("kernel", [body])
        program = clCreateProgramWithSource(fn.context, kernel.codegen()).build()
        ptr = program['warp_img2D']
        return fn.finalize(ptr, arg_cfg[0][1])


class WarpImg2DConcreteC(ConcreteSpecializedFunction):
    def finalize(self, entry_name, tree, entry_type):
        self._c_function = self._compile(entry_name, tree, entry_type)
        return self

    def __call__(self, input, u, v):
        output = zeros_like(input.data)
        self._c_function(input.data, u.data, v.data, output)
        return Array(unique_name(), output)


class WarpImg2DLazyC(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        return tuple((arg.data.dtype, arg.data.shape, arg.data.ndim) for arg in args)

    def transform(self, tree, program_config):
        arg_cfg = program_config[0]
        param_types = [
            np.ctypeslib.ndpointer(arg[0], arg[2], arg[1])
            for arg in arg_cfg
        ]
        param_types.append(param_types[0])


        body = StringTemplate("""
            #include <math.h>
            int clamp(float x, float minval, float maxval) {
                return fmin(fmax(x, minval), maxval);
            }
            void warp_img_2D($type* input, $type* u, $type* v, $type* output) {
                for (int x = 0; x < $len_x; x++) {
                    for (int y = 0; y < $len_y; y++) {
                        int my_x = (int) u[x + y * $len_x];
                        int my_y = (int) v[x + y * $len_x];
                        float xfrac = u[x + $len_x * y] - my_x;
                        float yfrac = v[x + $len_x * y] - my_y;
                        if (u[x + $len_x * y] < 0.0) {
                            my_x --;
                            xfrac = 1.0 + xfrac;
                        }
                        if (v[x + $len_x * y] < 0.0) {
                            my_y --;
                            yfrac = 1.0 + yfrac;
                        }
                        $type tmp = 0.0;
                        if ((x + my_x >= 0) && (x + my_x + 1 < $len_x) &&
                            (y + my_y >= 0) && (y + my_y + 1 <  $len_y)) {
                            tmp += input[(x + my_x) + $len_x * (y + my_y)] * (1.0 - xfrac) * (1.0 - yfrac);
                            tmp += input[(x + my_x + 1) + $len_x * (y + my_y)] * (xfrac) * (1.0 - yfrac);
                            tmp += input[(x + my_x) + $len_x * (y + my_y + 1)] * (1.0 - xfrac) * (yfrac);
                            tmp += input[(x + my_x + 1) + $len_x * (y + my_y + 1)] * (xfrac) * (yfrac);
                        } else {
                            tmp = input[clamp(x + my_x, 0, $len_x - 1) +
                                        $len_x * clamp(y + my_y, 0, $len_y - 1)];
                        }
                        output[x + $len_x * y] = tmp;
                    }
                }
            }
        """,
                              {
                                  'type': SymbolRef('float'),
                                  'len_x': Constant(arg_cfg[0][1][0]),
                                  'len_y': Constant(arg_cfg[0][1][1])
                              }
        )
        fn = WarpImg2DConcreteC()
        proj = Project([CFile("generated", [body])])
        return fn.finalize("warp_img_2D", proj,  CFUNCTYPE(c_void_p, *param_types))


class WarpImg2D(object):
    def __new__(cls, backend='ocl'):
        if backend == 'python':
            cls.__call__ = cls.pure_python
            return object.__new__(cls)
        elif backend == 'c':
            return WarpImg2DLazyC(None)
        elif backend == 'ocl':
            return WarpImg2DLazyOcl(None)
        raise UnsupportedBackendError(
            "Teller found an unsupported backend: {0}".format(backend)
        )

    def pure_python(self, tex_Ix, u, v):
        raise NotImplementedError()
        # FIXME: This doesn't work due to python's casting being different than C
        u = u.data
        v = v.data
        tex_Ix = tex_Ix.data
        data = zeros_like(tex_Ix)
        len_x = tex_Ix.shape[0]
        len_y = tex_Ix.shape[1]
        for x in range(len_x):
            for y in range(len_y):
                index = (x, y)
                my_x = int(u[index])
                my_y = int(v[index])
                xfrac = u[index] - float(my_x)
                yfrac = v[index] - float(my_y)
                if u[index] < 0.0:
                    my_x -= 1
                    xfrac += 1.0
                if v[index] < 0.0:
                    my_y -= 1
                    yfrac += 1.0
                if x + my_x >= 0 and x + my_x + 1 < len_x and \
                   y + my_y >= 0 and y + my_y + 1 < len_y:
                    tmp = 0.0
                    tmp += tex_Ix[(x + my_x, y + my_y)] * (1.0 - xfrac) * (1.0 - yfrac)
                    tmp += tex_Ix[(x + my_x + 1, y + my_y)] * (xfrac) * (1.0 - yfrac)
                    tmp += tex_Ix[(x + my_x, y + my_y + 1)] * (1.0 - xfrac) * (yfrac)
                    tmp += tex_Ix[(x + my_x + 1, y + my_y + 1)] * (xfrac) * (yfrac)
                    data[index] = tmp
                else:
                    data[index] = tex_Ix[(clamp(x + my_x, 0, len_x - 1),
                                          clamp(y + my_y, 0, len_y - 1))]
        return Array(unique_name(), data)

warp_img2d = WarpImg2D()
