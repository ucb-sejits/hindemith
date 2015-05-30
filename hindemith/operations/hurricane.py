from hindemith.operations.core import DeviceLevel
from hindemith.cl import hm_compile_and_load
from string import Template
import numpy as np


class GradientAndDenom(DeviceLevel):
    @classmethod
    def get_launcher(cls, sources, sinks, keywords, symbol_table):
        im0 = symbol_table[sources[0].name]
        size = np.prod(im0.shape)
        func = Template("""
#include <math.h>
#define max(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a > _b ? _a : _b; })
#define min(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a < _b ? _a : _b; })
void fn(float* im0, float* Ix, float* Iy, float* It, float* im1,
        float* denom) {
    for (int index = 0; index < $size; index++) {
        float _hm_generated_2;
        float _hm_generated_0;
        float _hm_generated_1;

        It[index] = im1[index] - im0[index];

        {
            int x = index % $width;
            int y = index / $width;
            float accum = 0.0;
            accum += -0.0833333333333f * im1[max(y + -2, 0) * $width + x];
            accum += -0.666666666667f * im1[max(y + -1, 0) * $width + x];
            accum += 0.666666666667f * im1[min(y + 1, $height - 1) * $width + x];
            accum += 0.0833333333333f * im1[min(y + 2, $height - 1) * $width + x];
            Iy[index] = accum;
        }

        {
            int x = index % $width;
            int y = index / $width;
            float accum = 0.0;
            accum += -0.0833333333333f * im1[y * $width + max(x + -2, 0)];
            accum += -0.666666666667f * im1[y * $width + max(x + -1, 0)];
            accum += 0.666666666667f * im1[y * $width + min(x + 1, $width - 1)];
            accum += 0.0833333333333f * im1[y * $width + min(x + 2, $width - 1)];
            Ix[index] = accum;
        }
        _hm_generated_1 = pow(Iy[index], 2);
        _hm_generated_2 = pow(Ix[index], 2);
        _hm_generated_0 = _hm_generated_2 + _hm_generated_1;
        denom[index] = _hm_generated_0 + $alpha;
    }
}""").substitute(size=size, alpha=keywords['alpha']**2, width=im0.shape[1],
                 height=im0.shape[0])

        lib = hm_compile_and_load(func)
        fn = lib.fn
        class GradientAndDenomLauncher(object):
            def __init__(self, sources, sinks):
                self.sources = sources
                self.sinks = sinks

            def compile(self):
                pass

            def launch(self, symbol_table, wait_for):
                im0 = symbol_table[sources[0].name]
                im1 = symbol_table[sources[1].name]
                It = symbol_table[sinks[0].name]
                Iy = symbol_table[sinks[1].name]
                Ix = symbol_table[sinks[2].name]
                denom = symbol_table[sinks[3].name]
                fn.argtypes = tuple(
                    np.ctypeslib.ndpointer(p.dtype, p.ndim, p.shape) for p in
                    [im0, im1, It, Iy, Ix, denom])

                fn(im0, Ix, Iy, It, im1, denom)
        return GradientAndDenomLauncher(sources, sinks)


gradient_and_denom = GradientAndDenom


class UpdateUV(DeviceLevel):
    @classmethod
    def get_launcher(cls, sources, sinks, keywords, symbol_table):
        u = symbol_table[sources[0].name]
        func = Template("""
#include <math.h>
#define max(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a > _b ? _a : _b; })
#define min(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a < _b ? _a : _b; })
void fn(float* Ix, float* v, float* It, float* Iy,
        float* denom, float* u, float* u_new, float* v_new) {
    for (int index = 0; index < $size; index++) {
        float _hm_generated_6;
        float _hm_generated_7;
        float _hm_generated_5;
        float _hm_generated_4;
        float _hm_generated_8;
        float _hm_generated_3;
        float ubar, vbar, t;


        {
            int x = index % $width;
            int y = index / $width;
            float accum = 0.0;

            accum += 0.0833333333333f * u[max(y + -1, 0) * $width + max(x + -1, 0)];
            accum += 0.166666666667f * u[max(y + -1, 0) * $width + x];
            accum += 0.0833333333333f * u[max(y + -1, 0) * $width + min(x + 1, $width - 1)];
            accum += 0.166666666667f * u[y * $width + max(x + -1, 0)];
            accum += 0.166666666667f * u[y * $width + min(x + 1, $width - 1)];
            accum += 0.0833333333333f * u[min(y + 1, $height - 1) * $width + max(x + -1, 0)];
            accum += 0.166666666667f * u[min(y + 1, $height - 1) * $width + x];
            accum += 0.0833333333333f * u[min(y + 1, $height - 1) * $width + min(x + 1, $width - 1)];

            ubar = accum;
        }

        {
            int x = index % $width;
            int y = index / $width;
            float accum = 0.0;

            accum += 0.0833333333333f * v[max(y + -1, 0) * $width + max(x + -1, 0)];
            accum += 0.166666666667f * v[max(y + -1, 0) * $width + x];
            accum += 0.0833333333333f * v[max(y + -1, 0) * $width + min(x + 1, $width - 1)];
            accum += 0.166666666667f * v[y * $width + max(x + -1, 0)];
            accum += 0.166666666667f * v[y * $width + min(x + 1, $width - 1)];
            accum += 0.0833333333333f * v[min(y + 1, $height - 1) * $width + max(x + -1, 0)];
            accum += 0.166666666667f * v[min(y + 1, $height - 1) * $width + x];
            accum += 0.0833333333333f * v[min(y + 1, $height - 1) * $width + min(x + 1, $width - 1)];

            vbar = accum;
        }
        _hm_generated_5 = Iy[index] * vbar;
        _hm_generated_6 = Ix[index] * ubar;
        _hm_generated_4 = _hm_generated_6 + _hm_generated_5;
        _hm_generated_3 = _hm_generated_4 + It[index];
        t = _hm_generated_3 / denom[index];
        _hm_generated_7 = Ix[index] * t;
        u_new[index] = ubar - _hm_generated_7;
        _hm_generated_8 = Iy[index] * t;
        v_new[index] = vbar - _hm_generated_8;
    }
}""").substitute(size=np.prod(u.shape), width=u.shape[1], height=u.shape[0])
        lib = hm_compile_and_load(func)
        fn = lib.fn

        class UpdateUVLauncher(object):
            def __init__(self, sources, sinks):
                self.sources = sources
                self.sinks = sinks

            def compile(self):
                pass

            def launch(self, symbol_table, wait_for):
                u = symbol_table[sources[0].name]
                v = symbol_table[sources[1].name]
                Ix = symbol_table[sources[2].name]
                Iy = symbol_table[sources[3].name]
                It = symbol_table[sources[4].name]
                denom = symbol_table[sources[5].name]
                new_u = symbol_table[sinks[0].name]
                new_v = symbol_table[sinks[1].name]
                fn.argtypes = tuple(
                    np.ctypeslib.ndpointer(p.dtype, p.ndim, p.shape) for p in
                    [u, v, Ix, Iy, It, denom, new_u, new_v])

                fn(Ix, v, It, Iy, denom, u, new_u, new_v)
        return UpdateUVLauncher(sources, sinks)

update_u_v = UpdateUV
