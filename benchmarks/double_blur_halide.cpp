// This lesson demonstrates how express multi-stage pipelines.
// put this file in the halide/tutorials directory and
//
// On linux, you can compile and run it like so:
// g++ double_blur_halide.cpp -g -I ../include -L ../bin -lHalide `libpng-config --cflags --ldflags` -lpthread -ldl -o double_blur_halide
// LD_LIBRARY_PATH=../bin ./double_blur_halide

// On os x:
// g++ double_blur_halide.cpp -g -I ../include -L ../bin -lHalide `libpng-config --cflags --ldflags` -o double_blur_halide
// DYLD_LIBRARY_PATH=../bin ./double_blur_halide

#include <Halide.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif


using namespace Halide;

// Support code for loading pngs.
#include "image_io.h"

#ifdef __MACH__
double conversion_factor;

void Init() {
	mach_timebase_info_data_t timebase;
	mach_timebase_info(&timebase);
	conversion_factor = (double)timebase.numer / (double)timebase.denom;
}
#endif

int main(int argc, char **argv) {
    struct timespec start, stop;

#ifdef __MACH__
    Init();
#endif

    // First we'll declare some Vars to use below.
    Var x("x"), y("y"), c("c");

    // Now we'll express a multi-stage pipeline that blurs an image
    // first horizontally, and then vertically.

    // The same pipeline, with a boundary condition on the input.
    {
        halide_set_ocl_device_type("gpu");

        // Take a color 8-bit input
        Image<uint8_t> input = load<uint8_t>("images/rgb.png");

        // This time, we'll wrap the input in a Func that prevents
        // reading out of bounds:
        Func clamped("clamped");

        // Define an expression that clamps x to lie within the the
        // range [0, input.width()-1].
        Expr clamped_x = clamp(x, 0, input.width()-1);
        // Similarly clamp y.
        Expr clamped_y = clamp(y, 0, input.height()-1);
        // Load from input at the clamped coordinates. This means that
        // no matter how we evaluated the Func 'clamped', we'll never
        // read out of bounds on the input. This is a clamp-to-edge
        // style boundary condition, and is the simplest boundary
        // condition to express in Halide.
        clamped(x, y, c) = input(clamped_x, clamped_y, c);

        // Upgrade it to 16-bit, so we can do math without it
        // overflowing. This time we'll refer to our new Func
        // 'clamped', instead of referring to the input image
        // directly.
        Func input_16("input_16");
        input_16(x, y, c) = cast<uint16_t>(clamped(x, y, c));

        // The rest of the pipeline will be the same...

        // Blur it horizontally:
        Func blur_x("blur_x");
        blur_x(x, y, c) = (input_16(x-1, y, c) + input_16(x, y, c) + input_16(x+1, y, c))/3;

        // Blur it vertically:
        Func blur_y("blur_y");
        blur_y(x, y, c) = (blur_x(x, y-1, c) + blur_x(x, y, c) + blur_x(x, y+1, c))/3;

        // Blur it horizontally again:
        Func blur_x2("blur_x2");
        blur_x2(x, y, c) = (blur_y(x-1, y, c) + blur_y(x, y, c) + blur_y(x+1, y, c))/3;

        // Blur it vertically:
        Func blur_y2("blur_y2");
        blur_y2(x, y, c) = (blur_x2(x, y-1, c) + blur_x2(x, y, c) + blur_x2(x, y+1, c))/3;

        // Convert back to 8-bit.
        Func output("output");
        output(x, y, c) = cast<uint8_t>(blur_y2(x, y, c));

#ifdef __MACH__
    	uint64_t start = mach_absolute_time();
#else
	    clock_gettime( CLOCK_REALTIME, &start);
#endif

        // This time it's safe to evaluate the output over the some
        // domain as the input, because we have a boundary condition.
        Image<uint8_t> result = output.realize(input.width(), input.height(), 3);


#ifdef __MACH__
	    uint64_t stop = mach_absolute_time();
        printf( "%lf\n", (double)(stop - start) * conversion_factor );
#else
	    clock_gettime( CLOCK_REALTIME, &stop);
	    double accum = ( stop.tv_sec - start.tv_sec ) + ( stop.tv_nsec - start.tv_nsec ) / 1000000000L;
        printf( "%lf\n", accum );
#endif

        // Save the result. It should look like a slightly blurry
        // parrot, but this time it will be the same size as the
        // input.
        save(result, "blurry_parrot_2.png");
    }

    printf("Success!\n");
    return 0;
}
