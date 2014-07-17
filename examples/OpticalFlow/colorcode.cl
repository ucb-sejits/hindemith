
__kernel void draw_kernel(__global unsigned char * img,
				int width,
				int height)
{
  const int gidx = get_global_id(0);
  const int gidy = get_global_id(1);
  const int lidx = get_local_id(0)*16;
  const int lidy = get_local_id(1)*16;
  if(gidx >= width || gidy >= height) return;
  img[4*gidx + 4*gidy*width + 0] = (unsigned char)(((float)gidx / (float)width) * 255.0f);
  img[4*gidx + 4*gidy*width + 1] = (unsigned char)(((float)gidy / (float)height) * 255.0f);;
  img[4*gidx + 4*gidy*width + 2] = (unsigned char)(0);
  img[4*gidx + 4*gidy*width + 3] = (unsigned char)0;
}

__kernel void compute_color_kernel(__global float * flow,
				    __global char * pix_arr,
				    __global int * colorwheel,
				    int ncols,
				    int width,
				    int height,
				    int pitch,
				    __global float * maxrad_arr)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int npixels = pitch * height;

    if(gidx >= width || gidy >= height ) return;

    float maxrad = *maxrad_arr;

    float fx = flow[gidx + gidy * pitch];
    float fy = flow[gidx + gidy * pitch + npixels];
    if(fabs(fx) > maxrad) maxrad = fabs(fx);
    if(fabs(fy) > maxrad) maxrad = fabs(fy);
    fx = flow[gidx + gidy * pitch] / maxrad;
    fy = flow[gidx + gidy * pitch + npixels] / maxrad;
    __global char * pix = pix_arr + 4*(gidx) + 4*(height - gidy - 1) * width;

    float rad = sqrt(fx * fx + fy * fy);
    float a = atan2(-fy, -fx) / M_PI;
    float fk = (a + 1.0) / 2.0 * (ncols-1);
    int k0 = (int)fk;
    int k1 = (k0 + 1) % ncols;
    float f = fk - k0;
    //f = 0; // uncomment to see original color wheel
    for (int b = 0; b < 3; b++) {
	float col0 = colorwheel[3*k0 + b] / 255.0;
	float col1 = colorwheel[3*k1 + b] / 255.0;
	float col = (1 - f) * col0 + f * col1;
	if (rad <= 1)
	    col = 1 - rad * (1 - col); // increase saturation with radius
	else
	    col *= .75; // out of range
	//pix[2 - b] = (int)(255.0 * col);
	pix[b] = (int)(255.0 * col);
    }
    pix[3] = (unsigned char)0;
}

__kernel 
void max_reduction2_kernel(__global float * buf,
			       __local float * scratch,
			       const int n,
			       __global float * result)
{

  const int local_index = get_local_id(0);
  float res = 0.0f;
  for(int i = local_index ; i < n; i += get_local_size(0))
  {
    if(buf[i] > res)
    {
      res = buf[i];
    }
  }
  scratch[get_local_id(0)] = (local_index< n) ? res : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset = offset / 2) {
    if (local_index < offset) {
      float other = scratch[local_index + offset];
      float mine = scratch[local_index];
      scratch[local_index] = (mine > other) ? mine : other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[get_group_id(0)] = scratch[0];
  }

}

__kernel
void max_reduction_kernel(__global float* buffer1,
            __local float* scratch,
            __const int length,
            __global float* result) {
  
  int global_index = get_global_id(0);
  float accumulator = 0.0f;
  // Loop sequentially over chunks of input vector
  while (global_index < length) {
    float element1 = buffer1[global_index];
    float element2 = buffer1[global_index + length];
    float rad = sqrt(element1*element1 + element2*element2);
    if(rad > accumulator)
    {
      accumulator = rad;
    }
    global_index += get_global_size(0);
  }
  
  // Perform parallel reduction
  int local_index = get_local_id(0);
  scratch[local_index] = accumulator;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset = offset / 2) {
    if (local_index < offset) {
      float other = scratch[local_index + offset];
      float mine = scratch[local_index];
      scratch[local_index] = (mine > other) ? mine : other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[get_group_id(0)] = scratch[0];
  }
}


