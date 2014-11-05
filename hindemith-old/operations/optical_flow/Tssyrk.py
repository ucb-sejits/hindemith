import numpy as np
# pragma: no cover
class Tssyrk(object):
    def __init__(self, pure_python=False):
        self.pure_python = pure_python
    def __call__(self, Matrix, offset):
        return tssyrk(Matrix,offset)
    def tssyrk(self, Matrix,offset):
        depth = Matrix.shape[0]
        output = np.zeros([depth,depth])
        for i in xrange(depth):
            for j in xrange(i+1):
                accum = 0
                for x in xrange(Matrix.shape[1]):
                    for y in xrange(Matrix.shape[2]):
                        accum += Matrix[i][x][y] * Matrix[j][x][y]
                output[j][i] = accum
        return output

# pragma: no cover
if __name__ == '__main__':
    input = np.ones([6,512,512])
    tssyrk = Tssyrk()
    print tssyrk(input,0)

"""
// tssyrk expansion
//
// tssyrk reduction
// takes input as 3d array, outputs 2d array
// guessing output is square.
__kernel void tssyrk_reduction(__global float* input, __global float* output)
{
    __local float sh[ $nthreads ];
    int lid = get_local_id(0) + get_local_id(1) * get_local_size(0);
    int nthreads = get_local_size(0) * get_local_size(1);
    for(int i = 0 ; i < $output_dim0 ; i++)
    {
        for(int j = 0 ; j <= i ; j++)
        {
            float res = 0.0f;
            for(int x = get_local_id(0) ; x < $input_dim1; x += get_local_size(0))
            {
                for(int y = get_local_id(1) ; y < $input_dim2 ; y += get_local_size(1))
                {
                    res += input[x,y,i+j*$output_dim0];
                }
            }
            // sum all res's in workgroup
            reduce_th(sh, res, lid, nthreads);
            if(lid == 0)
            {
                output[i,j] = sh[0];
            }
        }
    }
}
// sum operation?
void reduce_th(__local float* sh, float val, int lid, int nthreads){

}
"""
