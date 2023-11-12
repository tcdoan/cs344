
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <algorithm>
#include <chrono>

const unsigned int BLOCKS = 1;
const unsigned int  BLOCK_DIM = 1024;
const unsigned int N = 2*BLOCKS * BLOCK_DIM;

__global__ void blelloch_exclusive_scan(float *Y, float *X, int n)
{
    // allocated on invocation 
    extern __shared__ float XY[];

    unsigned int t = threadIdx.x;
    unsigned int stride = 1;

    // copy (2 * blockDim.x) entries from input X into shared memory XY
    XY[2*t] = X[2*t]; 
    XY[2*t + 1] = X[2*t + 1];

    // build patial sums in place up the tree 
    for (unsigned int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        
        if (t < d) {
            int ai = stride * (2*t + 1) -1;
            int bi = stride * (2*t + 2) -1;
            XY[bi] += XY[ai];
        }
        stride *= 2;
    }

    // zero out the last element 
    if (t == 0) { XY[n - 1] = 0; } 

     // traverse down tree and build scans
    for (int d = 1; d < n; d *= 2) {
        stride >>= 1;
        __syncthreads();

        if (t < d) {
            int ai = stride*(2*t+1)-1;
            int bi = stride*(2*t+2)-1;
            float tmp = XY[ai];
            XY[ai] = XY[bi];
            XY[bi] += tmp;
        }
    } 

    __syncthreads();
    Y[2*t] = XY[2*t];
    Y[2*t + 1] = XY[2*t + 1];
}


int main(int argc, char **argv)
{
    unsigned int ARR_BYTES = sizeof(float) *  N;

    float* h_in= new float[N];
    for (int i = 0; i < N; i++) {
        printf(i % 20 == 0 ? "\n" : "\t");
        h_in[i] = 1.0f;
        printf("%.0f", h_in[i]);
    }
    
    float *d_in;
    float *d_out;
    
    cudaMalloc((void **) &d_in, ARR_BYTES);
    cudaMalloc((void **) &d_out, ARR_BYTES);

    cudaMemcpy(d_in, h_in, ARR_BYTES, ::cudaMemcpyHostToDevice);

    blelloch_exclusive_scan<<<BLOCKS, BLOCK_DIM, 2 * sizeof(float) * BLOCK_DIM >>>(d_out, d_in, 2*BLOCK_DIM);

    float* h_out= new float[N];
    cudaMemcpy(h_out, d_out, ARR_BYTES, ::cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {        
        printf(i % 20 == 0 ? "\n" : "\t");
        printf("%.0f ", h_out[i]);
    }
    printf("\n");

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
