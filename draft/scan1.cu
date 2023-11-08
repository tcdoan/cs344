#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include "CycleTimer.h"

// Hillis and Steele parallel scan algorithm.
// This implementation handles arrays only as large as 
// it can processed by one thread block running on one SM of the GPU
__global__ void inclusive_scan(float * d_out, float * d_in) {    
    // double input-output buffers in shared memory, size is 2*n
    extern __shared__ float sdata[]; 
    int n = blockDim.x;
    int tid = threadIdx.x;
    int id = n * blockIdx.x + threadIdx.x;
        
    // inclusive_scan 
    sdata[tid] = d_in[id];
    __syncthreads(); 

    // inId used to compute index into the input buffer 
    // outId used to compute index into the output buffer 
    int inId = 1; int outId = 1 - inId;
    for (unsigned int offset = 1; offset < n; offset *= 2) {
        inId = 1 - inId;
        outId = 1 - inId;
        if (tid >= offset) 
            sdata[outId*n + tid] = sdata[inId*n + tid] + sdata[inId*n + tid - offset];
        else 
            sdata[outId*n + tid] = sdata[inId*n + tid];
        
        __syncthreads(); 
    }
    d_out[id] =  sdata[outId*n + tid];
}

int main(int argc, char **argv) {    
    int N = 1024; // int N = 9;   
    int N_BYTES = sizeof(float) *  N;

    float* h_in = new float[N];
    for (int i = 0; i < N; i++) {        
        h_in[i] = 1.0; // h_in[i] = float(i + 1);
    }

    float *d_in, *d_out;
    cudaMalloc((void **) &d_in, N_BYTES);
    cudaMalloc((void **) &d_out, N_BYTES);
    cudaMemcpy(d_in, h_in, N_BYTES, ::cudaMemcpyHostToDevice);
    double startTime = CycleTimer::currentSeconds();
    inclusive_scan<<<1, N, 2*N*sizeof(float)>>>(d_out, d_in);
    double endTime = CycleTimer::currentSeconds();    
    printf("Time elapsed %.3f ms \n", 1000.f * (endTime - startTime));

    float* h_out = new float[N];
    cudaMemcpy(h_out, d_out, N_BYTES, ::cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%.2f ", h_out[i]);
    }
    printf("\n");
    
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}