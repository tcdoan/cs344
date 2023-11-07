#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include "CycleTimer.h"

__global__ void hillis_steele_scan(float * d_out, float * d_in) {
    for (unsigned int i =1; i <= blockDim.x/2; i = i*2) {
        if (threadIdx.x >= i) 
            d_out[threadIdx.x] = d_in[threadIdx.x] + d_in[threadIdx.x-i];
        else 
            d_out[threadIdx.x] = d_in[threadIdx.x];
        
        __syncthreads(); // ensure all 

        // TODO: swap(d_in, d_out) pointers instead of deep copy d_out into d_in array?
        d_in[threadIdx.x] = d_out[threadIdx.x];
        __syncthreads();
    }
}

int main(int argc, char **argv) {
    int N = 16;
    int N_BYTES = sizeof(float) *  N;

    float* h_in = new float[N];
    for (int i = 0; i < N; i++) {
        h_in[i] = float(i + 1);
    }

    float *d_in, *d_out;
    cudaMalloc((void **) &d_in, N_BYTES);
    cudaMalloc((void **) &d_out, N_BYTES);

    cudaMemcpy(d_in, h_in, N_BYTES, ::cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    hillis_steele_scan<<<1, N>>>(d_out, d_in);
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