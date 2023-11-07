#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include "CycleTimer.h"

constexpr int N = 1024 * 1024;
constexpr int BLOCK_WIDTH = 1024;

__global__ void sharedmem_reduce_kernel(float * d_out, float * d_in)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<blocks, threads, shmem_size>>>
    extern __shared__ float sdata[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[i];
    __syncthreads();

    for (unsigned int half = blockDim.x / 2; half > 0; half >>= 1) {
        if (tid < half) {
            sdata[tid] += sdata[tid + half];
        }

        // wait for all adds at one stage are done
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

void reduce(float * d_out, float * d_intermediate ,float * d_in, int size) {    
    int blocks = size/BLOCK_WIDTH;
    int block_threads = size/blocks;

    sharedmem_reduce_kernel<<<blocks, block_threads, block_threads * sizeof(float)>>>(d_intermediate, d_in);

    block_threads = blocks;
    blocks = 1;
    sharedmem_reduce_kernel<<<blocks, block_threads, block_threads * sizeof(float)>>>(d_out, d_intermediate);
} 


int main(int argc, char **argv)
{
    printf("%d total threads in %d blocks writting into %d array elements \n", N, N/BLOCK_WIDTH, N);

    int IN_BYTES = sizeof(float) *  N;
    int INTERMEDIATE_BYTES = sizeof(float) *  (N/BLOCK_WIDTH);
    int OUT_BYTES = sizeof(float);

    float* h_in= new float[N];
    float  h_out;
    std::fill_n(h_in, N, 1);

    float *d_in;
    float *d_intermediate;
    float *d_out;
    
    cudaMalloc((void **) &d_in, IN_BYTES);
    cudaMalloc((void **) &d_intermediate, INTERMEDIATE_BYTES);
    cudaMalloc((void **) &d_out, OUT_BYTES);

    // copy N values from CPU to device 
    cudaMemcpy(d_in, h_in, IN_BYTES, ::cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    reduce(d_out, d_intermediate, d_in, N);
    double endTime = CycleTimer::currentSeconds();    
    printf("Time elapsed %.3f ms \n", 1000.f * (endTime - startTime));

    // copy back result from GPU to CPU
    cudaMemcpy(&h_out, d_out, OUT_BYTES, ::cudaMemcpyDeviceToHost);

    // print out the resulting array
    printf("sum = %.3f \n", h_out);

    cudaFree(d_in);
    return 0;
}