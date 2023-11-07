#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include "CycleTimer.h"

#define NUM_THREADS 1000000
#define BLOCK_WIDTH 1000
#define ARRAY_SIZE 10

__global__ void incr_naive(int *g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    i = i %  ARRAY_SIZE;
    g[i] = g[i] + 1;
}


__global__ void incr_atomic(int *g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    i = i %  ARRAY_SIZE;
    atomicAdd(& g[i], 1);
}

float GBPerSec(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

int main(int argc, char **argv)
{
    printf("%d total threads in %d blocks writting into %d array elements \n", NUM_THREADS, NUM_THREADS/BLOCK_WIDTH, ARRAY_SIZE);

    int A_BYTES = sizeof(int) *  ARRAY_SIZE;

    int* h_array= new int[ARRAY_SIZE];
    int *d_array;
    
    cudaMalloc((void **) &d_array, sizeof(int)*ARRAY_SIZE);
    cudaMemset((void*) d_array, 0, A_BYTES);

    double startTime = CycleTimer::currentSeconds();
    incr_atomic<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
    double endTime = CycleTimer::currentSeconds();    
    printf("Time elapsed %.3f ms \n", 1000.f * (endTime - startTime));

    // copy back result from GPU to CPU
    cudaMemcpy(h_array, d_array, A_BYTES, ::cudaMemcpyDeviceToHost);

    // print out the resulting array
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    cudaFree(d_array);
    return 0;
}