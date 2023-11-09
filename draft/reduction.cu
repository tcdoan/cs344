
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <algorithm>
#include <chrono>

const unsigned int COARSE_FACTOR = 2;
const unsigned int BLOCKS = 200*1024;
const unsigned int  BLOCK_DIM = 1024;
const unsigned int N = COARSE_FACTOR * BLOCKS * 2 * BLOCK_DIM;

__global__ void simple_reduce_kernel(float *d_out, float *d_in)
{
    unsigned int i = threadIdx.x*2;
    for (unsigned int k = 1; k <= blockDim.x; k *= 2) {
        if (threadIdx.x % k == 0) {
            d_in[i] += d_in[i+k];
        } 
        // wait for all adds at one stage are done
        __syncthreads();
    }

    if (i == 0) *d_out = d_in[0];
}

__global__ void convergent_reduce_kernel(float *d_out, float *d_in)
{
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            d_in[i] += d_in[i + stride];
        } 
        // wait for all adds at one stage are done
        __syncthreads();
    }

    if (i == 0) *d_out = d_in[0];
}

__global__ void shmem_convergent_reduce_kernel(float *d_out, float *d_in)
{
    __shared__ float shared_in[BLOCK_DIM];

    unsigned int i = threadIdx.x;
    shared_in[i] = d_in[i] + d_in[i + BLOCK_DIM];

    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();

        if (threadIdx.x < stride) {
            shared_in[i] += shared_in[i + stride];
        } 
    }

    if (i == 0) *d_out = shared_in[0];
}

__global__ void segmented_shmem_sum_reduction(float *d_out, float *d_in)
{
    extern __shared__ float sdata[];
    
    unsigned int segment = 2*blockDim.x*blockIdx.x;        
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    sdata[t] = d_in[i] + d_in[i + BLOCK_DIM];

    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();

        if (t < stride) {
            sdata[t] += sdata[t + stride];
        } 
    }

    if (threadIdx.x == 0) {
        atomicAdd(d_out, sdata[0]);
    }
}

__global__ void coarsened_shmem_sum_reduction(float *d_out, float *d_in, unsigned int coarse_factor)
{
    extern __shared__ float sdata[];
    
    unsigned int segment = coarse_factor * 2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    float sum = d_in[i];
    for (unsigned int tile = 1; tile < coarse_factor * 2; ++tile) {
        sum += d_in[i + tile * BLOCK_DIM];
    }
    sdata[t] = sum;

    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();

        if (t < stride) {
            sdata[t] += sdata[t + stride];
        } 
    }

    if (threadIdx.x == 0) {
        atomicAdd(d_out, sdata[0]);
    }
}

int main(int argc, char **argv)
{
    unsigned int IN_BYTES = sizeof(float) *  N;
    int OUT_BYTES = sizeof(float);

    float* h_in= new float[N];
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0;
    }
    
    float *d_in;
    float *d_out;
    
    cudaMalloc((void **) &d_in, IN_BYTES);
    cudaMalloc((void **) &d_out, OUT_BYTES);

    cudaMemcpy(d_in, h_in, IN_BYTES, ::cudaMemcpyHostToDevice);

    // simple_reduce_kernel<<<1, N/2>>>(d_out, d_in);
    // convergent_reduce_kernel<<<1, N/2>>>(d_out, d_in);
    // shmem_convergent_reduce_kernel<<<1, N/2>>>(d_out, d_in);

    // segmented_shmem_sum_reduction<<<BLOCKS, BLOCK_DIM, sizeof(float) * BLOCK_DIM>>>(d_out, d_in);

    coarsened_shmem_sum_reduction<<<BLOCKS, BLOCK_DIM, sizeof(float) * BLOCK_DIM>>>(d_out, d_in, COARSE_FACTOR);

    float h_out;
    cudaMemcpy(&h_out, d_out, OUT_BYTES, ::cudaMemcpyDeviceToHost);

    printf("sum = %f \n", h_out);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);    
    return 0;
}