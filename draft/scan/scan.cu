
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <algorithm>
#include <chrono>

const unsigned int BLOCKS = 2;
const unsigned int  BLOCK_DIM = 8;
const unsigned int  SECTION_SIZE = 2*BLOCK_DIM;
const unsigned int N = BLOCKS * BLOCK_DIM;

__global__ void double_buffers_inclusive_scan_kernel(float *Y, float *X)
{    
    extern __shared__ float XY[];

    int n = blockDim.x;

    unsigned int t = threadIdx.x;
    unsigned int i = blockIdx.x * n + t;

    XY[t] = X[i];
    __syncthreads();

    int pout = 0;
    int pin = 1-pout;

    for (unsigned int stride = 1; stride < n; stride *= 2) {
        pin = 1 - pin;
        pout = 1 - pin;
        if (t >= stride) {
            XY[pout*n + t] = XY[pin*n + t] + XY[pin*n + t - stride];
        }
        else  {  
            XY[pout*n + t] = XY[pin*n + t];
        }
        
        __syncthreads(); 
    }

    Y[i] = XY[pout * n + t];
}

__global__ void double_buffers_exclusive_scan(float *Y, float *X)
{
    extern __shared__ float XY[];

    int bs = blockDim.x;
    unsigned int t = threadIdx.x;
    unsigned int i = blockIdx.x * bs + t;

    XY[t] = ((t == 0) ? 0.0f : X[i-1]);
    __syncthreads();

    int pout = 0;
    int pin = 1-pout;
    for (unsigned int stride = 1; stride < bs; stride *= 2) {
        pin = 1 - pin;
        pout = 1 - pin;
        if (t >= stride) {
            XY[pout*bs + t] = XY[pin*bs + t] + XY[pin*bs + t - stride];
        }
        else  {  
            XY[pout*bs + t] = XY[pin*bs + t];
        }
        
        __syncthreads(); 
    }

    Y[i] = XY[pout * bs + t];
}

__global__ void brent_kung_inclusive_scan(float *Y, float *X, unsigned int n)
{
    extern __shared__ float XY[];

    // Only validated with block size bs = 2^n up to 1024
    int bs = blockDim.x;
    unsigned int t = threadIdx.x;

    // why times 2 in 2*blockIdx.x * bs + t
    unsigned int i = 2*blockIdx.x * bs + t;

    if (i < n) XY[t] = X[i];
    if (i + bs < n) XY[t + bs] = X[i + bs];

    for (unsigned int stride = 1; stride <= bs; stride *= 2) {
        __syncthreads(); 
        unsigned int index = (t+1) * 2 * stride - 1;

        if (index < SECTION_SIZE) {
            XY[index] += XY[index - stride];
        }
    }

    for (unsigned int stride = SECTION_SIZE/4; stride > 0; stride /= 2) {
        __syncthreads(); 
        unsigned int index = (t+1) * 2 * stride - 1;

        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }        
    }

    __syncthreads();
    if (i < n) Y[i] = XY[t];
    if (i + bs < N) Y[i + bs] = XY[t + bs];
}

int main(int argc, char **argv)
{
    unsigned int ARR_BYTES = sizeof(float) *  N;

    float* h_in= new float[N];
    for (int i = 0; i < N; i++) {
        printf(i % 20 == 0 ? "\n" : "\t");
        h_in[i] = 2.0f;
        printf("%.0f", h_in[i]);
    }
    
    float *d_in;
    float *d_out;
    
    cudaMalloc((void **) &d_in, ARR_BYTES);
    cudaMalloc((void **) &d_out, ARR_BYTES);

    cudaMemcpy(d_in, h_in, ARR_BYTES, ::cudaMemcpyHostToDevice);

    // double_buffers_inclusive_scan_kernel<<<1, N, sizeof(float)* 2 * N>>>(d_out, d_in);
    // double_buffers_exclusive_scan<<<1, N, sizeof(float)* 2 * N>>>(d_out, d_in);
    brent_kung_inclusive_scan<<<1, N, sizeof(float)* 2 * N>>>(d_out, d_in, 2*N);

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
