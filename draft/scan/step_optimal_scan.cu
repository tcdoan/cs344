
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <algorithm>
#include <chrono>

const unsigned int BLOCKS = 1;
const unsigned int  BLOCK_DIM = 256;
const unsigned int N = BLOCKS * BLOCK_DIM;

// Kogge-Stone kernels 
__global__ void double_buffers_inclusive_scan_kernel(float *Y, float *X)
{    
    // allocated on invocation 
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
    // allocated on invocation 
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

    double_buffers_inclusive_scan_kernel<<<BLOCKS, N, sizeof(float)* 2 * N>>>(d_out, d_in);

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
