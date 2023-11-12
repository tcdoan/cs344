
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <algorithm>
#include <chrono>

const unsigned int BLOCKS = 1;
const unsigned int BLOCK_DIM = 256;

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
    int pin = 1 - pout;

    for (unsigned int stride = 1; stride < n; stride *= 2)
    {
        pin = 1 - pin;
        pout = 1 - pin;
        if (t >= stride)
        {
            XY[pout * n + t] = XY[pin * n + t] + XY[pin * n + t - stride];
        }
        else
        {
            XY[pout * n + t] = XY[pin * n + t];
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

    XY[t] = ((t == 0) ? 0.0f : X[i - 1]);
    __syncthreads();

    int pout = 0;
    int pin = 1 - pout;
    for (unsigned int stride = 1; stride < bs; stride *= 2)
    {
        pin = 1 - pin;
        pout = 1 - pin;
        if (t >= stride)
        {
            XY[pout * bs + t] = XY[pin * bs + t] + XY[pin * bs + t - stride];
        }
        else
        {
            XY[pout * bs + t] = XY[pin * bs + t];
        }

        __syncthreads();
    }

    Y[i] = XY[pout * bs + t];
}

int main(int argc, char **argv)
{
    int N = BLOCKS * BLOCK_DIM;

    // int N = 65;
    unsigned int bytes = sizeof(float) * N;

    float *h_data = (float *)malloc(bytes);
    for (unsigned int i = 0; i < min(N, 100); ++i)
        h_data[i] = 1.0f;

    float *d_in;
    float *d_out;

    cudaMalloc((void **)&d_in, bytes);
    cudaMalloc((void **)&d_out, bytes);

    cudaMemcpy(d_in, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_data, bytes, cudaMemcpyHostToDevice);

    double_buffers_inclusive_scan_kernel<<<BLOCKS, N, sizeof(float) * 2 * N>>>(d_out, d_in);

    cudaMemcpy(h_data, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < min(N, 100); i++)
    {
        printf(i == 20 ? "\n" : "\t");
        printf("%.0f ", h_data[i]);
    }
    printf("\n");

    free(h_data);
    cudaFree(d_out);
    cudaFree(d_in);

    return 0;
}
