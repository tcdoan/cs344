
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <algorithm>
#include <chrono>

const unsigned int BLOCKS = 1;
const unsigned int BLOCK_DIM = 1024;

// power_of_2(7.0) = 4
inline int power_of_2(float f)
{
    int e;
    frexp(f, &e);
    return 1 << (e - 1);
}

inline bool power_of_2(int n)
{
    return (((n - 1) & n) == 0);
}

__global__ void blelloch_exclusive_scan(float *Y, float *X, int n)
{
    // allocated on invocation
    extern __shared__ float XY[];

    unsigned int t = threadIdx.x;
    unsigned int stride = 1;

    // copy (2 * blockDim.x) entries from input X into shared memory XY
    XY[2 * t] = X[2 * t];
    XY[2 * t + 1] = X[2 * t + 1];

    // build patial sums in place up the tree
    for (unsigned int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (t < d)
        {
            int ai = stride * (2 * t + 1) - 1;
            int bi = stride * (2 * t + 2) - 1;
            XY[bi] += XY[ai];
        }
        stride *= 2;
    }

    // zero out the last element
    if (t == 0)
    {
        XY[n - 1] = 0;
    }

    // traverse down tree and build scans
    for (int d = 1; d < n; d *= 2)
    {
        stride >>= 1;
        __syncthreads();

        if (t < d)
        {
            int ai = stride * (2 * t + 1) - 1;
            int bi = stride * (2 * t + 2) - 1;
            float tmp = XY[ai];
            XY[ai] = XY[bi];
            XY[bi] += tmp;
        }
    }

    __syncthreads();
    Y[2 * t] = XY[2 * t];
    Y[2 * t + 1] = XY[2 * t + 1];
}

int main(int argc, char **argv)
{
    int N = BLOCKS * 2 * BLOCK_DIM;
    unsigned int mem_size = sizeof(float) * N;

    float *h_data = new float[N];
    for (int i = 0; i < N; i++)
    {
        printf(i % 20 == 0 ? "\n" : "\t");
        h_data[i] = 1.0f;
        printf("%.0f", h_data[i]);
    }

    float *d_in;
    float *d_out;

    cudaMalloc((void **)&d_in, mem_size);
    cudaMalloc((void **)&d_out, mem_size);

    cudaMemcpy(d_in, h_data, mem_size, ::cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_data, mem_size, ::cudaMemcpyHostToDevice);

    blelloch_exclusive_scan<<<BLOCKS, BLOCK_DIM, 2 * sizeof(float) * BLOCK_DIM>>>(d_out, d_in, 2 * BLOCK_DIM);

    cudaMemcpy(h_data, d_out, mem_size, ::cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        printf(i % 20 == 0 ? "\n" : "\t");
        printf("%.0f ", h_data[i]);
    }
    printf("\n");

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_data);
    return 0;
}
