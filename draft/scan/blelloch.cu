
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cassert>
#include <chrono>

const unsigned int BLOCKS = 1;
const unsigned int BLOCK_DIM = 1024;

// blelloch kernels
__global__ void exclusive_scan_kernel(float *Y, float *X, int unsigned n) {
    // allocated on invocation
    extern __shared__ float XY[];

    unsigned int t = threadIdx.x;

    XY[2 * t] = X[2 * t];
    XY[2 * t + 1] = X[2 * t + 1];

    int stride = 1;

    // build sums up the tree in-place
    for (unsigned int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();

        if (t < d) {
            unsigned int ai = stride * (2 * t + 1) - 1;
            unsigned int bi = stride * (2 * t + 2) - 1;
            XY[bi] += XY[ai];
        }
        stride <<= 1;
    }

    // zero out the last element
    if (t == 0) XY[n - 1] = 0;

    // scanning by traversing down the tree
    for (unsigned int d = 1; d < n; d <<= 1) {
        __syncthreads();
        stride >>= 1;

        if (t < d) {
            unsigned int ai = stride * (2 * t + 1) - 1;
            unsigned int bi = stride * (2 * t + 2) - 1;
            float temp = XY[ai];
            XY[ai] = XY[bi];
            XY[bi] += temp;
        }
    }

    __syncthreads();
    Y[2 * t] = XY[2 * t];
    Y[2 * t + 1] = XY[2 * t + 1];
}

int main(int argc, char **argv) {
    int N = 2 * BLOCKS * BLOCK_DIM;

    // int N = 65;
    unsigned int bytes = sizeof(float) * N;

    float *h_data = (float *)malloc(bytes);
    for (unsigned int i = 0; i < N; ++i) h_data[i] = 1.0f;

    float *d_in;
    float *d_out;

    cudaMalloc((void **)&d_in, bytes);
    cudaMalloc((void **)&d_out, bytes);

    cudaMemcpy(d_in, h_data, bytes, cudaMemcpyHostToDevice);

    exclusive_scan_kernel<<<BLOCKS, BLOCK_DIM, 2 * BLOCK_DIM * sizeof(float)>>>(d_out, d_in, 2 * BLOCK_DIM);

    cudaMemcpy(h_data, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        printf(i == 20 ? "\n" : "");
        printf("%04.0f ", h_data[i]);
    }
    printf("\n");

    free(h_data);
    cudaFree(d_out);
    cudaFree(d_in);

    return 0;
}
