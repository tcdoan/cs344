
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cassert>
#include <chrono>

const unsigned int BLOCKS = 4;
const unsigned int BLOCK_DIM = 8;

__global__ void add(float *Y, float *sums, int unsigned n) {
    extern __shared__ float temp[];

    int t = threadIdx.x;
    int id = blockDim.x * blockIdx.x + t;

    temp[2 * t] = Y[2 * id];
    temp[2 * t + 1] = Y[2 * id + 1];

    temp[2 * t] += sums[blockIdx.x];
    temp[2 * t + 1] += sums[blockIdx.x];

    __syncthreads();

    Y[2 * id] = temp[2 * t];
    Y[2 * id + 1] = temp[2 * t + 1];
}

__global__ void exclusive_parallel_scan(float *Y, float *X, float *sums, int unsigned n) {
    extern __shared__ float temp[];

    int t = threadIdx.x;
    int id = blockDim.x * blockIdx.x + t;

    temp[2 * t] = X[2 * id];
    temp[2 * t + 1] = X[2 * id + 1];

    int s = 1;
    for (int d = n / 2; d > 0; d >>= 1) {
        __syncthreads();

        if (t < d) {
            int ai = s * (2 * t + 1) - 1;
            int bi = s * (2 * t + 2) - 1;

            temp[bi] += temp[ai];
        }
        s <<= 1;
    }

    if (t == 0) {
        if (sums != 0) {
            sums[blockIdx.x] = temp[n - 1];
        }
        temp[n - 1] = 0;
    }

    for (int d = 1; d < n; d <<= 1) {
        s >>= 1;
        __syncthreads();

        if (t < d) {
            int ai = s * (2 * t + 1) - 1;
            int bi = s * (2 * t + 2) - 1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    Y[2 * id] = temp[2 * t];
    Y[2 * id + 1] = temp[2 * t + 1];
}

int main(int argc, char **argv) {
    int N = 2 * BLOCKS * BLOCK_DIM;

    unsigned int bytes = sizeof(float) * N;

    float *h_data = (float *)malloc(bytes);
    for (unsigned int i = 0; i < N; ++i) h_data[i] = 1.0f;

    float *d_in;
    float *d_out;

    float *d_block_sums;

    cudaMalloc((void **)&d_in, bytes);
    cudaMalloc((void **)&d_out, bytes);

    cudaMalloc((void **)&d_block_sums, BLOCKS * sizeof(float));

    cudaMemcpy(d_in, h_data, bytes, cudaMemcpyHostToDevice);

    exclusive_parallel_scan<<<BLOCKS, BLOCK_DIM, 2 * BLOCK_DIM * sizeof(float)>>>(d_out, d_in, d_block_sums,
                                                                                  2 * BLOCK_DIM);

    exclusive_parallel_scan<<<1, BLOCKS / 2, BLOCKS * sizeof(float)>>>(d_block_sums, d_block_sums, 0, BLOCKS);

    add<<<BLOCKS, BLOCK_DIM, 2 * BLOCK_DIM * sizeof(float)>>>(d_out, d_block_sums, 2 * BLOCK_DIM);

    cudaMemcpy(h_data, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf(i == 20 ? "\n" : "");
        printf("%04.0f ", h_data[i]);
    }
    printf("\n");

    free(h_data);

    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_block_sums);

    return 0;
}
