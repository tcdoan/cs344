#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>

const unsigned int BLOCKS = 1;
const unsigned int BLOCK_DIM = 8;

__global__ void parallel_scan(int *Y, int *X, int unsigned n) {
    extern __shared__ int XY[];

    int i = threadIdx.x;
    XY[2 * i] = X[2 * i];
    XY[2 * i + 1] = X[2 * i + 1];

    // for d from 0 to log(n) - 1
    //      in parallel for i from 0 to n-1 by 2^(d+1)
    //          XY[i + 2^(d+1) -1] = XY[i + 2^(d+1) -1] + XY[i + 2^d -1]
    for (int two_power_d = 1; two_power_d < n; two_power_d <<= 1) {
        __syncthreads();

        int two_power_dplus1 = two_power_d << 1;
        if (i % two_power_dplus1 == 0) {
            int ai = i + two_power_d - 1;
            int bi = i + two_power_dplus1 - 1;
            XY[bi] += XY[ai];
        }
    }

    if (i == 0) XY[n - 1] = 0;  // clear last element

    for (int two_power_d = n >> 1; two_power_d > 0; two_power_d >>= 1) {
        __syncthreads();

        int two_power_dplus1 = two_power_d << 1;
        if (i % two_power_dplus1 == 0) {
            int ai = i + two_power_d - 1;
            int bi = i + two_power_dplus1 - 1;
            int temp = XY[ai];
            XY[ai] = XY[bi];
            XY[bi] += temp;
        }
    }

    Y[2 * i] = XY[2 * i];
    Y[2 * i + 1] = XY[2 * i + 1];
}

int main(int argc, char **argv) {
    int N = BLOCKS * BLOCK_DIM;

    unsigned int bytes = sizeof(int) * N;
    int h_data[8] = {3, 1, 7, 0, 4, 1, 6, 3};

    for (int i = 0; i < N; i++) {
        printf(i == 20 ? "\n" : "\t");
        printf("%d", h_data[i]);
    }
    printf("\n");

    int *d_in;
    int *d_out;

    cudaMalloc((void **)&d_in, bytes);
    cudaMalloc((void **)&d_out, bytes);

    cudaMemcpy(d_in, h_data, bytes, cudaMemcpyHostToDevice);

    parallel_scan<<<BLOCKS, BLOCK_DIM, N * sizeof(int)>>>(d_out, d_in, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_out, sizeof(int) * N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        printf(i == 20 ? "\n" : "\t");
        printf("%d", h_data[i]);
    }
    printf("\n");

    cudaFree(d_out);
    cudaFree(d_in);

    return 0;
}