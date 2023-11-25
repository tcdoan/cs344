#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>

const unsigned int BLOCKS = 1;
const unsigned int BLOCK_DIM = 4;

__global__ void parallel_scan(int *Y, int *X, int unsigned n) {
    extern __shared__ int XY[];

    int i = threadIdx.x;
    XY[2 * i] = X[2 * i];
    XY[2 * i + 1] = X[2 * i + 1];

    // for d from 0 to log(n) - 1
    //      in parallel for i from 0 to n-1 by 2^(d+1)
    //          XY[i + 2^(d+1) -1] = XY[i + 2^(d+1) -1] + XY[i + 2^d -1]
    for (int d = 0, two_power_d = 1; two_power_d < n; d++, two_power_d <<= 1) {
        __syncthreads();

        int two_power_dplus1 = two_power_d << 1;
        int ai = i + two_power_d - 1;
        int bi = i + two_power_dplus1 - 1;
        XY[bi] = XY[ai] + XY[bi];
    }

    Y[2 * i] = XY[2 * i];
    Y[2 * i + 1] = XY[2 * i + 1];
}

int main(int argc, char **argv) {
    int N = 2 * BLOCKS * BLOCK_DIM;

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
