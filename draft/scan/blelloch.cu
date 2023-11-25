
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cassert>
#include <chrono>

const unsigned int BLOCKS = 1;
const unsigned int BLOCK_DIM = 4;

// Blelloch algorithm
//
// A work-efficient scan algorithm that builds a balanced binary tree on the input data
// and sweep it to and from the root to compute the prefix sum
//
// A binary tree with n leaves, and log(n) levels.
// Each level d∈[0,n) has 2^d nodes.
//
// The tree is not an actual data structure. It is an abstract model to determine what
// each thread does at each step of the traversal.
//
// In Blelloch algorithm, we perform the operations in place on an array in shared memory.
// The algorithm consists of two phases; The reduce phase (aka the up-sweep phase) and the down-sweep phase.
//
// In the reduce phase we traverse the tree from leaves to root computing partial sums at internal
// nodes of the tree.
//
// This is also known as a parallel reduction, the root node, the last node in the array
// holds the sum of all nodes in the  array.
__global__ void parallel_scan(float *Y, float *X, int unsigned n) {
    extern __shared__ float XY[];

    int i = threadIdx.x;
    XY[2 * i] = X[2 * i];
    XY[2 * i + 1] = X[2 * i + 1];

    // for d from 0 to log(n) -1
    // in parallel for thread i from 0 to n-1 step by 2^(d+1)
    // XY[i + 2^(d+1) -1] = XY[i + 2^(d+1) -1] + XY[i + 2^d -1]
    for (int d = 0, two_power_d = 1; two_power_d < n / 2; d++, two_power_d <<= 1) {
        __syncthreads();
        int two_power_dplus1 = two_power_d << 1;
        int bi = i + two_power_dplus1 - 1;
        int ai = i + two_power_d - 1;

        printf("\n thread %d, d %d, two_power_d %d, two_power_dplus1 %d, ai %d, bi %d \n", i, d, two_power_d,
               two_power_dplus1, ai, bi);

        XY[i + two_power_dplus1 - 1] = XY[i + two_power_d - 1] + XY[i + two_power_dplus1 - 1];
    }

    Y[2 * i] = XY[2 * i];
    Y[2 * i + 1] = XY[2 * i + 1];
}

int main(int argc, char **argv) {
    int N = 2 * BLOCKS * BLOCK_DIM;

    // int N = 65;
    unsigned int bytes = sizeof(float) * N;

    float h_data[8] = {3.0f, 1.0f, 7.0f, 0.0f, 4.0f, 1.0f, 6.0f, 3.0f};
    // float *h_data = (float *)malloc(bytes);
    // for (unsigned int i = 0; i < N; ++i) h_data[i] = 1.0f;

    for (int i = 0; i < N; i++) {
        printf(i == 20 ? "\n" : "");
        printf("%2.0f ", h_data[i]);
    }
    printf("\n");

    float *d_in;
    float *d_out;

    cudaMalloc((void **)&d_in, bytes);
    cudaMalloc((void **)&d_out, bytes);

    cudaMemcpy(d_in, h_data, bytes, cudaMemcpyHostToDevice);

    parallel_scan<<<BLOCKS, BLOCK_DIM, 2 * BLOCK_DIM * sizeof(float)>>>(d_out, d_in, 2 * BLOCK_DIM);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        printf(i == 20 ? "\n" : "");
        printf("%2.0f ", h_data[i]);
    }
    printf("\n");

    // free(h_data);
    cudaFree(d_out);
    cudaFree(d_in);

    return 0;
}
