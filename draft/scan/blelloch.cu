
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cassert>
#include <chrono>

const unsigned int BLOCKS = 1;
const unsigned int BLOCK_DIM = 1024;

inline unsigned int get_levels(unsigned int n, unsigned int b) {
    unsigned int levels = (unsigned int)(log((n - 0.5)) / log((float)b));
    return levels;
}

inline unsigned int low_power_of_2(unsigned int x) {
    int e;
    frexp((float)x, &e);
    return 1 << (e - 1);
}

inline bool power_of_2(unsigned int n) { return (((n - 1) & n) == 0); }

unsigned int num_blocks(unsigned int x, unsigned int block_size) {
    unsigned int data_size = 2 * block_size;
    return (x + data_size - 1) / data_size;
}

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

unsigned int d_num_elements;
unsigned int d_num_levels;
float **d_block_sums;

template <bool STORE_SUM>
__device__ void scan_block(float *shared_data, int block_idx, float *block_sums) {
    int stride = leaves_2_root_scan(shared_data);

    zero_last_element<STORE_SUM>(shared_data, block_sums, (block_idx == 0) ? blockIdx.x : block_idx);

    root_2_leaves_scan(shared_data, stride);
}

template <bool STORE_SUM>
__global__ void scan(float *d_out, const float *d_in, float *d_block_sums_level, int n, int block_idx, int base_idx) {
    int ai, bi, mem_ai, mem_bi, offset_bank_a, offset_bank_b;
    extern __shared__ float shared_data[];

    copy_to_shared_data<POWER_OF_2>(shared_data, d_in, n,
                                    (base_idx == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)) : base_idx, ai, bi, mem_ai,
                                    mem_bi, offset_bank_a, offset_bank_b);

    scan_block<STORE_SUM>(shared_data, block_idx, d_block_sums_level);

    copy_to_global_mem<POWER_OF_2>(d_out, shared_data, n, ai, bi, mem_ai, mem_bi, offset_bank_a, offset_bank_b);
}

void init_block_sums(unsigned int elements, unsigned int block_dim) {
    d_num_elements = elements;
    int levels = get_levels(elements, 2 * block_dim);
    d_num_levels = levels;
    d_block_sums = (float **)malloc(levels * sizeof(float *));

    int level = 0;
    do {
        unsigned int blocks = num_blocks(elements, block_dim);
        if (blocks > 1) {
            cudaMalloc((void **)&d_block_sums[level++], blocks * sizeof(float));
        }
    } while (elements > 1);
}

void scan_loop(float *d_out, const float *d_in, int elements, int level) {
    unsigned int blocks = num_blocks(elements, BLOCK_DIM);
    unsigned int num_threads;

    if (blocks > 1) {
        num_threads = BLOCK_DIM;
    } else {
        assert(power_of_2(elements));
        num_threads = elements / 2;
    }

    unsigned int block_elements = num_threads * 2;
    unsigned int last_block_elements = elements - (blocks - 1) * elements;
    unsigned int last_block_threads = max(1, last_block_elements / 2);
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
    cudaMemcpy(d_out, h_data, bytes, cudaMemcpyHostToDevice);

    init_block_sums(N, BLOCK_DIM);

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
