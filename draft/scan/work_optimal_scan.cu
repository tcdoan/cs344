// Blelloch algorithm

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>

#define BLOCK_SIZE 256
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#define FIND_OFFSET(index) ((index) >> LOG_NUM_BANKS)

float **d_block_sums;
unsigned int d_num_elements = 0;
unsigned int d_num_levels = 0;

int get_levels(unsigned int x, unsigned int n)
{
    int levels = log((x - 0.5)) / log((float)n);
    return levels;
}

inline int low_power_of_2(int x)
{
    int e;
    frexp((float)x, &e);
    return 1 << (e - 1);
}

inline bool power_of_2(int n) { return (((n - 1) & n) == 0); }

template <bool POWER_OF_2>
__device__ void copy_to_shared_data(float *shared_data, const float *d_in, int n, int base_idx, int &ai, int &bi,
                                    int &mem_ai, int &mem_bi, int &offset_bank_a, int &offset_bank_b)
{
    int t = threadIdx.x;
    mem_ai = base_idx + threadIdx.x;
    mem_bi = mem_ai + blockDim.x;

    ai = t;
    bi = t + blockDim.x;

    offset_bank_a = FIND_OFFSET(ai);
    offset_bank_b = FIND_OFFSET(bi);

    shared_data[ai + offset_bank_a] = d_in[mem_ai];

    if (POWER_OF_2)
    {
        shared_data[bi + offset_bank_b] = (bi < n) ? d_in[mem_bi] : 0;
    }
    else
    {
        shared_data[bi + offset_bank_b] = d_in[mem_bi];
    }
}

template <bool POWER_OF_2>
__device__ void copy_to_global_mem(float *d_out, const float *shared_data, int n, int ai, int bi, int mem_ai,
                                   int mem_bi, int offset_bank_a, int offset_bank_b)
{
    d_out[mem_ai] = shared_data[ai + offset_bank_a];
    if (POWER_OF_2)
    {
        if (bi < n)
            d_out[mem_bi] = shared_data[bi + offset_bank_b];
    }
    else
    {
        d_out[mem_bi] = shared_data[bi + offset_bank_b];
    }
}

template <bool STORE_SUM>
__device__ void zero_last_element(float *shared_data, float *d_block_sums, int block_idx)
{
    if (threadIdx.x == 0)
    {
        int index = (blockDim.x << 1) - 1;
        index += FIND_OFFSET(index);

        if (STORE_SUM)
        {
            d_block_sums[block_idx] = shared_data[index];
        }

        shared_data[index] = 0;
    }
}

__device__ unsigned int leaves_2_root_scan(float *shared_data)
{
    unsigned int thid = threadIdx.x;
    unsigned int stride = 1;

    for (int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)
        {
            int i = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += FIND_OFFSET(ai);
            bi += FIND_OFFSET(bi);

            shared_data[bi] += shared_data[ai];
        }

        stride *= 2;
    }

    return stride;
}

__device__ void root_2_leaves_scan(float *s_data, unsigned int stride)
{
    unsigned int thid = threadIdx.x;
    for (int d = 1; d <= blockDim.x; d *= 2)
    {
        stride >>= 1;

        __syncthreads();
        if (thid < d)
        {
            int i = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += FIND_OFFSET(ai);
            bi += FIND_OFFSET(bi);

            float t = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
}

template <bool STORE_SUM>
__device__ void scan_block(float *shared_data, int block_idx, float *block_sums)
{
    int stride = leaves_2_root_scan(shared_data);
    zero_last_element<STORE_SUM>(shared_data, block_sums, (block_idx == 0) ? blockIdx.x : block_idx);
    root_2_leaves_scan(shared_data, stride);
}

template <bool STORE_SUM, bool POWER_OF_2>
__global__ void scan(float *d_out, const float *d_in, float *d_block_sums_level, int n, int block_idx, int base_idx)
{
    int ai, bi, mem_ai, mem_bi, offset_bank_a, offset_bank_b;
    extern __shared__ float shared_data[];

    copy_to_shared_data<POWER_OF_2>(shared_data, d_in, n,
                                    (base_idx == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)) : base_idx, ai, bi, mem_ai,
                                    mem_bi, offset_bank_a, offset_bank_b);

    scan_block<STORE_SUM>(shared_data, block_idx, d_block_sums_level);
    copy_to_global_mem<POWER_OF_2>(d_out, shared_data, n, ai, bi, mem_ai, mem_bi, offset_bank_a, offset_bank_b);
}

__global__ void uniform_add(float *d_out, float *uniforms, int n, int block_offset, int base_idx)
{
    __shared__ float init;
    if (threadIdx.x == 0)
        init = uniforms[blockIdx.x + block_offset];
    unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + base_idx + threadIdx.x;
    __syncthreads();
    d_out[address] += init;
    d_out[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * init;
}

unsigned int num_blocks(unsigned int x)
{
    unsigned int data_size = 2 * BLOCK_SIZE;
    return (x + data_size - 1) / data_size;
}

void allocate_block_sums(unsigned int elements)
{
    d_num_elements = elements;
    int levels = get_levels(elements, 2 * BLOCK_SIZE);

    d_num_levels = levels;
    d_block_sums = (float **)malloc(levels * sizeof(float *));

    int level = 0;
    do
    {
        unsigned int blocks = num_blocks(elements);
        if (blocks > 1)
        {
            cudaMalloc((void **)&d_block_sums[level++], blocks * sizeof(float));
        }
        elements = blocks;
    } while (elements > 1);
}

void deallocate_block_sums()
{
    for (int i = 0; i < d_num_levels; i++)
        cudaFree(d_block_sums[i]);
    free((void **)d_block_sums);
    d_block_sums = 0;
    d_num_levels = 0;
    d_num_elements = 0;
}

void recursive_scan(float *d_out, const float *d_in, int num_elements, int level)
{
    unsigned int blocks = num_blocks(num_elements);
    unsigned int num_threads;

    if (blocks > 1)
    {
        num_threads = BLOCK_SIZE;
    }
    else if (power_of_2(num_elements))
    {
        num_threads = num_elements / 2;
    }
    else
    {
        num_threads = low_power_of_2(num_elements);
    }

    unsigned int block_elements = num_threads * 2;
    unsigned int last_block_elements = num_elements - (blocks - 1) * block_elements;
    unsigned int last_block_threads = max(1, last_block_elements / 2);
    unsigned int np2_last_block = 0;
    unsigned int last_lock_shmem = 0;

    if (last_block_elements != block_elements)
    {
        np2_last_block = 1;

        if (!power_of_2(last_block_elements))
        {
            last_block_threads = low_power_of_2(last_block_elements);
        }

        unsigned int extra_space = (2 * last_block_threads) / NUM_BANKS;
        last_lock_shmem = sizeof(float) * (2 * last_block_threads + extra_space);
    }

    unsigned int extra_space = block_elements / NUM_BANKS;
    unsigned int shared_mem_size = sizeof(float) * (block_elements + extra_space);

    dim3 grid(max(1, blocks - np2_last_block), 1, 1);
    dim3 threads(num_threads, 1, 1);

    if (blocks > 1)
    {
        scan<true, false><<<grid, threads, shared_mem_size>>>(d_out, d_in, d_block_sums[level], num_threads * 2, 0, 0);
        if (np2_last_block)
        {
            scan<true, true><<<1, last_block_threads, last_lock_shmem>>>(
                d_out, d_in, d_block_sums[level], last_block_elements, blocks - 1, num_elements - last_block_elements);
        }

        recursive_scan(d_block_sums[level], d_block_sums[level], blocks, level + 1);
        uniform_add<<<grid, threads>>>(d_out, d_block_sums[level], num_elements - last_block_elements, 0, 0);
        if (np2_last_block)
        {
            uniform_add<<<1, last_block_threads>>>(d_out, d_block_sums[level], last_block_elements, blocks - 1,
                                                   num_elements - last_block_elements);
        }
    }
    else if (power_of_2(num_elements))
    {
        scan<false, false><<<grid, threads, shared_mem_size>>>(d_out, d_in, 0, num_threads * 2, 0, 0);
    }
    else
    {
        scan<false, true><<<grid, threads, shared_mem_size>>>(d_out, d_in, 0, num_elements, 0, 0);
    }
}

int main(int argc, char **argv)
{
    int N = 40000000;
    // int N = 65;
    unsigned int bytes = sizeof(float) * N;

    float *h_data = (float *)malloc(bytes);
    for (unsigned int i = 0; i < N; ++i)
        h_data[i] = 1.0f;

    float *d_in;
    float *d_out;

    cudaMalloc((void **)&d_in, bytes);
    cudaMalloc((void **)&d_out, bytes);

    cudaMemcpy(d_in, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_data, bytes, cudaMemcpyHostToDevice);

    allocate_block_sums(N);

    recursive_scan(d_out, d_in, N, 0);

    deallocate_block_sums();

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
}