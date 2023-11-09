# Simple CUDA kernel to perform parallel sum reduction tree.

This kernel performs sum reduction tree within a single block. 

- For an input array of N elements main() call this kernel and launch a grid with one block of N/2 threads.
- A block can have up to 1024 threads, so this can process up to 2048 input elements. 
> See reduction.cu for code that solves 2048-elements array limitation.

## Steps
- During step 1, all N/2 threads will participate; each thread adds two elements to produce N/2 partial sums. 
- During step 2, half of the threads will drop off, and only N/4 threads will continue to participate to produce N/4 partial sums. 
- This process will continue until the last time step, in which only one thread will remain and produce the total sum.

![Figure 1](f1.png)

## Code version 1

```C++
constexpr int N = 2*1024;
__global__ void simple_reduce_kernel(float *d_out, float *d_in)
{
    unsigned int i = threadIdx.x*2;
    for (unsigned int k = 1; k <= blockDim.x; k *= 2) {
        if (threadIdx.x % k == 0) {
            d_in[i] += d_in[i+k];
        } 
        // wait for all adds at one stage are done
        __syncthreads();
    }

    if (i == 0) *d_out = d_in[0];
}

int main(int argc, char **argv)
{
    int IN_BYTES = sizeof(float) *  N;
    int OUT_BYTES = sizeof(float);

    float* h_in= new float[N];
    std::fill_n(h_in, N, 1);

    float *d_in;
    float *d_out;
    
    cudaMalloc((void **) &d_in, IN_BYTES);
    cudaMalloc((void **) &d_out, OUT_BYTES);

    cudaMemcpy(d_in, h_in, IN_BYTES, ::cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    simple_reduce_kernel<<<1, N/2>>>(d_out, d_in);
    double endTime = CycleTimer::currentSeconds();    
    printf("Time elapsed %.3f ms \n", 1000.f * (endTime - startTime));

    float h_out;
    cudaMemcpy(&h_out, d_out, OUT_BYTES, ::cudaMemcpyDeviceToHost);

    printf("sum = %.3f \n", h_out);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
```

## Minimizing control divergence 

The version 1 of kernel code has management of active and inactive threads in each iteration results in a high degree of control divergence. For example, as shown in abobe figrure, only those threads whose threadIdx.x values are even will execute the addition statement during the second iteration. Control divergence can significantly reduce the execution resource utilization efficiency, or the percentage of resources that are used in generating useful results. 

- In the first iteration, all 32 threads in a warp consume execution resources, but only half of them are active, wasting half of the execution resources. 
- During the second iteration, only one-fourth of the threads in a warp are active, wasting three-quarters of the execution resources. 
- During iteration 5, only one out of the 32 threads in a warp are active, wasting 31/32 of the execution resources. 

If the size of the input array is greater than 32, entire warps will become inactive after the fifth iteration. 
- E.g., for an input size of 256, 128 threads or four warps would be launched. 
- All four warps would have the same divergence pattern, as we explained in the previous paragraph for iterations 1 through 5. 
- During the sixth iteration, warp 1 and warp 3 would become completely inactive and thus exhibit no control divergence. 
- On the other hand, warp 0 and warp 2 would have only one active thread, exhibiting control divergence and wasting 31/32 of the execution resource. 
- During the seventh iteration, only warp 0 would be active, exhibiting control divergence and wasting 31/32 of the execution resource. 

The execution resource utilization efficiency for an input array of size N is the ratio between the total number of active threads to the total number of execution resources that are consumed. 


Total number of execution resources that are consumed is proportional to the total number of active warps across all iterations, since every active warp, no matter how few of its threads are active, consumes full execution resources. 

$$ (\frac{N}{64} * 5 + \frac{N}{64} * \frac{1}{2} + \frac{N}{64} * \frac{1}{4} + ... + 1) * 32  $$

Here:
$$ \frac{N}{2} threads * 32 \frac{threads}{wrap} = \frac{N}{64} wraps $$

Term:
$$ \frac{N}{64} $$ 
    is multiplied by 5 because all launched warps are active for five iterations. 
    
- After the fifth iteration the number of warps is reduced by half in each successive iteration. 
- The expression in parentheses gives the total number of active warps across all the iterations. 

The second term, `32`, reflects that each active warp consumes full execution resources for all 32 threads regardless of the number of active threads in these warps. 

For an input array size of 256, the consumed execution resource is (4*5+2+1)*32=736. 

The number of execution results `committed by the active threads` is the total number of active threads across all iterations:
$$ \frac{N}{64} * (32+16+8+4+2+1) + \frac{N}{64} * \frac{1}{2} *1 + \frac{N}{64} * \frac{1}{4}*1 + ... + 1 $$

- The terms in the parenthesis give the active threads in the first five iterations for all N/64 warps. 
- Starting at the sixth iteration, the number of active warps is reduced by half in each iteration, and there is only one active thread in each active warp. 

For an input array size of 256, the total number of committed results is $$4*(32+16+8+4+2+1)+2+1=255$$ 
- This result should be intuitive because the total number of operations that are needed to reduce 256 values is 255. 

Putting the previous two results together, we find that the execution resource utilization efficiency for an input array size of 256 is $$ 255/736=0.35 $$

This ratio states that the parallel execution resources did not achieve their full potential in speeding up this computation. On average, only about 35% of the resources consumed contributed to the sum reduction result. That is, we used only about 35% of the hardware’s potential to speed up the computation.

### A better assignment strategy 

To significantly reduces control divergence we should arrange the threads and their owned positions so that they can remain close to each other as time progresses.

We would like to have the stride value decrease, rather than increase, over time. The revised assignment strategy, shown below, is for an input array of 16 elements.
- Here, we assign the threads to the first half of the locations. 
- During the first iteration, each thread reaches halfway across the input array and adds an input element to its owner location. Thread 0 adds input[8] to its owned position input[0], thread 1 adds input[9] to its owned position input[1], and so on. 
- During each subsequent iteration, half of the active threads drop off, and all remaining active threads add an input element whose position is the number of active threads away from its owner position. In this example, during the third iteration there are two remaining active threads: Thread 0 adds input[2] into its owned position input[0], and thread 1 adds input[3] into its owned position input[1]. 

![Figure 2](f2.png)


### Code version 2
```C++
__global__ void convergent_reduce_kernel(float *d_out, float *d_in)
{
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            d_in[i] += d_in[i + stride];
        } 
        // wait for all adds at one stage are done
        __syncthreads();
    }

    if (i == 0) *d_out = d_in[0];
}
```

## Minimizing memory divergence 

It is important to achieve memory coalescing within each warp. Adjacent threads in a warp should access adjacent locations when they access global memory. 

The `simple_reduce_kernel` also has performance issue: memory divergence.
- Adjacent threads do not access adjacent locations. In each iteration, each thread performs two global memory reads and one global memory write. 
- The first read is from its owned location, the second read is from the location that is of stride distance away from its owned location, and the write is to its owned location.
- Since the locations owned by adjacent thread are not adjacent locations, the accesses that are made by adjacent threads will not be fully coalesced. 
- During each iteration the memory locations that are collectively accessed by a warp are of stride distance away from each other.

Thus the total number of global memory requests is as follows:

$$ (\frac{N}{64} * 5 * 2 + \frac{N}{64} *1 + \frac{N}{64} * \frac{1}{2} + \frac{N}{64} * \frac{1}{4} + ... + 1) * 3 $$


The first term $$ \frac{N}{64} * 5 * 2 $$ corresponds to the first five iterations, in which all N/64 warps have two or more active threads, so each warp performs two global memory requests.

 The remaining terms account for the final iterations, in which each warp has only one active thread and performs one global memory request and half of the warps drop out in each subsequent iteration. The multiplication by 3 accounts for the two reads and one write by each active thread during each iteration. 
 
 In the 256-element example the total number of global memory requests performed by the kernel is `(4*5*2+4+2+1)*3=141`.

For the `convergent_reduce_kernel` the adjacent threads in each warp always access adjacent locations in the global memory, so the accesses are always coalesced. As a result, each warp triggers only one global memory request on any read or write. As the iterations progress, entire warps drop out, so no global memory access will be performed by any thread in these inactive warps. Half of the warps drop out in each iteration until there is only one warp for the final five iterations. Therefore the total number of global memory requests performed by the kernel is:

$$ ( (\frac{N}{64} + \frac{N}{64} * \frac{1}{2} + \frac{N}{64} * \frac{1}{4} + ... + 1) + 5) * 3 $$

For the 256-element example the total number of global memory requests performed is `((4+2+1)+5)*3=36`. 

The improved kernel results in `141/36=3.9×` fewer global memory requests. 

For a 2048-element example the total number of global memory requests that are performed by `simple_reduce_kernel` is `(32*5*2+32+16+8+4+2+1)*3=1149`, whereas
the number of global memory requests that are performed by `convergent_reduce_kernel` is `(32+16+8+4+2+1+5)*3=204`. 

- The ratio is 5.6, even more than in the 256-element example.
- This is because of the inefficient execution pattern of the simple_reduce_kernel, in which there are more active warps during the initial five iterations of the execution and each active warp triggers twice the number of global memory requests as the convergent kernel.



## Minimizing global memory accesses 

The `convergent_reduce_kernel` can be further improved by using shared memory.

- In each iteration, threads write their partial sum result values out to the global memory, and these values are reread by the same threads and other threads in the next iteration. 
- Since the shared memory has much shorter latency and higher bandwidth than the global memory, we can further improve the execution speed by keeping the partial sum results in the shared memory. 

![](f3.png)

### shmem_convergent_reduce_kernel

```C++
__global__ void shmem_convergent_reduce_kernel(float *d_out, float *d_in)
{
    __shared__ float shared_in[BLOCK_DIM];

    unsigned int i = threadIdx.x;
    shared_in[i] = d_in[i] + d_in[i + BLOCK_DIM];

    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();

        if (threadIdx.x < stride) {
            shared_in[i] += shared_in[i + stride];
        } 
    }

    if (i == 0) *d_out = shared_in[0];
}
```

Using the `shmem_convergent_reduce_kernel`, the number of global memory accesses are reduced to the initial loading of the original contents of the input array and the final write to input[0]. 
- Thus for an N-element reduction the number of global memory accesses is just N+1. 
- Note also that both global memory reads are coalesced. 
- So with coalescing, there will be only (N/32)+1 global memory requests. 

For the 256-element example the total number of global memory requests that are triggered will be reduced from 36 for `convergent_reduce_kernel` to 8+1=9 for the shared memory kernel, a 4× improvement. 

Besides reducing the number of global memory accesses, the input array is not modified.

## Hierarchical reduction for arbitrary input length




## References
- Programming Massively Parallel Processors - A Hands-on Approach, David B. Kirk, Wen-mei W. Hwu, First Edition, Morgan Kaufmann, Elsevier, 2010
