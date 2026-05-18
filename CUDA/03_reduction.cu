/* CUDA exercise 3: reduction (sum) — shared-mem tree reduce.
 *
 * Device config:
 *   Hardware : NVIDIA Tesla T4 (sm_75, Turing)
 *   Memory   : 15.36 GB GDDR6
 *   Driver   : 580.82.07, CUDA 13.0
 *   Host     : Linux 6.6.122 x86_64
 *
 * Compile : nvcc -O3 -arch=sm_75 03_reduction.cu -o reduce
 * Run     : ./reduce
 *
 * Note: avoids bank conflicts by sequential addressing (Mark Harris pattern 3).
 */
#include <cstdio>
#include <cuda_runtime.h>

#define BLOCK 256

__global__ void reduce_block(const float* in, float* out, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;
    float v = 0.f;
    if (i < n)               v += in[i];
    if (i + blockDim.x < n)  v += in[i + blockDim.x];
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

int main() {
    const int N = 1 << 22;  // 4M
    size_t bytes = N * sizeof(float);
    float* h = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) h[i] = 1.0f;

    float *d_in, *d_partial;
    int blocks = (N + BLOCK * 2 - 1) / (BLOCK * 2);
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_partial, blocks * sizeof(float));
    cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice);

    reduce_block<<<blocks, BLOCK>>>(d_in, d_partial, N);
    float* h_partial = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    double total = 0.0;
    for (int i = 0; i < blocks; ++i) total += h_partial[i];
    printf("N=%d, blocks=%d\n", N, blocks);
    printf("sum = %.1f (expect %.1f)\n", total, (double)N);

    cudaFree(d_in); cudaFree(d_partial);
    free(h); free(h_partial);
    return 0;
}
