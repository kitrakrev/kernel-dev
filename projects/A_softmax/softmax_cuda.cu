/* Project A — row-wise softmax in raw CUDA (numerically stable, single-pass per row).
 *
 * Device config:
 *   Hardware : NVIDIA Tesla T4 (sm_75, Turing)
 *   Memory   : 15.36 GB GDDR6
 *   Driver   : 580.82.07, CUDA 13.0
 *   Host     : Linux 6.6.122 x86_64
 *
 * Compile : nvcc -O3 -arch=sm_75 softmax_cuda.cu -o softmax_cuda
 * Run     : ./softmax_cuda
 *
 * One block per row, BLOCK threads. 3 passes per row over D:
 *   1) max reduction
 *   2) exp + sum reduction
 *   3) normalize
 */
#include <cstdio>
#include <cfloat>
#include <cuda_runtime.h>

#define BLOCK 256

__global__ void softmax_row(const float* __restrict__ x, float* __restrict__ y, int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;
    const float* xr = x + row * D;
    float* yr = y + row * D;
    int tid = threadIdx.x;
    __shared__ float sdata[BLOCK];

    float m = -FLT_MAX;
    for (int j = tid; j < D; j += BLOCK) m = fmaxf(m, xr[j]);
    sdata[tid] = m; __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];

    float sum = 0.f;
    for (int j = tid; j < D; j += BLOCK) {
        float e = __expf(xr[j] - row_max);
        yr[j] = e;
        sum += e;
    }
    sdata[tid] = sum; __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];

    float inv = 1.f / row_sum;
    for (int j = tid; j < D; j += BLOCK) yr[j] *= inv;
}

int main() {
    const int N = 4096, D = 4096;
    size_t bytes = (size_t)N * D * sizeof(float);
    float* h = (float*)malloc(bytes);
    for (int i = 0; i < N * D; ++i) h[i] = (i % 17) * 0.01f;

    float *dx, *dy;
    cudaMalloc(&dx, bytes); cudaMalloc(&dy, bytes);
    cudaMemcpy(dx, h, bytes, cudaMemcpyHostToDevice);

    // warm
    softmax_row<<<N, BLOCK>>>(dx, dy, N, D);
    cudaDeviceSynchronize();

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 20; ++i) softmax_row<<<N, BLOCK>>>(dx, dy, N, D);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    ms /= 20.f;

    float* hy = (float*)malloc(bytes);
    cudaMemcpy(hy, dy, bytes, cudaMemcpyDeviceToHost);
    float row_sum = 0.f;
    for (int j = 0; j < D; ++j) row_sum += hy[j];
    double bw = 3.0 * bytes / 1e9 / (ms / 1e3);
    printf("backend: CUDA (Tesla T4)\n");
    printf("shape: (%d, %d)\n", N, D);
    printf("row 0 sum (should ~1.0): %.6f\n", row_sum);
    printf("per-call: %.3f ms\n", ms);
    printf("approx bandwidth: %.2f GB/s\n", bw);

    cudaFree(dx); cudaFree(dy); free(h); free(hy);
    return 0;
}
