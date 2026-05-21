/* CUDA exercise 4: matrix transpose — coalesced vs strided, shared-mem tiling.
 *
 * Device config:
 *   Hardware : NVIDIA Tesla T4 (sm_75, Turing)
 *   Memory   : 15.36 GB GDDR6
 *   Driver   : 580.82.07, CUDA 13.0
 *   Host     : Linux 6.6.122 x86_64
 *
 * Compile : nvcc -O3 -arch=sm_75 04_transpose.cu -o transpose
 * Run     : ./transpose
 *
 * Naive write is strided (bad); tiled version writes coalesced. +1 pad avoids
 * shared-mem bank conflicts.
 */
#include <cstdio>
#include <cuda_runtime.h>

#define TILE 32

__global__ void naive_transpose(const float* in, float* out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n) out[x * n + y] = in[y * n + x];  // strided write
}

__global__ void tiled_transpose(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + 1];  // +1 pad: no bank conflict
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < n && y < n) out[y * n + x] = tile[threadIdx.x][threadIdx.y];
}

float time_kernel(void (*launch)(const float*, float*, int), const float* d_in, float* d_out, int n) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    dim3 block(TILE, TILE), grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    cudaEventRecord(s);
    launch<<<grid, block>>>(d_in, d_out, n);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    return ms;
}

int main() {
    const int N = 4096;
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h = (float*)malloc(bytes);
    for (int i = 0; i < N * N; ++i) h[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);

    cudaEventRecord(s);
    naive_transpose<<<grid, block>>>(d_in, d_out, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_naive; cudaEventElapsedTime(&ms_naive, s, e);

    cudaEventRecord(s);
    tiled_transpose<<<grid, block>>>(d_in, d_out, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_tiled; cudaEventElapsedTime(&ms_tiled, s, e);

    float* h_out = (float*)malloc(bytes);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool ok = (h_out[N * 1 + 0] == h[0 * N + 1]);  // out[1,0] == in[0,1]

    double gb = 2.0 * bytes / 1e9;
    printf("N=%d (matrix %dx%d, %.1f MB)\n", N, N, N, bytes / 1e6);
    printf("naive transpose: %.3f ms, %.2f GB/s\n", ms_naive, gb / (ms_naive / 1e3));
    printf("tiled transpose: %.3f ms, %.2f GB/s\n", ms_tiled, gb / (ms_tiled / 1e3));
    printf("speedup: %.2fx\n", ms_naive / ms_tiled);
    printf("correctness: %s\n", ok ? "OK" : "FAIL");

    cudaFree(d_in); cudaFree(d_out);
    free(h); free(h_out);
    return 0;
}
