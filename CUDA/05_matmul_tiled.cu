/* CUDA exercise 5: tiled matmul (shared-mem), naive vs tiled comparison.
 *
 * Device config:
 *   Hardware : NVIDIA Tesla T4 (sm_75, Turing)
 *   Memory   : 15.36 GB GDDR6
 *   Driver   : 580.82.07, CUDA 13.0
 *   Host     : Linux 6.6.122 x86_64
 *
 * Compile : nvcc -O3 -arch=sm_75 05_matmul_tiled.cu -o matmul
 * Run     : ./matmul
 *
 * Naive: each thread reads N elements from A and N from B in global memory.
 * Tiled: cooperatively load TILE x TILE blocks into shared memory, reuse.
 * Expected speedup: ~3-5x on T4 vs naive at N=1024 (still far from cuBLAS).
 */
#include <cstdio>
#include <cuda_runtime.h>

#define TILE 16

__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float acc = 0.f;
        for (int k = 0; k < N; ++k) acc += A[row * N + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int A_col = t * TILE + threadIdx.x;
        int B_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < N && A_col < N) ? A[row * N + A_col] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (B_row < N && col < N) ? B[B_row * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = acc;
}

int main() {
    const int N = 1024;
    size_t bytes = (size_t)N * N * sizeof(float);
    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes), *hC = (float*)malloc(bytes);
    for (int i = 0; i < N * N; ++i) { hA[i] = 1.0f; hB[i] = 2.0f; }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes); cudaMalloc(&dB, bytes); cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);

    cudaEventRecord(s);
    matmul_naive<<<grid, block>>>(dA, dB, dC, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_naive; cudaEventElapsedTime(&ms_naive, s, e);

    cudaEventRecord(s);
    matmul_tiled<<<grid, block>>>(dA, dB, dC, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_tiled; cudaEventElapsedTime(&ms_tiled, s, e);

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);
    double flops = 2.0 * N * N * N;
    printf("N=%d (matmul %dx%d)\n", N, N, N);
    printf("naive: %.3f ms, %.2f GFLOPS\n", ms_naive, flops / (ms_naive / 1e3) / 1e9);
    printf("tiled: %.3f ms, %.2f GFLOPS\n", ms_tiled, flops / (ms_tiled / 1e3) / 1e9);
    printf("speedup: %.2fx\n", ms_naive / ms_tiled);
    printf("C[0,0]=%.1f (expect %.1f)\n", hC[0], 2.0f * N);
    printf("C[N-1,N-1]=%.1f (expect %.1f)\n", hC[N * N - 1], 2.0f * N);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
