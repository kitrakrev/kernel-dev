/* Project B step 1 — naive global-memory matmul on T4.
 *
 * Device config:
 *   Hardware : NVIDIA Tesla T4 (sm_75, Turing, 65 fp32 TFLOPS spec? No — 8.1 TFLOPS fp32, 65 TFLOPS fp16 tensor)
 *   Memory   : 15.36 GB GDDR6, ~320 GB/s peak HBM BW
 *   Driver   : 580.82.07, CUDA 13.0
 *   Host     : Linux 6.6.122 x86_64
 *
 * Compile : nvcc -O3 -arch=sm_75 01_naive_cuda.cu -o naive
 * Run     : ./naive
 *
 * Each thread reads 2N elements from global mem -> compute bound? No -- memory bound.
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

int main() {
    const int N = 2048;
    size_t bytes = (size_t)N * N * sizeof(float);
    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes), *hC = (float*)malloc(bytes);
    for (int i = 0; i < N * N; ++i) { hA[i] = 1.0f; hB[i] = 1.0f; }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes); cudaMalloc(&dB, bytes); cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);

    // warm
    matmul_naive<<<grid, block>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    for (int i = 0; i < 5; ++i) matmul_naive<<<grid, block>>>(dA, dB, dC, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    ms /= 5.f;

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);
    double flops = 2.0 * N * N * N;
    double tflops = flops / (ms / 1e3) / 1e12;
    printf("backend: naive CUDA (Tesla T4)\n");
    printf("shape: (%d, %d) x (%d, %d)\n", N, N, N, N);
    printf("C[0,0] = %.1f (expect %.1f)\n", hC[0], (float)N);
    printf("per-call: %.3f ms\n", ms);
    printf("perf: %.3f TFLOPS\n", tflops);
    printf("T4 fp32 peak: 8.1 TFLOPS -> %.1f%% of peak\n", tflops / 8.1 * 100);

    cudaFree(dA); cudaFree(dB); cudaFree(dC); free(hA); free(hB); free(hC);
    return 0;
}
