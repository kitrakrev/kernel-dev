/* Project B step 2 — shared-memory tiled matmul on T4.
 *
 * Device config:
 *   Hardware : NVIDIA Tesla T4 (sm_75, Turing)
 *   Memory   : 15.36 GB GDDR6, ~320 GB/s peak HBM BW
 *   Driver   : 580.82.07, CUDA 13.0
 *   Host     : Linux 6.6.122 x86_64
 *
 * Compile : nvcc -O3 -arch=sm_75 02_tiled_cuda.cu -o tiled
 *
 * Tile A, B blocks into shared mem; reuse across TILE inner iterations.
 * Arithmetic intensity: 2*TILE flops per byte loaded -> approaches compute-bound.
 */
#include <cstdio>
#include <cuda_runtime.h>

#define TILE 32

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
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = acc;
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

    matmul_tiled<<<grid, block>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    for (int i = 0; i < 10; ++i) matmul_tiled<<<grid, block>>>(dA, dB, dC, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    ms /= 10.f;

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);
    double flops = 2.0 * N * N * N;
    double tflops = flops / (ms / 1e3) / 1e12;
    printf("backend: tiled CUDA (TILE=%d, T4)\n", TILE);
    printf("shape: (%d, %d) x (%d, %d)\n", N, N, N, N);
    printf("C[0,0] = %.1f (expect %.1f)\n", hC[0], (float)N);
    printf("per-call: %.3f ms\n", ms);
    printf("perf: %.3f TFLOPS\n", tflops);
    printf("T4 fp32 peak: 8.1 TFLOPS -> %.1f%% of peak\n", tflops / 8.1 * 100);

    cudaFree(dA); cudaFree(dB); cudaFree(dC); free(hA); free(hB); free(hC);
    return 0;
}
