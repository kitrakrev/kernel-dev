/* CUDA exercise 2: SAXPY — y = a*x + y, classic streaming kernel.
 *
 * Device config:
 *   Hardware : NVIDIA Tesla T4 (sm_75, Turing)
 *   Memory   : 15.36 GB GDDR6
 *   Driver   : 580.82.07, CUDA 13.0
 *   Host     : Linux 6.6.122 x86_64
 *
 * Compile : nvcc -O3 -arch=sm_75 02_saxpy.cu -o saxpy
 * Run     : ./saxpy
 */
#include <cstdio>
#include <cuda_runtime.h>

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    const int N = 1 << 24;
    size_t bytes = N * sizeof(float);
    float *hx = (float*)malloc(bytes), *hy = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) { hx[i] = 1.0f; hy[i] = 2.0f; }

    float *dx, *dy;
    cudaMalloc(&dx, bytes); cudaMalloc(&dy, bytes);
    cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    saxpy<<<(N + 255) / 256, 256>>>(N, 3.0f, dx, dy);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);

    cudaMemcpy(hy, dy, bytes, cudaMemcpyDeviceToHost);
    printf("N=%d, a=3.0, x=1.0, y=2.0\n", N);
    printf("y[0]=%.1f (expect 5.0)\n", hy[0]);
    printf("y[N-1]=%.1f (expect 5.0)\n", hy[N-1]);
    printf("kernel time: %.3f ms\n", ms);
    double gb = 3.0 * bytes / 1e9;  // 2 reads + 1 write
    printf("effective bandwidth: %.2f GB/s\n", gb / (ms / 1e3));

    cudaFree(dx); cudaFree(dy); free(hx); free(hy);
    return 0;
}
