/* CUDA exercise 1: vector add — kernel, grid/block, cudaMemcpy.
 *
 * Device config:
 *   Hardware : NVIDIA Tesla T4 (sm_75, Turing)
 *   Memory   : 15.36 GB GDDR6
 *   Driver   : 580.82.07, CUDA 13.0 runtime
 *   Host     : Linux 6.6.122 x86_64 (Google Colab VM)
 *
 * Compile : nvcc -O3 -arch=sm_75 01_vector_add.cu -o vec_add
 * Run     : ./vec_add
 */
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA err %s\n", cudaGetErrorString(e)); return 1; } } while (0)

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);
    float *ha = (float*)malloc(bytes), *hb = (float*)malloc(bytes), *hc = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) { ha[i] = i * 1.0f; hb[i] = i * 2.0f; }

    float *da, *db, *dc;
    CHECK(cudaMalloc(&da, bytes));
    CHECK(cudaMalloc(&db, bytes));
    CHECK(cudaMalloc(&dc, bytes));
    CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vec_add<<<blocks, threads>>>(da, db, dc, N);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
    printf("N=%d, blocks=%d, threads=%d\n", N, blocks, threads);
    printf("c[0]=%.1f (expect 0.0)\n", hc[0]);
    printf("c[1]=%.1f (expect 3.0)\n", hc[1]);
    printf("c[N-1]=%.1f (expect %.1f)\n", hc[N-1], (N-1) * 3.0f);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(ha); free(hb); free(hc);
    return 0;
}
