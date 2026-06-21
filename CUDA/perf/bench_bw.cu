/* CUDA perf — memory bandwidth (copy/add/scale streaming).
 *
 * Device config:
 *   Hardware : NVIDIA Tesla T4 (sm_75)
 *   Memory   : 15.36 GB GDDR6, ~320 GB/s peak HBM BW
 *   Driver   : 580.82.07, CUDA 13.0
 *
 * Compile : nvcc -O3 -arch=sm_75 bench_bw.cu -o bench_bw
 */
#include <cstdio>
#include <cuda_runtime.h>

__global__ void k_copy (const float* a, float* c, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]; }
__global__ void k_scale(const float* a, float* c, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]*2.f; }
__global__ void k_add  (const float* a, const float* b, float* c, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i]; }

float bench3(void(*k)(const float*, const float*, float*, int), const float* a, const float* b, float* c, int n, double bytes) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    k<<<(n+255)/256, 256>>>(a,b,c,n); cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < 20; ++i) k<<<(n+255)/256, 256>>>(a,b,c,n);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e); ms /= 20.f;
    return bytes / 1e9 / (ms / 1e3);
}
float bench2(void(*k)(const float*, float*, int), const float* a, float* c, int n, double bytes) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    k<<<(n+255)/256, 256>>>(a,c,n); cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < 20; ++i) k<<<(n+255)/256, 256>>>(a,c,n);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e); ms /= 20.f;
    return bytes / 1e9 / (ms / 1e3);
}

int main() {
    const int N = 1 << 26;  // 64M
    size_t bytes = N * sizeof(float);
    float *da, *db, *dc;
    cudaMalloc(&da, bytes); cudaMalloc(&db, bytes); cudaMalloc(&dc, bytes);
    cudaMemset(da, 0, bytes); cudaMemset(db, 0, bytes);

    printf("backend: CUDA streaming kernels (Tesla T4)\n");
    printf("N=%d (%.0f MB per array)\n", N, bytes/1e6);
    printf("%10s : %8s\n", "op", "GB/s");
    printf("%10s : %8.2f\n", "copy",  bench2(k_copy,  da, dc, N, 2.0*bytes));
    printf("%10s : %8.2f\n", "scale", bench2(k_scale, da, dc, N, 2.0*bytes));
    printf("%10s : %8.2f\n", "add",   bench3(k_add,   da, db, dc, N, 3.0*bytes));
    printf("T4 HBM peak ~320 GB/s\n");

    cudaFree(da); cudaFree(db); cudaFree(dc);
    return 0;
}
