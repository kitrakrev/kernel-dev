"""CUDA perf — matmul TFLOPS sweep (cuBLAS via torch).

Device config:
  Hardware : NVIDIA Tesla T4 (sm_75, Turing)
  Memory   : 15.36 GB GDDR6
  Driver   : 580.82.07, CUDA 13.0
  Backend  : torch.matmul -> cuBLAS

fp32 -> SGEMM, fp16 -> HGEMM on Tensor Cores.
"""
import time
import torch

torch.backends.cuda.matmul.allow_tf32 = True

def bench(N, dtype, iters=20):
    A = torch.randn(N, N, device='cuda', dtype=dtype)
    B = torch.randn(N, N, device='cuda', dtype=dtype)
    C = A @ B; torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        C = A @ B
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    return ms, 2.0 * N * N * N / (ms / 1e3) / 1e12

print(f"backend: cuBLAS via torch (Tesla T4)")
print(f"{'N':>6} {'dtype':>10} {'ms':>10} {'TFLOPS':>10}")
for N in [256, 512, 1024, 2048, 4096]:
    for dt in [torch.float32, torch.float16]:
        ms, tf = bench(N, dt)
        print(f"{N:>6} {str(dt).replace('torch.',''):>10} {ms:>10.3f} {tf:>10.3f}")
print("T4 peak fp32: 8.1 TFLOPS, fp16 Tensor Core: 65 TFLOPS")
