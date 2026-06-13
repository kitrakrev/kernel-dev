"""Project B step 3 — Tensor Core matmul via cuBLAS (fp16 -> fp32 accumulate).

Device config:
  Hardware : NVIDIA Tesla T4 (sm_75, Turing — 2nd-gen Tensor Cores)
  Peak     : 65 TFLOPS fp16 (Tensor Core), 8.1 TFLOPS fp32
  Memory   : 15.36 GB GDDR6
  Driver   : 580.82.07, CUDA 13.0
  Host     : Linux 6.6.122 x86_64

Uses torch.matmul on fp16 inputs -> cuBLAS HGEMM -> Tensor Core mma instructions.
This is the realistic 'tensor core' rung in the ladder — beating cuBLAS by hand
needs CUTLASS / wmma intrinsics and is outside scope here.
"""
import time
import torch

assert torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = True

N = 2048
A = torch.randn(N, N, device='cuda', dtype=torch.float16)
B = torch.randn(N, N, device='cuda', dtype=torch.float16)

# warm
C = A @ B; torch.cuda.synchronize()

iters = 20
t0 = time.perf_counter()
for _ in range(iters):
    C = A @ B
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / iters * 1e3

flops = 2.0 * N * N * N
tflops = flops / (ms / 1e3) / 1e12
print(f"backend: cuBLAS HGEMM (Tensor Cores, fp16, Tesla T4)")
print(f"shape: ({N}, {N}) x ({N}, {N}) fp16")
print(f"per-call: {ms:.3f} ms")
print(f"perf: {tflops:.3f} TFLOPS")
print(f"T4 fp16 Tensor Core peak: 65 TFLOPS -> {tflops/65*100:.1f}% of peak")
