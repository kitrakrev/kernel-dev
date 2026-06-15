"""Project B step 4 — matmul on Apple Silicon GPU via MLX.

Device config:
  Hardware : Apple M4 (10 cores: 4P + 6E)
  GPU      : Apple Silicon integrated GPU (Metal)
  Memory   : 16 GB unified
  OS       : macOS 26.2 (arm64)
  Backend  : MLX 0.31.1

M4 GPU peak fp32 ~4.6 TFLOPS (10-core variant); fp16 ~9 TFLOPS.
MLX dispatches to MPS/Metal Performance Shaders matmul kernel.
"""
import time
import mlx.core as mx

N = 2048
A = mx.random.normal(shape=(N, N))
B = mx.random.normal(shape=(N, N))
mx.eval(A, B)

# warm
C = A @ B; mx.eval(C)

t0 = time.perf_counter()
for _ in range(10):
    C = A @ B
    mx.eval(C)
ms = (time.perf_counter() - t0) / 10 * 1e3

flops = 2.0 * N * N * N
tflops = flops / (ms / 1e3) / 1e12
print(f"backend: MLX matmul (Apple M4 GPU)")
print(f"shape: ({N}, {N}) x ({N}, {N}) fp32")
print(f"per-call: {ms:.3f} ms")
print(f"perf: {tflops:.3f} TFLOPS")
print(f"M4 GPU fp32 peak ~4.6 TFLOPS -> {tflops/4.6*100:.1f}% of peak")

# fp16
A = A.astype(mx.float16); B = B.astype(mx.float16)
mx.eval(A, B)
C = A @ B; mx.eval(C)

t0 = time.perf_counter()
for _ in range(10):
    C = A @ B
    mx.eval(C)
ms = (time.perf_counter() - t0) / 10 * 1e3
tflops = flops / (ms / 1e3) / 1e12
print(f"\nshape: ({N}, {N}) x ({N}, {N}) fp16")
print(f"per-call: {ms:.3f} ms")
print(f"perf: {tflops:.3f} TFLOPS")
print(f"M4 GPU fp16 peak ~9 TFLOPS -> {tflops/9*100:.1f}% of peak")
