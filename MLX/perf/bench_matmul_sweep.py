"""MLX perf — matmul TFLOPS sweep across sizes + dtypes.

Device config:
  Hardware : Apple M4 (10 cores: 4P + 6E)
  GPU      : Apple Silicon integrated GPU (Metal)
  Memory   : 16 GB unified
  Backend  : MLX 0.31.1
"""
import time
import mlx.core as mx

def bench(N, dtype, iters=10):
    A = mx.random.normal(shape=(N, N)).astype(dtype)
    B = mx.random.normal(shape=(N, N)).astype(dtype)
    mx.eval(A, B)
    C = A @ B; mx.eval(C)  # warm
    t0 = time.perf_counter()
    for _ in range(iters):
        C = A @ B
        mx.eval(C)
    ms = (time.perf_counter() - t0) / iters * 1e3
    tflops = 2.0 * N * N * N / (ms / 1e3) / 1e12
    return ms, tflops

print(f"{'N':>6} {'dtype':>10} {'ms':>10} {'TFLOPS':>10}")
for N in [256, 512, 1024, 2048, 4096]:
    for dt in [mx.float32, mx.float16]:
        ms, tf = bench(N, dt)
        print(f"{N:>6} {str(dt).split('.')[-1]:>10} {ms:>10.3f} {tf:>10.3f}")
