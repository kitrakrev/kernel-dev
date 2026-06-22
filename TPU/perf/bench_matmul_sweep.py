"""TPU perf — matmul TFLOPS sweep across sizes + dtypes.

Device config:
  Hardware : Google TPU v5 lite (v5e), 1 chip
  Backend  : JAX 0.7.2, PJRT C API
  Host     : Linux 6.6.122 x86_64

v5e peak: bf16 ~197 TFLOPS, fp32 (no MXU acceleration) much lower.
"""
import time
import jax
import jax.numpy as jnp
from jax import random

def bench(N, dtype, iters=20):
    key = random.PRNGKey(0)
    A = random.normal(key, (N, N), dtype=dtype)
    B = random.normal(random.PRNGKey(1), (N, N), dtype=dtype)
    matmul = jax.jit(lambda a, b: a @ b)
    C = matmul(A, B).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        C = matmul(A, B).block_until_ready()
    ms = (time.perf_counter() - t0) / iters * 1e3
    return ms, 2.0 * N * N * N / (ms / 1e3) / 1e12

print(f"backend: JAX MXU (TPU v5e)")
print(f"device: {jax.devices()[0]}")
print(f"{'N':>6} {'dtype':>10} {'ms':>10} {'TFLOPS':>10}")
for N in [256, 512, 1024, 2048, 4096]:
    for dt in [jnp.bfloat16, jnp.float32]:
        ms, tf = bench(N, dt)
        print(f"{N:>6} {str(dt.dtype):>10} {ms:>10.3f} {tf:>10.3f}")
print("v5e bf16 peak ~197 TFLOPS")
