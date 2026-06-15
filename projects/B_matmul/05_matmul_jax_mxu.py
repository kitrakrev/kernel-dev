"""Project B step 5 — matmul on TPU v5e MXU via JAX.

Device config:
  Hardware : Google TPU v5 lite (v5e), 1 chip
  Peak     : v5e ~197 TFLOPS bf16 (MXU systolic array)
  Backend  : JAX 0.7.2, PJRT C API
  Host     : Linux 6.6.122 x86_64

XLA lowers jnp.matmul to MXU systolic-array ops (128x128 bf16 native shape).
"""
import time
import jax
import jax.numpy as jnp
from jax import random

N = 2048
key = random.PRNGKey(0)
A = random.normal(key, (N, N), dtype=jnp.bfloat16)
B = random.normal(random.PRNGKey(1), (N, N), dtype=jnp.bfloat16)

matmul = jax.jit(lambda a, b: a @ b)

# warm
C = matmul(A, B).block_until_ready()

t0 = time.perf_counter()
for _ in range(20):
    C = matmul(A, B).block_until_ready()
ms = (time.perf_counter() - t0) / 20 * 1e3

flops = 2.0 * N * N * N
tflops = flops / (ms / 1e3) / 1e12
print(f"backend: JAX/XLA -> MXU (TPU v5e)")
print(f"device: {jax.devices()[0]}")
print(f"shape: ({N}, {N}) x ({N}, {N}) bf16")
print(f"per-call: {ms:.3f} ms")
print(f"perf: {tflops:.3f} TFLOPS")
print(f"v5e bf16 peak ~197 TFLOPS -> {tflops/197*100:.1f}% of peak")
