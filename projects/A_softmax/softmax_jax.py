"""Project A — softmax in JAX (TPU v5e).

Device config:
  Hardware : Google TPU v5 lite (v5e), 1 chip
  Backend  : JAX 0.7.2, PJRT C API
  Host     : Linux 6.6.122 x86_64
"""
import time
import jax
import jax.numpy as jnp
from jax import random

def softmax(x, axis=-1):
    m = jnp.max(x, axis=axis, keepdims=True)
    e = jnp.exp(x - m)
    return e / jnp.sum(e, axis=axis, keepdims=True)

softmax_jit = jax.jit(softmax)

key = random.PRNGKey(0)
N, D = 4096, 4096
x = random.normal(key, (N, D))

# warm
y = softmax_jit(x).block_until_ready()

t0 = time.perf_counter()
for _ in range(20):
    y = softmax_jit(x).block_until_ready()
ms = (time.perf_counter() - t0) / 20 * 1e3

bytes_moved = 3 * N * D * 4
bw = bytes_moved / (ms / 1e3) / 1e9
print(f"backend: JAX/TPU")
print(f"device: {jax.devices()[0]}")
print(f"shape: ({N}, {D})")
print(f"row sums (first 4): {jnp.sum(y, axis=-1)[:4].tolist()}")
print(f"per-call: {ms:.3f} ms")
print(f"approx bandwidth: {bw:.2f} GB/s")
