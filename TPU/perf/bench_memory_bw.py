"""TPU perf — HBM bandwidth via element-wise streaming ops.

Device config:
  Hardware : Google TPU v5 lite (v5e), 1 chip (HBM ~819 GB/s peak)
  Backend  : JAX 0.7.2, PJRT C API
"""
import time
import jax
import jax.numpy as jnp
from jax import random

def bench(name, fn, x, bytes_per_iter, iters=50):
    y = fn(x).block_until_ready()  # warm
    t0 = time.perf_counter()
    for _ in range(iters):
        y = fn(x).block_until_ready()
    ms = (time.perf_counter() - t0) / iters * 1e3
    bw = bytes_per_iter / (ms / 1e3) / 1e9
    print(f"{name:>10} : {ms:>8.4f} ms  {bw:>8.2f} GB/s")

N = 1 << 24
x = random.normal(random.PRNGKey(0), (N,))

copy = jax.jit(lambda a: a + 0.0)
scale = jax.jit(lambda a: a * 2.0)
addk = jax.jit(lambda a: a + a)

print("backend: JAX/TPU v5e streaming")
bench("copy",  copy,  x, 2 * N * 4)
bench("scale", scale, x, 2 * N * 4)
bench("add",   addk,  x, 2 * N * 4)  # XLA may fuse a+a -> 2a
print("v5e HBM peak ~819 GB/s")
