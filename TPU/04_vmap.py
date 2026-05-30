"""TPU/JAX exercise 4: vmap — write per-example fn, batch it.

Device config:
  Hardware : Google TPU v5 lite (v5e), 1 chip
  Backend  : JAX 0.7.2, PJRT C API
  Host     : Linux 6.6.122 x86_64
"""
import jax
import jax.numpy as jnp
from jax import random
import time

key = random.PRNGKey(0)
W = random.normal(key, (64, 64))

def single(x):
    # per-example: (64,) -> (64,)
    return jnp.tanh(W @ x)

batched = jax.vmap(single)
batched_jit = jax.jit(batched)

X = random.normal(random.PRNGKey(1), (1024, 64))

# warm + run
_ = batched_jit(X).block_until_ready()
t0 = time.perf_counter()
for _ in range(10):
    _ = batched_jit(X).block_until_ready()
ms = (time.perf_counter() - t0) / 10 * 1e3

print("single example shape:", single(X[0]).shape)
print("vmapped batch shape:", batched(X).shape)
print(f"jit+vmap mean time: {ms:.3f} ms/iter (batch=1024)")

# nested vmap: 2D batch
def f(x):
    return jnp.sum(x ** 2)

XX = jnp.ones((4, 8, 16))
print("nested vmap shape:", jax.vmap(jax.vmap(f))(XX).shape)
print("device:", jax.devices()[0])
