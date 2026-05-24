"""TPU/JAX exercise 1: jnp ops, jit, inspect HLO.

Device config:
  Hardware : Google TPU v5 lite (v5e), 1 chip
  Backend  : JAX 0.7.2, PJRT C API (TFRT TPU v5 lite, cl/795515814)
  Host     : Linux 6.6.122 x86_64 (Google Colab VM)
  Device   : jax.devices() = [TpuDevice(id=0, process_index=0,
             coords=(0,0,0), core_on_chip=0)]
"""
import jax
import jax.numpy as jnp

print("device:", jax.devices()[0])

def f(x, y):
    return jnp.dot(x, y) + jnp.sin(x).sum()

x = jnp.ones((128, 64))
y = jnp.ones((64, 32))

# eager
out = f(x, y)
print("eager out shape:", out.shape, "sum:", float(out.sum()))

# jit
fj = jax.jit(f)
out = fj(x, y).block_until_ready()
print("jit out shape:", out.shape, "sum:", float(out.sum()))

# inspect HLO of compiled fn
hlo = fj.lower(x, y).compile().as_text()
print("--- compiled HLO (first 20 lines) ---")
for line in hlo.splitlines()[:20]:
    print(line)
