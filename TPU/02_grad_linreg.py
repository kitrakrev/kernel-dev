"""TPU/JAX exercise 2: grad — train linear regression on TPU.

Device config:
  Hardware : Google TPU v5 lite (v5e), 1 chip
  Backend  : JAX 0.7.2, PJRT C API
  Host     : Linux 6.6.122 x86_64

Fits y = 3x + 2 using jax.value_and_grad + plain SGD.
"""
import jax
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(0)
k1, k2 = random.split(key)
N = 256
x = random.normal(k1, (N, 1))
y = 3.0 * x + 2.0 + 0.1 * random.normal(k2, (N, 1))

def loss(params, x, y):
    pred = params["w"] * x + params["b"]
    return jnp.mean((pred - y) ** 2)

grad_fn = jax.jit(jax.value_and_grad(loss))
params = {"w": jnp.array(0.0), "b": jnp.array(0.0)}
lr = 0.1

for step in range(200):
    l, g = grad_fn(params, x, y)
    params = {k: params[k] - lr * g[k] for k in params}
    if step % 40 == 0:
        print(f"step {step:3d} loss={float(l):.4f} w={float(params['w']):.4f} b={float(params['b']):.4f}")

print(f"final w={float(params['w']):.4f} (true 3.0), b={float(params['b']):.4f} (true 2.0)")
print("device:", jax.devices()[0])
