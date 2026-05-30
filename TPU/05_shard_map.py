"""TPU/JAX exercise 5: data parallel via shard_map (and degraded path for 1-chip).

Device config:
  Hardware : Google TPU v5 lite (v5e), 1 chip (single-device)
  Backend  : JAX 0.7.2, PJRT C API
  Host     : Linux 6.6.122 x86_64

Notes: v5e1 has 1 chip, so sharding across 'data' axis is degenerate (mesh
size 1). Demonstrates the API; on a multi-chip pod (v5e-8 etc.) the same
code would actually parallelize.
"""
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import numpy as np

devices = jax.devices()
print("device count:", len(devices), "device[0]:", devices[0])

mesh = Mesh(np.array(devices).reshape(len(devices)), axis_names=("data",))
print("mesh:", mesh)

key = random.PRNGKey(0)
W = random.normal(key, (128, 128))
X = random.normal(random.PRNGKey(1), (64, 128))

@jax.jit
def per_shard(x, w):
    # each shard computes its own slice locally
    return jnp.tanh(x @ w)

sharded = shard_map(
    per_shard,
    mesh=mesh,
    in_specs=(P("data", None), P(None, None)),
    out_specs=P("data", None),
)

out = sharded(X, W).block_until_ready()
print("input shape:", X.shape)
print("output shape:", out.shape)
print("sample mean:", float(out.mean()))

# show sharding spec
print("output sharding:", out.sharding)
