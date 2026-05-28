"""TPU/JAX exercise 3: MLP on synthetic MNIST-shaped data with flax.nnx.

Device config:
  Hardware : Google TPU v5 lite (v5e), 1 chip
  Backend  : JAX 0.7.2 + flax.nnx, PJRT C API
  Host     : Linux 6.6.122 x86_64
"""
import jax
import jax.numpy as jnp
from jax import random
from flax import nnx
import optax

class MLP(nnx.Module):
    def __init__(self, d_in, d_h, d_out, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(d_in, d_h, rngs=rngs)
        self.fc2 = nnx.Linear(d_h, d_h, rngs=rngs)
        self.fc3 = nnx.Linear(d_h, d_out, rngs=rngs)
    def __call__(self, x):
        x = jax.nn.relu(self.fc1(x))
        x = jax.nn.relu(self.fc2(x))
        return self.fc3(x)

key = random.PRNGKey(0)
k1, k2, k3 = random.split(key, 3)
N, D, C = 1024, 784, 10
X = random.normal(k1, (N, D))
W = random.normal(k2, (D, C))
y = jnp.argmax(X @ W, axis=1)

model = MLP(D, 128, C, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.sgd(0.05), wrt=nnx.Param)

@nnx.jit
def step_fn(model, optimizer, X, y):
    def loss_fn(m):
        logits = m(X)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

for epoch in range(20):
    loss = step_fn(model, optimizer, X, y)
    if epoch % 4 == 0:
        logits = model(X)
        acc = float((jnp.argmax(logits, axis=1) == y).mean())
        print(f"epoch {epoch:2d} loss={float(loss):.4f} acc={acc:.3f}")

logits = model(X)
acc = float((jnp.argmax(logits, axis=1) == y).mean())
print(f"final loss={float(loss):.4f} acc={acc:.3f}")
print("device:", jax.devices()[0])
