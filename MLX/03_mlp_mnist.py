"""MLX exercise 3: MLP on synthetic MNIST-shaped data (784 -> 10).

Skips download; trains a 3-layer MLP via mlx.nn.Module on synthetic labels.

Device config:
  Hardware : Apple M4 (10 cores: 4P + 6E)
  GPU      : Apple Silicon integrated GPU (Metal)
  Memory   : 16 GB unified
  OS       : macOS 26.2 (arm64)
  Backend  : MLX 0.31.1, Metal
  Device   : mx.default_device() = Device(gpu, 0)
"""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

mx.random.seed(0)

class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=128, out=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out)

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return self.fc3(x)

N, D, C = 1024, 784, 10
X = mx.random.normal(shape=(N, D))
W_true = mx.random.normal(shape=(D, C))
y = mx.argmax(X @ W_true, axis=1)

model = MLP()
mx.eval(model.parameters())

def loss_fn(model, X, y):
    logits = model(X)
    return nn.losses.cross_entropy(logits, y, reduction="mean")

loss_and_grad = nn.value_and_grad(model, loss_fn)
lr = 0.05

loss = mx.array(0.0)
acc = 0.0
for epoch in range(20):
    loss, grads = loss_and_grad(model, X, y)
    updated = tree_map(lambda p, g: p - lr * g, model.parameters(), grads)
    model.update(updated)
    mx.eval(model.parameters(), loss)
    logits = model(X)
    acc = mx.mean(mx.argmax(logits, axis=1) == y).item()
    if epoch % 4 == 0:
        print(f"epoch {epoch:2d} loss={loss.item():.4f} acc={acc:.3f}")
print(f"final loss={loss.item():.4f} acc={acc:.3f}")
