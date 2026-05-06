"""MLX exercise 2: linear regression via mx.value_and_grad + manual SGD.

Fit y = 3x + 2 from noisy samples.

Device config:
  Hardware : Apple M4 (10 cores: 4P + 6E)
  GPU      : Apple Silicon integrated GPU (Metal)
  Memory   : 16 GB unified
  OS       : macOS 26.2 (arm64)
  Backend  : MLX 0.31.1, Metal
  Device   : mx.default_device() = Device(gpu, 0)
"""
import mlx.core as mx

mx.random.seed(0)
N = 256
x = mx.random.normal(shape=(N, 1))
true_w, true_b = 3.0, 2.0
y = true_w * x + true_b + 0.1 * mx.random.normal(shape=(N, 1))

params = {"w": mx.array([0.0]), "b": mx.array([0.0])}

def loss_fn(params):
    pred = params["w"] * x + params["b"]
    return mx.mean((pred - y) ** 2)

loss_and_grad = mx.value_and_grad(loss_fn)
lr = 0.1

for step in range(200):
    loss, grads = loss_and_grad(params)
    params = {k: params[k] - lr * grads[k] for k in params}
    mx.eval(params, loss)
    if step % 40 == 0:
        print(f"step {step:3d} loss={loss.item():.4f} w={params['w'].item():.4f} b={params['b'].item():.4f}")

print(f"final w={params['w'].item():.4f} (true 3.0), b={params['b'].item():.4f} (true 2.0)")
