"""Project A — softmax in MLX (Apple Silicon).

Device config:
  Hardware : Apple M4 (10 cores: 4P + 6E)
  GPU      : Apple Silicon integrated GPU (Metal)
  Memory   : 16 GB unified
  OS       : macOS 26.2 (arm64)
  Backend  : MLX 0.31.1
"""
import time
import mlx.core as mx

def softmax(x, axis=-1):
    m = mx.max(x, axis=axis, keepdims=True)
    e = mx.exp(x - m)
    return e / mx.sum(e, axis=axis, keepdims=True)

mx.random.seed(0)
N, D = 4096, 4096
x = mx.random.normal(shape=(N, D))
mx.eval(x)

# warm
y = softmax(x); mx.eval(y)

t0 = time.perf_counter()
for _ in range(20):
    y = softmax(x); mx.eval(y)
ms = (time.perf_counter() - t0) / 20 * 1e3

bytes_moved = 3 * N * D * 4  # read x, write y, plus reduce reads ~3x
bw = bytes_moved / (ms / 1e3) / 1e9
print(f"backend: MLX")
print(f"shape: ({N}, {D})")
print(f"row sums (first 4): {mx.sum(y, axis=-1)[:4].tolist()}")
print(f"per-call: {ms:.3f} ms")
print(f"approx bandwidth: {bw:.2f} GB/s")
