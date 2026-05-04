"""MLX exercise 1: mx.array basics, ops, mx.eval semantics.

MLX is lazy — ops build a graph; mx.eval() forces materialization.

Device config:
  Hardware : Apple M4 (10 cores: 4P + 6E)
  GPU      : Apple Silicon integrated GPU (Metal)
  Memory   : 16 GB unified
  OS       : macOS 26.2 (arm64)
  Backend  : MLX 0.31.1, Metal
  Device   : mx.default_device() = Device(gpu, 0)
"""
import mlx.core as mx

a = mx.array([1.0, 2.0, 3.0, 4.0])
b = mx.array([10.0, 20.0, 30.0, 40.0])

c = a + b
d = c * 2.0
e = mx.sum(d)

print("graph built (lazy), no compute yet")
mx.eval(e)
print("after eval:")
print("a =", a)
print("b =", b)
print("c = a+b =", c)
print("d = c*2 =", d)
print("e = sum(d) =", e.item())

m = mx.random.normal(shape=(3, 4))
n = mx.random.normal(shape=(4, 2))
out = m @ n
mx.eval(out)
print("matmul (3,4)@(4,2) =", out.shape)
print(out)
