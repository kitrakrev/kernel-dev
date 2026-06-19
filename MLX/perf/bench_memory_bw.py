"""MLX perf — memory bandwidth via copy + add streaming kernels.

Device config:
  Hardware : Apple M4 (10 cores: 4P + 6E)
  GPU      : Apple Silicon integrated GPU (Metal)
  Memory   : 16 GB LPDDR5X unified (~120 GB/s peak)
  Backend  : MLX 0.31.1
"""
import time
import mlx.core as mx

def bench(op_name, fn, bytes_per_iter, iters=20):
    fn()  # warm
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    mx.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    bw = bytes_per_iter / (ms / 1e3) / 1e9
    print(f"{op_name:>10} : {ms:>8.3f} ms  {bw:>8.2f} GB/s")

N = 1 << 24  # 16M floats = 64 MB
x = mx.random.normal(shape=(N,))
y = mx.random.normal(shape=(N,))
mx.eval(x, y)

# copy: read 4N, write 4N -> 8N bytes
def op_copy():
    z = x + 0.0
    mx.eval(z)
bench("copy", op_copy, 2 * N * 4)

# add: read 4N + 4N, write 4N -> 12N
def op_add():
    z = x + y
    mx.eval(z)
bench("add", op_add, 3 * N * 4)

# scale: read 4N, write 4N -> 8N
def op_scale():
    z = x * 2.0
    mx.eval(z)
bench("scale", op_scale, 2 * N * 4)
