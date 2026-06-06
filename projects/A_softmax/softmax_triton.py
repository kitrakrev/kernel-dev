"""Project A — softmax in Triton (T4 GPU).

Device config:
  Hardware : NVIDIA Tesla T4 (sm_75, Turing)
  Memory   : 15.36 GB GDDR6
  Driver   : 580.82.07, CUDA 13.0
  Host     : Linux 6.6.122 x86_64

Triton: write kernel in Python, JIT to PTX. One program per row, BLOCK_D wide.
"""
import time
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(y_ptr + row * n_cols + cols, y, mask=mask)

def softmax_triton(x):
    N, D = x.shape
    y = torch.empty_like(x)
    BLOCK_D = triton.next_power_of_2(D)
    softmax_kernel[(N,)](x, y, D, BLOCK_D=BLOCK_D, num_warps=8)
    return y

N, D = 4096, 4096
x = torch.randn(N, D, device='cuda', dtype=torch.float32)

# warm
y = softmax_triton(x); torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(20):
    y = softmax_triton(x)
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / 20 * 1e3

bytes_moved = 2 * N * D * 4  # 1 read + 1 write (single fused pass)
bw = bytes_moved / (ms / 1e3) / 1e9
print(f"backend: Triton (Tesla T4)")
print(f"shape: ({N}, {D})")
print(f"row sums (first 4): {y.sum(dim=-1)[:4].tolist()}")
print(f"per-call: {ms:.3f} ms")
print(f"approx bandwidth: {bw:.2f} GB/s")
