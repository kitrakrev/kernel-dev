"""MLX exercise 4: multi-head attention from scratch.

Implements scaled-dot-product attention + multi-head wrapper.

Device config:
  Hardware : Apple M4 (10 cores: 4P + 6E)
  GPU      : Apple Silicon integrated GPU (Metal)
  Memory   : 16 GB unified
  OS       : macOS 26.2 (arm64)
  Backend  : MLX 0.31.1, Metal
  Device   : mx.default_device() = Device(gpu, 0)
"""
import math
import mlx.core as mx
import mlx.nn as nn

mx.random.seed(0)

def scaled_dot_product(q, k, v, mask=None):
    # q,k,v: (B, H, T, D)
    d = q.shape[-1]
    scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(d)
    if mask is not None:
        scores = mx.where(mask, scores, mx.array(-1e9))
    attn = mx.softmax(scores, axis=-1)
    return attn @ v, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_head = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x, causal=False):
        B, T, D = x.shape
        q = self.wq(x).reshape(B, T, self.h, self.d_head).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.h, self.d_head).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.h, self.d_head).transpose(0, 2, 1, 3)
        mask = None
        if causal:
            mask = mx.tril(mx.ones((T, T))).astype(mx.bool_)
        out, attn = scaled_dot_product(q, k, v, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.wo(out), attn

B, T, D, H = 2, 8, 64, 4
x = mx.random.normal(shape=(B, T, D))
mha = MultiHeadAttention(D, H)
out, attn = mha(x, causal=True)
mx.eval(out, attn)
print("input  :", x.shape)
print("output :", out.shape)
print("attn   :", attn.shape, "(B,H,T,T)")
print("causal check — row 0 only attends pos 0:")
print(attn[0, 0, 0])
print("row 3 attends pos 0..3:")
print(attn[0, 0, 3])
print("row sums (should be 1):", mx.sum(attn[0, 0], axis=-1))
