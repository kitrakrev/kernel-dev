"""MLX exercise 5: tiny GPT-style transformer forward pass + greedy decode.

Builds a small decoder stack from scratch (no HF download), random weights.
Demonstrates inference loop pattern used by mlx-lm for Llama.

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

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.h = h
        self.dh = d // h

    def attn(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dh).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.dh)
        mask = mx.tril(mx.ones((T, T))).astype(mx.bool_)
        scores = mx.where(mask, scores, mx.array(-1e9))
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.proj(out)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(nn.gelu(self.fc1(self.ln2(x))))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab=128, d=64, h=4, n_layers=2, max_t=32):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_t, d)
        self.blocks = [Block(d, h) for _ in range(n_layers)]
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.max_t = max_t

    def __call__(self, idx):
        B, T = idx.shape
        pos = mx.arange(T)
        x = self.tok(idx) + self.pos(pos)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))

VOCAB = 128
model = TinyGPT(vocab=VOCAB)
mx.eval(model.parameters())

prompt = mx.array([[1, 2, 3, 4, 5]])
print("prompt:", prompt.tolist())

generated = prompt
for step in range(10):
    logits = model(generated)
    next_id = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    generated = mx.concatenate([generated, next_id], axis=1)
    mx.eval(generated)

print("generated:", generated.tolist())
print("len:", generated.shape[1])
