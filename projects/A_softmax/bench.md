# Project A — softmax 4 ways

Row-wise softmax over `(4096, 4096)` float32. 20-iter timing after warmup.

| Backend  | Hardware       | per-call (ms) | approx BW (GB/s) | Notes                       |
|----------|----------------|--------------:|-----------------:|-----------------------------|
| MLX      | Apple M4       |         7.595 |            26.51 | local, Metal                |
| JAX/TPU  | TPU v5e (1ch)  |         0.517 |           389.14 | XLA fuses 3 passes -> 1     |
| CUDA     | Tesla T4       |         0.844 |           238.66 | hand-written, 3 passes      |
| Triton   | Tesla T4       |         0.590 |           227.65 | single fused kernel         |

Observations:
- **TPU wins on time** for this shape — XLA fuses the max/exp/sum into a
  single sweep with TPU's high HBM bandwidth (~819 GB/s peak v5e).
- **Triton vs raw CUDA** on the same T4: Triton slightly faster because the
  generated kernel does one pass, not three (raw CUDA above is intentionally
  pedagogical with 3 separate sweeps over D).
- **MLX/M4** is bandwidth-limited by laptop HBM/LPDDR; M-series GPU peaks much
  lower than discrete accelerators.
- T4 peak HBM BW ≈ 320 GB/s — CUDA/Triton are within 70-75% of peak.
- TPU v5e peak HBM BW ≈ 819 GB/s — JAX is at ~47% of peak (small problem,
  launch/sync dominates).

Reproduce:
```bash
python3 softmax_mlx.py
colab exec -s tpu -f softmax_jax.py
colab exec -s t4 -f softmax_triton.py
# CUDA: compile + run on T4 (see softmax_cuda.cu header)
```
