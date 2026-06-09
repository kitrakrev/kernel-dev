# kernel-dev

Hands-on accelerator kernels and training primitives across multiple stacks.

## Layout

```
MLX/    Apple Silicon exercises (local M4)
CUDA/   NVIDIA exercises (Colab T4)
TPU/    JAX/TPU exercises (Colab v5e)
projects/
  A_softmax/   softmax 4 ways (MLX, JAX, CUDA, Triton)
```

## Device matrix

| Stack | Hardware            | Compiler / runtime        | Verified on             |
|-------|---------------------|---------------------------|-------------------------|
| MLX   | Apple M4, 16 GB     | MLX 0.31.1, Metal         | macOS 26.2 arm64        |
| CUDA  | Tesla T4 (sm_75)    | nvcc, CUDA 13.0           | Colab Linux 6.6 x86_64  |
| TPU   | v5 lite (v5e), 1 ch | JAX 0.7.2, PJRT C API     | Colab Linux 6.6 x86_64  |

## CUDA exercises

| # | File                  | Concept                          |
|---|-----------------------|----------------------------------|
| 1 | `01_vector_add.cu`    | kernel, grid/block, cudaMemcpy   |
| 2 | `02_saxpy.cu`         | streaming, bandwidth measurement |
| 3 | `03_reduction.cu`     | shared-mem tree reduce           |
| 4 | `04_transpose.cu`     | coalescing, bank conflicts       |
| 5 | `05_matmul_tiled.cu`  | tiled matmul vs naive            |

Build : `nvcc -O3 -arch=sm_75 <file> -o <exe>`

## TPU/JAX exercises

| # | File                     | Concept                                |
|---|--------------------------|----------------------------------------|
| 1 | `01_jnp_jit.py`          | jnp ops, jit, inspect HLO              |
| 2 | `02_grad_linreg.py`      | value_and_grad, plain SGD              |
| 3 | `03_flax_mnist.py`       | flax.nnx MLP + optax SGD               |
| 4 | `04_vmap.py`             | per-example fn, vmap, nested vmap      |
| 5 | `05_shard_map.py`        | Mesh + PartitionSpec + shard_map       |

Run : `colab exec -s tpu -f <file>`

## MLX exercises

| # | File                       | Concept                                 |
|---|----------------------------|-----------------------------------------|
| 1 | `01_array_ops.py`          | mx.array, lazy ops, mx.eval             |
| 2 | `02_linear_regression.py`  | value_and_grad, manual SGD              |
| 3 | `03_mlp_mnist.py`          | nn.Module MLP + nn.value_and_grad       |
| 4 | `04_attention.py`          | scaled-dot-product + multi-head attn    |
| 5 | `05_llama_inference.py`    | tiny GPT decoder + greedy decode        |

Run : `python3 <file>`

## Projects

### Project A — softmax 4 ways

Row-wise softmax on `(4096, 4096)` implemented in MLX, JAX, raw CUDA, and Triton.
See `projects/A_softmax/bench.md` for cross-stack timings.

## Outputs

Each `NN_*.py` / `NN_*.cu` has a sibling `NN_*.out` containing real captured
stdout from the listed hardware. No mocked results.

## Identity

All commits in this repo are authored as `kitrakrev` via per-command `git -c`
overrides — no global config change on the development host.