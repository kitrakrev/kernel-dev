# kernel-dev

Hands-on accelerator kernels and training primitives across multiple stacks.

## Layout

```
MLX/    Apple Silicon exercises (local M4)
CUDA/   NVIDIA exercises (Colab T4)
TPU/    JAX/TPU exercises (Colab v5e)
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

