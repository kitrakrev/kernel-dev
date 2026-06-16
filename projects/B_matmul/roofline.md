# Project B — matmul ladder + roofline

Square matmul `(2048, 2048) x (2048, 2048)`. Each rung climbs toward peak.

## T4 ladder (fp32 -> Tensor Core fp16)

| Rung | Backend            | Precision | per-call (ms) | TFLOPS | % of peak |
|-----:|--------------------|-----------|--------------:|-------:|----------:|
|   1  | naive CUDA         | fp32      |        47.689 |  0.360 | 4.4% (fp32 8.1 TFLOPS) |
|   2  | tiled CUDA (32x32) | fp32      |        21.217 |  0.810 | 10.0%                  |
|   3  | cuBLAS HGEMM (TC)  | fp16      |         0.693 | 24.794 | 38.1% (fp16 65 TFLOPS) |

Naive -> tiled: **2.25x** speedup (shared-mem reuse).
Tiled -> Tensor Core: **30.6x** speedup (hardware mma + fp16).
End-to-end ladder: **~69x** total.

## Cross-stack peak utilization

| Backend         | Hardware        | Precision | TFLOPS | Peak     | % of peak |
|-----------------|-----------------|-----------|-------:|---------:|----------:|
| MLX (M4 GPU)    | Apple M4        | fp32      |  1.591 |  ~4.6    | 34.6%     |
| MLX (M4 GPU)    | Apple M4        | fp16      |  2.057 |  ~9      | 22.9%     |
| naive CUDA      | T4              | fp32      |  0.360 |   8.1    |  4.4%     |
| tiled CUDA      | T4              | fp32      |  0.810 |   8.1    | 10.0%     |
| cuBLAS HGEMM    | T4 Tensor Core  | fp16      | 24.794 |  65      | 38.1%     |
| JAX MXU         | TPU v5e         | bf16      | 54.409 | ~197     | 27.6%     |

## Roofline interpretation

Arithmetic intensity of N=2048 matmul = 2N flops / (3*N^2 / N^2) bytes ≈ N/6 fp32
= **~341 flops/byte** for fp32, **~683** for fp16.

- T4 HBM BW = 320 GB/s. Compute crosses BW limit at intensity 8.1e12/320e9 = ~25
  for fp32, ~203 for fp16. At intensity 341/683 we're firmly compute-bound.
- Naive matmul has terrible *effective* intensity because of repeated global
  loads -> memory-bound in practice. Tiling fixes this by reusing TILE x TILE
  blocks from shared mem.
- Tensor Cores raise the compute ceiling 8x (8.1 -> 65 TFLOPS fp16 on T4).
- TPU MXU pushes ceiling another ~3x over T4 fp16; v5e is engineered for matmul.

## Why not 100% of peak?

- **Naive (4.4%)**: each output element does 2N global loads -> bandwidth bound.
- **Tiled (10%)**: shared-mem helps, but still no fma pipelining, no async copy.
- **cuBLAS TC (38%)**: small N (2048) -> insufficient work to amortize launch +
  wave quantization. Closer to peak at N=4096+.
- **JAX MXU (28%)**: launch overhead on a single 0.3 ms call dominates;
  large-batch jit-fused workloads (transformer fwd pass) typically see 60-80%.
- **MLX (35%)**: Metal command-buffer overhead per matmul; persistent batches
  get higher.

## Next rungs (not implemented)

- CUDA: hand-written wmma (`cuda::wmma`) — matches cuBLAS within 10-20%.
- TPU: pallas custom kernel — can push to 50-60% on small shapes.
- MLX: `mx.fast.metal_kernel` custom Metal — fp16 simdgroup_matrix ops.
