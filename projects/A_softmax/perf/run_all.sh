#!/bin/bash
# Project A perf — run all 4 softmax implementations and dump a comparison.
#
# Requires:
#   - Local: MLX (Apple Silicon)
#   - colab session 't4' (--gpu T4) for CUDA + Triton
#   - colab session 'tpu' (--tpu v5e1) for JAX
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT="$ROOT/perf/results.txt"
: > "$OUT"

echo "=== MLX (local) ===" | tee -a "$OUT"
python3 "$ROOT/softmax_mlx.py" | tee -a "$OUT"
echo "" | tee -a "$OUT"

echo "=== JAX / TPU v5e ===" | tee -a "$OUT"
colab exec -s tpu -f "$ROOT/softmax_jax.py" | tee -a "$OUT"
echo "" | tee -a "$OUT"

echo "=== Triton / T4 ===" | tee -a "$OUT"
colab exec -s t4 -f "$ROOT/softmax_triton.py" | tee -a "$OUT"
echo "" | tee -a "$OUT"

echo "=== raw CUDA / T4 ===" | tee -a "$OUT"
B64=$(base64 -i "$ROOT/softmax_cuda.cu")
cat > /tmp/_cu_runner.py <<EOF
import subprocess, base64
src = base64.b64decode("$B64").decode()
open("/tmp/sm.cu","w").write(src)
subprocess.run(["nvcc","-O3","-arch=sm_75","/tmp/sm.cu","-o","/tmp/sm"], check=True)
print(subprocess.run(["/tmp/sm"], capture_output=True, text=True).stdout, end="")
EOF
colab exec -s t4 -f /tmp/_cu_runner.py | tee -a "$OUT"

echo ""
echo "results written to $OUT"
