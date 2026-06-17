#!/bin/bash
# Project B perf — run the full matmul ladder and dump comparison.
#
# Requires:
#   - Local MLX for step 4
#   - colab 't4' for steps 1/2/3
#   - colab 'tpu' for step 5
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT="$ROOT/perf/results.txt"
: > "$OUT"

run_cu() {
    local file="$1"
    local name=$(basename "$file" .cu)
    local b64=$(base64 -i "$file")
    cat > /tmp/_cu_runner.py <<EOF
import subprocess, base64
src = base64.b64decode("$b64").decode()
open("/tmp/$name.cu","w").write(src)
r = subprocess.run(["nvcc","-O3","-arch=sm_75","/tmp/$name.cu","-o","/tmp/$name"], capture_output=True, text=True)
if r.returncode != 0:
    print("COMPILE ERR:", r.stderr); raise SystemExit(1)
print(subprocess.run(["/tmp/$name"], capture_output=True, text=True).stdout, end="")
EOF
    colab exec -s t4 -f /tmp/_cu_runner.py
}

echo "=== rung 1: naive CUDA (T4) ===" | tee -a "$OUT"
run_cu "$ROOT/01_naive_cuda.cu" | tee -a "$OUT"
echo "" | tee -a "$OUT"

echo "=== rung 2: tiled CUDA (T4) ===" | tee -a "$OUT"
run_cu "$ROOT/02_tiled_cuda.cu" | tee -a "$OUT"
echo "" | tee -a "$OUT"

echo "=== rung 3: cuBLAS HGEMM Tensor Core (T4) ===" | tee -a "$OUT"
colab exec -s t4 -f "$ROOT/03_tensorcore_cublas.py" | tee -a "$OUT"
echo "" | tee -a "$OUT"

echo "=== rung 4: MLX (Apple M4, local) ===" | tee -a "$OUT"
python3 "$ROOT/04_matmul_mlx.py" | tee -a "$OUT"
echo "" | tee -a "$OUT"

echo "=== rung 5: JAX MXU (TPU v5e) ===" | tee -a "$OUT"
colab exec -s tpu -f "$ROOT/05_matmul_jax_mxu.py" | tee -a "$OUT"

echo ""
echo "results written to $OUT"
