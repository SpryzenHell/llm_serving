#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 MODEL"
    exit 2
fi

MODEL="$1"
OUT="dlse_runtime/results"
mkdir -p "$OUT"

./dlse_runtime/scripts/record_environment.sh     > "$OUT/environment.txt"

for BATCH in 1 2 4 8; do
    PYTHONPATH=dlse_runtime/python     python3 dlse_runtime/benchmarks/trtllm_benchmark.py         --model "$MODEL"         --requests "$BATCH"         --max-tokens 64         --output "$OUT/trtllm_batch_$BATCH.json"
done

echo "Benchmark matrix written under $OUT"
