#!/usr/bin/env bash
set -euo pipefail

OUT="dlse_runtime/results"
mkdir -p "$OUT"

curl --fail --silent     http://127.0.0.1:8000/metrics     > "$OUT/server_metrics.json"

echo "Saved $OUT/server_metrics.json"
