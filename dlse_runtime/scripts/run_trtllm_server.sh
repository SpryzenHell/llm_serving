#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 MODEL"
    exit 2
fi

MODEL="$1"

exec trtllm-serve "$MODEL"     --host 0.0.0.0     --port 8000     --backend pytorch     --config dlse_runtime/configs/dlse-trtllm.yaml
