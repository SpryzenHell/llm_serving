#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 MODEL"
    exit 2
fi

MODEL="$1"

CUDA_VISIBLE_DEVICES=1 exec trtllm-serve "$MODEL"     --host 127.0.0.1     --port 8002     --backend pytorch     --config dlse_runtime/configs/dlse-disagg-generation.yaml
