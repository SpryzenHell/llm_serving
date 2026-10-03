#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 MODEL"
    exit 2
fi

MODEL="$1"

exec trtllm-serve "$MODEL"     --config dlse_runtime/configs/dlse-trtllm.yaml
