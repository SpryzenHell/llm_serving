#!/usr/bin/env bash
set -euo pipefail

echo "timestamp=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "kernel=$(uname -srmo)"
echo "compiler=$(c++ --version | head -1)"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "nvidia_smi=unavailable"
fi

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -1
else
    echo "nvcc=unavailable"
fi

python3 - <<'PY'
import importlib.util

for name in ("torch", "tensorrt_llm"):
    print(
        f"{name}_installed="
        f"{importlib.util.find_spec(name) is not None}"
    )
PY
