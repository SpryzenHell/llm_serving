#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="nvcr.io/nvidia/tensorrt-llm/devel:1.3.0rc29"

mkdir -p "$HOME/.cache"
docker pull "$IMAGE"

exec docker run --rm -it     --ipc=host     --ulimit memlock=-1     --ulimit stack=67108864     --gpus=all     --volume "$ROOT:/workspace/llm_serving"     --volume "$HOME/.cache:/root/.cache:rw"     --workdir /workspace/llm_serving     "$IMAGE"     /bin/bash
