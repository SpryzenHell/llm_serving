#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build-cuda"

ARCH=86

cmake -S "$ROOT" -B "$BUILD"     -DCMAKE_BUILD_TYPE=Release     -DDLSE_ENABLE_CUDA=ON     -DCMAKE_CUDA_ARCHITECTURES="$ARCH"

cmake --build "$BUILD" --parallel
"$BUILD/dlse_paged_attention_bench"
"$BUILD/dlse_cuda_graph_bench" 10000
