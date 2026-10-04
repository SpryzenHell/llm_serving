#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/build-host"

cmake -S "$ROOT" -B "$BUILD"     -DCMAKE_BUILD_TYPE=Release     -DDLSE_ENABLE_CUDA=OFF

cmake --build "$BUILD" --parallel
ctest --test-dir "$BUILD" --output-on-failure
"$BUILD/bin/dlse_runtime_bench"
