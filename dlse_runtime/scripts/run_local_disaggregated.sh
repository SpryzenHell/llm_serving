#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 MODEL"
    exit 2
fi

MODEL="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$ROOT/logs/disaggregated"
CTX_PID=""
GEN_PID=""

mkdir -p "$LOG_DIR"

cleanup() {
    status=$?
    trap - EXIT INT TERM

    if [ -n "$CTX_PID" ]; then
        kill "$CTX_PID" 2>/dev/null || true
    fi

    if [ -n "$GEN_PID" ]; then
        kill "$GEN_PID" 2>/dev/null || true
    fi

    wait "$CTX_PID" 2>/dev/null || true
    wait "$GEN_PID" 2>/dev/null || true

    exit "$status"
}

trap cleanup EXIT INT TERM

echo "Starting context server on GPU 0 at 127.0.0.1:8001"
CUDA_VISIBLE_DEVICES=0 trtllm-serve "$MODEL"     --host 127.0.0.1     --port 8001     --backend pytorch     --config "$ROOT/dlse_runtime/configs/dlse-disagg-context.yaml"     > "$LOG_DIR/context.log" 2>&1 &
CTX_PID=$!

echo "Starting generation server on GPU 1 at 127.0.0.1:8002"
CUDA_VISIBLE_DEVICES=1 trtllm-serve "$MODEL"     --host 127.0.0.1     --port 8002     --backend pytorch     --config "$ROOT/dlse_runtime/configs/dlse-disagg-generation.yaml"     > "$LOG_DIR/generation.log" 2>&1 &
GEN_PID=$!

echo "Waiting for context/generation worker health"

for i in $(seq 1 120); do
    if curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1        && curl -fsS http://127.0.0.1:8002/health >/dev/null 2>&1; then
        break
    fi

    if ! kill -0 "$CTX_PID" 2>/dev/null        || ! kill -0 "$GEN_PID" 2>/dev/null; then
        echo "A worker exited during startup."
        echo "Check $LOG_DIR/context.log and $LOG_DIR/generation.log"
        exit 1
    fi

    sleep 2
done

if ! curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1    || ! curl -fsS http://127.0.0.1:8002/health >/dev/null 2>&1; then
    echo "Timed out waiting for context/generation workers."
    echo "Check $LOG_DIR/context.log and $LOG_DIR/generation.log"
    exit 1
fi

echo "Starting disaggregated router on 127.0.0.1:8000"
exec trtllm-serve disaggregated     --config "$ROOT/dlse_runtime/configs/disaggregated-cluster.yaml"
