#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 MODEL"
    exit 2
fi

MODEL="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/dlse_runtime/results/serve_ab"
PID=""

mkdir -p "$OUT"

cleanup() {
    if [ -n "$PID" ]; then
        kill "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

run_case() {
    MODE="$1"
    CONFIG="$2"
    PORT="$3"
    LOG_FILE="$OUT/$MODE"_server.log

    rm -rf "$OUT/$MODE"
    mkdir -p "$OUT/$MODE"

    echo "Starting $MODE server on port $PORT"

    trtllm-serve "$MODEL"         --host 127.0.0.1         --port "$PORT"         --backend pytorch         --config "$CONFIG"         > "$LOG_FILE" 2>&1 &

    PID=$!

    for i in $(seq 1 120); do
        if curl -fsS             "http://127.0.0.1:$PORT/health"             >/dev/null 2>&1; then
            break
        fi

        if ! kill -0 "$PID" 2>/dev/null; then
            echo "$MODE server exited during startup"
            cat "$LOG_FILE"
            exit 1
        fi

        sleep 2
    done

    if ! curl -fsS         "http://127.0.0.1:$PORT/health"         >/dev/null 2>&1; then
        echo "Timed out waiting for $MODE server"
        cat "$LOG_FILE"
        exit 1
    fi

    for concurrency in 1 2 4 8; do
        prompts=$((concurrency * 5))

        echo "Benchmarking $MODE at concurrency $concurrency"

        python3 -m tensorrt_llm.serve.scripts.benchmark_serving             --model "$MODEL"             --host 127.0.0.1             --port "$PORT"             --backend openai             --dataset-name random             --random-input-len 1024             --random-output-len 128             --random-prefix-len 0             --num-prompts "$prompts"             --max-concurrency "$concurrency"             --ignore-eos             --save-result             --result-dir "$OUT/$MODE"             --result-filename "concurrency_$concurrency.json"             --percentile-metrics "ttft,tpot,itl,e2el"
    done

    kill "$PID"
    wait "$PID" 2>/dev/null || true
    PID=""
}

run_case     baseline     "$ROOT/dlse_runtime/configs/dlse-baseline.yaml"     8100

run_case     optimized     "$ROOT/dlse_runtime/configs/dlse-trtllm.yaml"     8200

echo "A/B results written to $OUT"
