#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 MODEL [PORT]"
    exit 2
fi

MODEL="$1"
PORT=8000

if [ "$#" -eq 2 ]; then
    PORT="$2"
fi

BASE_URL="http://127.0.0.1:$PORT"

echo "Checking $BASE_URL/health"
curl --fail --silent --show-error     "$BASE_URL/health"

echo
echo "Checking chat completion endpoint"

curl --fail --silent --show-error     "$BASE_URL/v1/chat/completions"     -H "Content-Type: application/json"     -d "{
        "model": "$MODEL",
        "messages": [
            {
                "role": "user",
                "content": "Give a one-sentence explanation of paged KV caches."
            }
        ],
        "max_tokens": 32,
        "temperature": 0
    }"

echo
echo "Smoke test passed"
