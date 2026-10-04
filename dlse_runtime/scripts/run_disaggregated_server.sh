#!/usr/bin/env bash
set -euo pipefail

exec trtllm-serve disaggregated     --config dlse_runtime/configs/disaggregated-cluster.yaml
