# Disaggregated deployment

TensorRT-LLM's disaggregated architecture separates the context/prefill and
generation/decode phases onto different GPU workers.

The topology is:

client
  |
  v
trtllm-serve disaggregated :8000
  |                         |
  v                         v
context :8001           generation :8002
(prefill)               (decode)
  |                         ^
  +---- KV cache transfer--+

## 1. Start the context worker

    ./dlse_runtime/scripts/run_context_server.sh MODEL

Run this process on the GPU that should own prompt processing.

## 2. Start the generation worker

    ./dlse_runtime/scripts/run_generation_server.sh MODEL

Run this process on the GPU that should own autoregressive token generation.

## 3. Start the disaggregated router

    ./dlse_runtime/scripts/run_disaggregated_server.sh

The router configuration is in:

    dlse_runtime/configs/disaggregated-cluster.yaml

The example uses context port 8001, generation port 8002 and router port 8000.

## KV transfer

Both workers use:

    cache_transceiver_config:
      backend: NIXL

NIXL is the selected KV-cache transfer backend in this configuration. The
worker configuration is shared so context and generation agree on the transfer
backend and KV-cache settings.

## How this relates to DLSE

The C++ scheduler in dlse_runtime/src/runtime.cpp is the control-plane
research implementation for decode-first continuous batching and chunked
prefill.

The TensorRT-LLM worker topology is the production execution substrate. The
two should not be represented as one implementation.

## Measuring phase isolation

To demonstrate the effect of disaggregation:

1. Run an aggregated TensorRT-LLM serve instance.
2. Run the two-worker disaggregated topology.
3. Send the same prompt/output workload to both.
4. Compare TTFT and p50/p95/p99 ITL.
5. Record GPU memory separately for the context and generation workers.
6. Save the raw metrics output.

Current TensorRT-LLM metrics expose iteration latency, GPU memory and KV-cache
statistics. These are the relevant evidence sources for the performance
claims.
