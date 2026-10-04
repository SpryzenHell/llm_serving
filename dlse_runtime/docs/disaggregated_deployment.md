# Disaggregated deployment

TensorRT-LLM supports separating the context/prefill and generation/decode
phases onto different worker processes. The context worker handles prompt
processing and produces the KV cache needed by the generation worker.

The repository uses this layout for its two-GPU local example:

```
client
  |
  v
router :8000
  |                 |
  v                 v
context :8001    generation :8002
GPU 0             GPU 1
  |                 ^
  +--- KV transfer-+
```

## 1. Start the workers

The context worker uses the configuration in
`dlse_runtime/configs/dlse-disagg-context.yaml`.

The generation worker uses
`dlse_runtime/configs/dlse-disagg-generation.yaml`.

On a two-GPU host the repository can start both workers and the router with:

    ./dlse_runtime/scripts/run_local_disaggregated.sh MODEL

The script assigns GPU 0 to context and GPU 1 to generation, waits for both
workers to report healthy, then starts the router.

Worker logs are written to:

    logs/disaggregated/

For separate terminals, the individual commands are:

    ./dlse_runtime/scripts/run_context_server.sh MODEL
    ./dlse_runtime/scripts/run_generation_server.sh MODEL
    ./dlse_runtime/scripts/run_disaggregated_server.sh

## 2. KV-cache transfer

Both workers set:

    cache_transceiver_config:
      backend: NIXL

NIXL is the selected cache-transfer backend. The same backend must be present
on both sides of the context/generation transfer.

The repository leaves `max_tokens_in_buffer` unset and uses the runtime default.

## 3. Context worker

The context worker additionally sets:

    disable_overlap_scheduler: true

The TensorRT-LLM disaggregated serving documentation recommends this setting for
context workers.

## 4. Router configuration

The router file is:

    dlse_runtime/configs/disaggregated-cluster.yaml

It lists the context and generation worker URLs and uses round-robin routing for
the context group.

## 5. Readiness and inference

Worker readiness:

    curl -s -o /dev/null -w "context=%{http_code}
"       http://127.0.0.1:8001/health

    curl -s -o /dev/null -w "generation=%{http_code}
"       http://127.0.0.1:8002/health

Router readiness:

    curl -s -o /dev/null -w "router=%{http_code}
"       http://127.0.0.1:8000/health

Run the repository smoke test against the router:

    ./dlse_runtime/scripts/smoke_test.sh MODEL

## 6. Measuring the benefit of disaggregation

Run the same model and workload first with the aggregated server and then with
the disaggregated topology.

Record:
- TTFT;
- inter-token latency;
- output tokens/s;
- context-worker GPU memory;
- generation-worker GPU memory;
- KV-cache statistics from `/metrics`.

Keep the model revision, quantization, GPU assignment, TensorRT-LLM version and
runtime settings fixed between runs.

The repository does not include fabricated GPU numbers. The measurements should
be captured on the target hardware and committed as raw benchmark artifacts.
