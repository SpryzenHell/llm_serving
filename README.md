# Disaggregated LLM Serving Engine

This repository contains a systems-oriented LLM serving runtime built around
continuous batching, chunked prefill, paged KV-cache management, CUDA Graph
decode replay, and an optional TensorRT-LLM execution backend.

The project was originally assembled from three upstream repositories named in
the project manifest:

- karpathy/llama2.c — compact C LLaMA inference baseline
- hkproj/pytorch-llama — PyTorch LLaMA model and contiguous KV-cache reference
- Bruce-Lee-LY/cuda_hgemm — CUDA WMMA/MMA and HGEMM optimization reference

The original merged snapshot remains in this repository for provenance. The
new serving implementation is under dlse_runtime/ and is deliberately written
as a clean project layer instead of relying on the mechanically renamed legacy
files.

## Architecture

Requests enter a continuous-batching scheduler.

The scheduler gives already-running decode requests one token of service before
using remaining token budget for prompt work. Long prompts are divided into
fixed-size chunks and rotated through the waiting queue.

The execution path is:

Request queue
  -> ContinuousBatchScheduler
  -> IterationPlan
  -> prefill chunks / decode tokens
  -> PagedKVCache
  -> CUDA page-table attention
  -> TensorRT-LLM backend

The CUDA Graph path captures the stable autoregressive decode step and replays
it without rebuilding the individual kernel launch sequence.

## Resume bullet mapping

### Continuous batching + Chunked Prefill

Implemented in dlse_runtime/src/runtime.cpp and exercised by
dlse_runtime/benchmarks/runtime_bench.cpp.

The benchmark injects a 4096-token prompt while eight decode streams are active
and reports the maximum observed decode-iteration gap.

This proves the scheduling policy. End-to-end ITL still has to be measured on
the target GPU with a real model.

### PagedAttention + KV cache

Implemented in dlse_runtime/src/runtime.cpp and
dlse_runtime/cuda/paged_attention.cu.

The page allocator grows sequences on demand and exposes logical token count,
reserved token capacity, physical pages and internal fragmentation.

The CUDA benchmark compares the paged implementation against a contiguous KV
reference on identical inputs and fails if numerical error exceeds the stated
FP16 tolerance.

The memory benchmark compares KV-cache reservation, not total process VRAM.
The resume percentage must therefore be promoted only after the complete GPU
measurement is performed on the target workload.

### CUDA Graph autoregressive loop

Implemented in dlse_runtime/benchmarks/cuda_graph_bench.cu.

The benchmark uses a device-resident state and step counter so the graph replay
preserves the same dependency chain as the baseline kernel loop.

It reports host enqueue time per decode step. That value is intentionally not
described as GPU execution time or end-to-end ITL.

## Build and validate the host control plane

Run:

    cmake -S . -B build
    cmake --build build --parallel
    ctest --test-dir build --output-on-failure
    ./build/dlse_runtime_bench

Or use:

    ./dlse_runtime/scripts/run_host_validation.sh

The host path is useful even without CUDA because it proves scheduler and page
allocator semantics independently from GPU-specific integration.

## Build and validate CUDA

Run this on a machine with a compatible NVIDIA driver and CUDA toolkit:

    cmake -S . -B build-cuda       -DDLSE_ENABLE_CUDA=ON       -DCMAKE_CUDA_ARCHITECTURES=86

    cmake --build build-cuda --parallel

    ./build-cuda/dlse_paged_attention_bench
    ./build-cuda/dlse_cuda_graph_bench 10000

Use a CUDA architecture that matches the actual machine. For example, Ampere
RTX 3090/A6000 uses 86 and A100 uses 80.

The paged-attention benchmark reports both numerical error and CUDA-event
timings. The graph benchmark reports baseline and graph host enqueue overhead.

## TensorRT-LLM backend

Install the optional dependency inside the target NVIDIA environment:

    python -m pip install -r dlse_runtime/requirements-trtllm.txt

Check initialization:

    PYTHONPATH=dlse_runtime/python     python dlse_runtime/python/verify_trtllm.py       --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

Run concurrent streaming measurements:

    PYTHONPATH=dlse_runtime/python     python dlse_runtime/benchmarks/trtllm_benchmark.py       --model TinyLlama/TinyLlama-1.1B-Chat-v1.0       --requests 8       --max-tokens 64

The benchmark records request-level TTFT, inter-token latency and aggregate
generated tokens per second.

The configuration in dlse_runtime/configs/dlse-trtllm.yaml enables chunked
prefill, paged KV cache settings, block reuse and padded CUDA Graph batch
sizes.

For a service deployment, the current TensorRT-LLM CLI can be used with this
configuration:

    trtllm-serve MODEL --config dlse_runtime/configs/dlse-trtllm.yaml

The installed TensorRT-LLM version should be recorded in the benchmark output
alongside the model revision and GPU information.

## Benchmark discipline

The three numeric resume claims are deliberately treated as measured targets:

- 8x throughput: optimized tokens/s divided by the documented baseline
- >60% VRAM recovery: KV-cache reservation reduction against the explicit
  contiguous baseline
- <2 us launch: a precisely defined launch/enqueue measurement on the target
  GPU

The repository does not hard-code these numbers into the implementation.
Raw benchmark results and the environment used to obtain them should be kept
with the release.

See:
- dlse_runtime/docs/architecture.md
- dlse_runtime/docs/benchmark_protocol.md
- dlse_runtime/docs/evidence_matrix.md
- dlse_runtime/third_party/SOURCES.md

## Source revisions used

karpathy/llama2.c
Revision: 350e04fe35433e6d2941dce5a1f53308f87058eb

hkproj/pytorch-llama
Revision: 067f8a37fe36ac8b52dca9cc6f2a2e8d6aa372d6

Bruce-Lee-LY/cuda_hgemm
Revision: 0d26c2e4415ab0d0af5a6bfae301275c608d46b4

## Important provenance note

The supplied merge script is designed to create a synthetic historical Git
timeline by setting GIT_AUTHOR_DATE and GIT_COMMITTER_DATE to configured dates.
That mechanism is not part of the new runtime and should not be used as
evidence of historical implementation work.

This branch uses ordinary Git commits and explicitly separates upstream source
from DLSE-specific additions.
