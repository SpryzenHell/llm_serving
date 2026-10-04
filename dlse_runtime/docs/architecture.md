# Architecture

## Control plane

ContinuousBatchScheduler tracks waiting prefill work and active decode work.

Each iteration:
1. services active decode requests first, one token per request;
2. spends the remaining token budget on prompt chunks;
3. rotates incomplete prefills back into the waiting queue;
4. activates a request for decode after its prompt is complete.

This is the mechanism behind the bounded-sharing part of the resume statement.

## Memory plane

PagedKVCache separates logical sequence length from physical storage.

A sequence with N logical tokens owns ceil(N / page_size) physical pages. New
pages are allocated only when the logical sequence crosses a page boundary.

The allocator exposes logical tokens, reserved token capacity, used/free pages,
KV bytes reserved and internal fragmentation.

## CUDA execution plane

The CUDA layer provides two comparable attention paths:
- contiguous KV storage;
- page-table KV storage.

The benchmark feeds identical tensors into both paths and checks maximum absolute
output difference before reporting timing.

The current CUDA attention implementation is correctness-first. It is not
represented as the final production-fused attention kernel.

## CUDA Graph path

cuda_graph_bench contains a stateful one-dimensional autoregressive step.

The decode step reads a device-resident token counter, updates device state, and
increments that counter. The same step is measured through ordinary host
launches and captured-graph replay.

## TensorRT-LLM path

The Python backend configures:
- chunked prefill;
- paged KV cache;
- block reuse;
- CUDA Graph batch-size buckets;
- bounded batch/token limits.

Concurrent async generation is used for the throughput and ITL benchmark.

## Disaggregated deployment

The DLSE project separates scheduling/control concepts from the vendor serving
substrate. For true multi-process or multi-GPU disaggregated serving, use the
current TensorRT-LLM disaggregated server mode and document the server roles and
metadata configuration used on the cluster.

The local scheduler is not described as a networked disaggregated cluster.
