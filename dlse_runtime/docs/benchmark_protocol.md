# Benchmark protocol

## Scheduler test

Workload:
- eight active decode requests;
- one 4096-token prompt admitted afterwards.

Measure maximum decode-iteration gap, prompt tokens scheduled and active/waiting
request counts.

This is a scheduler metric, not end-to-end GPU ITL.

## KV-cache test

For identical sequence lengths compare:
- contiguous reservation at batch times max_seq_len;
- paged reservation using ceil(sequence_len / page_size).

Measure reserved KV bytes and internal fragmentation.

Then run the CUDA A/B benchmark to validate numerical equivalence.

## Throughput and ITL

Run the TensorRT-LLM benchmark at request counts 1, 2, 4 and 8.

Keep fixed:
- model checkpoint/revision;
- tokenizer;
- prompt distribution;
- maximum generated tokens;
- GPU;
- TensorRT-LLM version;
- quantization;
- parallelism settings.

Report aggregate tokens/s, TTFT p50/p95/p99, ITL p50/p95/p99, peak GPU memory
and KV-cache statistics.

## Testing the 8x statement

Compute optimized tokens/s divided by baseline tokens/s.

Do not copy 8x into the resume unless the experiment actually produces the
claimed ratio.

## Testing the >60 percent memory statement

Define wasted memory as the difference between the documented contiguous KV
reservation and paged KV reservation for the same workload.

Do not convert a KV reservation reduction into a total-process VRAM claim
without controlling model weights and other allocations.

## Testing the <2 microseconds statement

cuda_graph_bench reports host enqueue time per step.

For a release-quality claim also capture GPU execution time, graph-instantiation
time, warm-up policy, CUDA version, driver version and GPU model.

The resume wording must match the exact measurement.
