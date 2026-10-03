# Resume evidence matrix

| Resume point | Repository evidence | Required measurement |
| --- | --- | --- |
| Continuous batching + Chunked Prefill | scheduler + runtime benchmark | target-GPU p99 ITL under mixed prompt lengths |
| 8x throughput | TensorRT-LLM benchmark | identical-workload baseline/optimized ratio >= 8 |
| >60 percent VRAM recovered | paged KV allocator + CUDA A/B | documented KV reservation reduction >= 60 percent |
| <2 microseconds launch | CUDA Graph benchmark | target-GPU timing methodology supporting the wording |

Feature implementation and performance claims are intentionally separate. A
resume number is promoted only after the raw benchmark result and environment
record are checked in.
