"""Concurrent TensorRT-LLM benchmark for DLSE.

The script measures TTFT, inter-token latency and aggregate output throughput.
Run it repeatedly at requests=1,2,4,8 with the same model and workload.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from dlse_trtllm import DLSEConfig, TensorRTLLMBackend


async def measure_request(
    backend: TensorRTLLMBackend,
    prompt: str,
    max_tokens: int,
) -> dict:
    arrivals: list[float] = []
    token_count = 0
    started = time.perf_counter()

    async for output in backend.stream(
        prompt,
        max_tokens=max_tokens,
    ):
        arrivals.append(time.perf_counter())

        try:
            token_count = len(
                output.outputs[0].token_ids
            )
        except (AttributeError, IndexError):
            token_count = max(
                token_count,
                1,
            )

    ended = time.perf_counter()

    inter_token_ms = [
        (b - a) * 1000.0
        for a, b in zip(
            arrivals,
            arrivals[1:],
        )
    ]

    sorted_itl = sorted(inter_token_ms)

    return {
        "ttft_ms": (
            (arrivals[0] - started) * 1000.0
            if arrivals else None
        ),
        "itl_p50_ms": (
            statistics.median(inter_token_ms)
            if inter_token_ms
            else None
        ),
        "itl_p99_ms": (
            sorted_itl[
                max(
                    0,
                    int(
                        0.99 *
                        len(sorted_itl)
                    ) - 1,
                )
            ]
            if sorted_itl
            else None
        ),
        "tokens": token_count,
        "wall_ms": (
            ended - started
        ) * 1000.0,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Explain how continuous batching, "
            "chunked prefill and paged KV caches "
            "interact in an LLM server."
        ),
    )
    parser.add_argument(
        "--output",
        default="results/trtllm_benchmark.json",
    )

    args = parser.parse_args()

    if args.requests <= 0:
        raise SystemExit("--requests must be positive")

    backend = TensorRTLLMBackend(
        DLSEConfig(
            model=args.model,
            max_batch_size=args.requests,
            cuda_graph_batch_sizes=tuple(
                size
                for size in (1, 2, 4, 8)
                if size <= max(args.requests, 8)
            ),
        )
    )

    prompts = [
        f"{args.prompt}\nRequest={i}"
        for i in range(args.requests)
    ]

    started = time.perf_counter()

    results = await asyncio.gather(
        *(
            measure_request(
                backend,
                prompt,
                args.max_tokens,
            )
            for prompt in prompts
        )
    )

    wall_seconds = (
        time.perf_counter() - started
    )

    total_tokens = sum(
        int(row["tokens"])
        for row in results
    )

    valid_ttft = [
        row["ttft_ms"]
        for row in results
        if row["ttft_ms"] is not None
    ]

    valid_itl = [
        row["itl_p50_ms"]
        for row in results
        if row["itl_p50_ms"] is not None
    ]

    payload = {
        "model": args.model,
        "request_count": args.requests,
        "max_tokens": args.max_tokens,
        "total_tokens": total_tokens,
        "wall_seconds": wall_seconds,
        "aggregate_tokens_per_second": (
            total_tokens / wall_seconds
            if wall_seconds
            else None
        ),
        "ttft_p50_ms": (
            statistics.median(valid_ttft)
            if valid_ttft
            else None
        ),
        "itl_p50_ms": (
            statistics.median(valid_itl)
            if valid_itl
            else None
        ),
        "per_request": results,
    }

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
