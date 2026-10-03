"""Optional TensorRT-LLM backend for DLSE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

try:
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import (
        CudaGraphConfig,
        KvCacheConfig,
    )
except ImportError as exc:
    LLM = None
    SamplingParams = None
    CudaGraphConfig = None
    KvCacheConfig = None
    _IMPORT_ERROR: Optional[Exception] = exc
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class DLSEConfig:
    model: str
    max_batch_size: int = 8
    max_seq_len: int = 4096
    max_num_tokens: int = 8192
    kv_free_fraction: float = 0.70
    cuda_graph_batch_sizes: tuple[int, ...] = (
        1, 2, 4, 8
    )


class TensorRTLLMBackend:
    """TensorRT-LLM execution adapter for the DLSE control plane."""

    def __init__(self, config: DLSEConfig):
        if LLM is None:
            raise RuntimeError(
                "TensorRT-LLM is not installed. "
                "Install dlse_runtime/requirements-trtllm.txt "
                "inside the target CUDA environment."
            ) from _IMPORT_ERROR

        graph_config = CudaGraphConfig(
            batch_sizes=list(
                config.cuda_graph_batch_sizes
            ),
            enable_padding=True,
            mode="decode",
        )

        kv_config = KvCacheConfig(
            enable_block_reuse=True,
            free_gpu_memory_fraction=(
                config.kv_free_fraction
            ),
        )

        self._llm = LLM(
            model=config.model,
            max_batch_size=config.max_batch_size,
            max_seq_len=config.max_seq_len,
            max_num_tokens=config.max_num_tokens,
            enable_chunked_prefill=True,
            cuda_graph_config=graph_config,
            kv_cache_config=kv_config,
        )

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 64,
    ) -> Any:
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
        )

        return self._llm.generate(
            prompts,
            params,
            use_tqdm=False,
        )

    async def stream(
        self,
        prompt: str,
        max_tokens: int = 64,
    ) -> AsyncIterator[Any]:
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
        )

        async for output in self._llm.generate_async(
            prompt,
            params,
            streaming=True,
        ):
            yield output

    def kv_cache_capacity(self) -> Optional[Any]:
        executor = getattr(
            self._llm,
            "llm",
            None,
        )

        if executor is None:
            return None

        getter = getattr(
            executor,
            "get_kv_cache_capacity",
            None,
        )

        return (
            getter()
            if getter is not None
            else None
        )
