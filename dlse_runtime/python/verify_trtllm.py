import argparse

from dlse_trtllm import DLSEConfig, TensorRTLLMBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
    )

    args = parser.parse_args()

    backend = TensorRTLLMBackend(
        DLSEConfig(model=args.model)
    )

    print("TensorRT-LLM backend initialized")
    print(
        "KV cache capacity:",
        backend.kv_cache_capacity(),
    )


if __name__ == "__main__":
    main()
