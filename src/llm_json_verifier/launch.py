"""Build a reviewed vLLM command without embedding secrets in arguments."""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version

from .config import VLLM_VERSION, Settings


def engine_command(settings: Settings) -> list[str]:
    model, engine = settings.model, settings.engine
    args = [
        "vllm",
        "serve",
        model.id,
        "--revision",
        model.revision,
        "--tokenizer-revision",
        model.revision,
        "--served-model-name",
        model.served_name,
        "--host",
        os.environ.get("LLMJV_ENGINE_HOST", "127.0.0.1"),
        "--port",
        "8000",
        "--max-model-len",
        str(model.max_model_len),
        "--tensor-parallel-size",
        str(engine.tensor_parallel_size),
        "--dtype",
        engine.dtype,
        "--gpu-memory-utilization",
        str(engine.gpu_memory_utilization),
        "--max-num-batched-tokens",
        str(engine.max_num_batched_tokens),
        "--max-num-seqs",
        str(engine.max_num_seqs),
        "--kv-cache-dtype",
        engine.kv_cache_dtype,
        "--mamba-cache-mode",
        engine.mamba_cache_mode,
        "--logprobs-mode",
        "raw_logprobs",
        "--max-logprobs",
        "128",
        "--generation-config",
        "vllm",
        "--enable-prompt-tokens-details",
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
    ]
    for flag, value in [
        ("enable-prefix-caching", engine.enable_prefix_caching),
        ("enable-chunked-prefill", engine.enable_chunked_prefill),
        ("async-scheduling", engine.async_scheduling),
        ("language-model-only", engine.language_model_only),
    ]:
        args.append("--" + ("" if value else "no-") + flag)
    return args


def start_engine(settings: Settings) -> int:
    if sys.platform == "win32":
        raise RuntimeError("Run vLLM on Linux with a supported GPU (or WSL2 GPU environment).")
    try:
        installed = version("vllm")
    except PackageNotFoundError as exc:
        raise RuntimeError(f"Install vllm=={VLLM_VERSION} in the engine environment.") from exc
    if installed != VLLM_VERSION:
        raise RuntimeError(
            f"Expected vllm=={VLLM_VERSION}, found {installed}; revalidate before upgrading."
        )
    env = os.environ.copy()
    if key := env.get("LLMJV_VLLM_API_KEY"):
        env["VLLM_API_KEY"] = key
    args = engine_command(settings)
    os.execvpe(args[0], args, env)  # Replace the launcher so container signals reach vLLM.
    return 0
