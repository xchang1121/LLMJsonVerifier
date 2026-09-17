"""One configuration controls the tokenizer, gateway, and vLLM launcher."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
VLLM_VERSION = "0.29.0"


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class ModelSettings(StrictConfig):
    id: str = "Qwen/Qwen3.8-27B"
    revision: str = MODEL_REVISION
    served_name: str = "llm-json-verifier"
    max_model_len: int = Field(default=131_072, ge=512, le=1_000_000)
    tokenizer_path: str | None = None
    local_files_only: bool = False


class BackendSettings(StrictConfig):
    base_url: str = "http://127.0.0.1:8000"
    max_in_flight: int = Field(default=8, ge=1, le=256)
    timeout_seconds: float = Field(default=300, gt=0)
    queue_timeout_seconds: float = Field(default=30, gt=0)
    logprob_chunk_size: int = Field(default=128, ge=1, le=128)
    verify_on_startup: bool = True

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        from urllib.parse import urlsplit

        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) origin, e.g. http://127.0.0.1:8000")
        if url.path not in {"", "/"} or url.username or url.password:
            raise ValueError("base_url must have no path or embedded credentials; omit /v1")
        return value.rstrip("/")


class ServiceSettings(StrictConfig):
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    max_options: int = Field(default=256, ge=2, le=702)
    max_questions: int = Field(default=32, ge=1, le=128)
    max_request_bytes: int = Field(default=4_194_304, ge=1024)
    max_total_prompt_tokens: int = Field(default=4_194_304, ge=512)
    max_active_requests: int = Field(default=16, ge=1, le=256)
    request_timeout_seconds: float = Field(default=900, gt=0)
    prefix_cache_entries: int = Field(default=8, ge=0, le=1024)
    prefix_cache_tokens: int = Field(default=524_288, ge=0)
    prime_min_prefix_tokens: int = Field(default=4096, ge=0)
    warm_hint_ttl_seconds: float = Field(default=120, ge=0)
    warm_hint_entries: int = Field(default=128, ge=1)


class EngineSettings(StrictConfig):
    tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.90, gt=0, lt=1)
    max_num_batched_tokens: int = Field(default=8192, ge=512)
    max_num_seqs: int = Field(default=16, ge=1)
    dtype: Literal["auto", "bfloat16", "float16"] = "auto"
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"] = "auto"
    mamba_cache_mode: Literal["align", "none"] = "align"
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    async_scheduling: bool = True
    language_model_only: bool = True

    @model_validator(mode="after")
    def validate_cache_mode(self) -> EngineSettings:
        if self.enable_prefix_caching != (self.mamba_cache_mode == "align"):
            raise ValueError("use align with prefix caching; use none when caching is disabled")
        return self


class Settings(StrictConfig):
    model: ModelSettings = Field(default_factory=ModelSettings)
    backend: BackendSettings = Field(default_factory=BackendSettings)
    service: ServiceSettings = Field(default_factory=ServiceSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)


def load_settings(path: str | Path | None = None) -> Settings:
    selected = path or os.environ.get("LLMJV_CONFIG")
    if selected:
        with Path(selected).open("rb") as handle:
            data = tomllib.load(handle)
    else:
        data = {}
    overrides = {
        "LLMJV_BACKEND_URL": ("backend", "base_url"),
        "LLMJV_TOKENIZER_PATH": ("model", "tokenizer_path"),
        "LLMJV_HOST": ("service", "host"),
    }
    for env, (section, key) in overrides.items():
        if env in os.environ:
            data.setdefault(section, {})[key] = os.environ[env]
    return Settings.model_validate(data)
