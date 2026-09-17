"""vLLM 0.29.0 HTTP adapter: explicit token IDs, never natural top-k fallback."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Any

import httpx

from .admission import AdmissionGate
from .config import Settings
from .errors import BackendBusy, BackendError, BackendProtocolError, BackendTimeout


@dataclass(frozen=True)
class ScoreResult:
    logprobs: dict[int, float]
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int | None
    queue_wait_ms: float = 0.0


class VLLMBackend:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.admission = AdmissionGate(
            settings.backend.max_in_flight,
            settings.backend.max_queued_scores,
            settings.backend.queue_timeout_seconds,
            "scoring",
        )
        self._owns_client = client is None
        headers = {}
        if key := os.environ.get("LLMJV_VLLM_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        self.client = client or httpx.AsyncClient(
            base_url=settings.backend.base_url,
            headers=headers,
            timeout=httpx.Timeout(settings.backend.timeout_seconds, connect=10),
            limits=httpx.Limits(
                max_connections=settings.backend.max_in_flight + 2,
                max_keepalive_connections=settings.backend.max_in_flight + 2,
            ),
            trust_env=False,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await self.client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise BackendTimeout("vLLM request timed out") from exc
        except httpx.RequestError as exc:
            raise BackendError("cannot connect to vLLM; check its URL and readiness") from exc
        if response.status_code in {429, 503}:
            raise BackendBusy("vLLM is busy; retry with lower concurrency")
        if response.is_error:
            # Never reflect upstream bodies, which may contain document text.
            raise BackendError(
                f"vLLM returned HTTP {response.status_code}; use the pinned launcher and "
                "check model length, revision, and logprob_token_ids support"
            )
        if method == "GET" and path == "/health" and not response.content:
            return {}
        try:
            data = response.json()
        except ValueError as exc:
            raise BackendProtocolError("vLLM returned non-JSON data") from exc
        if not isinstance(data, dict) or "error" in data:
            raise BackendProtocolError("vLLM returned an unexpected response envelope")
        return data

    async def check_tokenizer(self, text: str, expected: tuple[int, ...]) -> None:
        models = await self._request("GET", "/v1/models")
        if self.settings.model.served_name not in {
            item.get("id") for item in models.get("data", []) if isinstance(item, dict)
        }:
            raise BackendProtocolError("configured served_name is not present in vLLM /v1/models")
        data = await self._request(
            "POST",
            "/tokenize",
            json={
                "model": self.settings.model.served_name,
                "prompt": text,
                "add_special_tokens": False,
            },
        )
        if data.get("tokens") != list(expected):
            raise BackendProtocolError("gateway and vLLM tokenizer disagree; pin the same revision")

    async def score(
        self, prompt_ids: tuple[int, ...], candidate_ids: tuple[int, ...], cache_salt: str
    ) -> ScoreResult:
        if not 1 <= len(candidate_ids) <= 128 or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("one vLLM scoring call requires 1..128 unique token IDs")
        async with self.admission.slot() as waited_ms:
            # Keep queued work as references to shared immutable token tuples;
            # materialize the large HTTP payload only after admission.
            payload = {
                "model": self.settings.model.served_name,
                "prompt": list(prompt_ids),
                "max_tokens": 1,
                "n": 1,
                "stream": False,
                "echo": False,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "seed": 0,
                "ignore_eos": True,
                "skip_special_tokens": False,
                "add_special_tokens": False,
                "logprobs": len(candidate_ids),
                "logprob_token_ids": list(candidate_ids),
                "return_tokens_as_token_ids": True,
                "cache_salt": cache_salt,
            }
            data = await self._request("POST", "/v1/completions", json=payload)
        return replace(
            parse_scores(data, candidate_ids, expected_prompt_tokens=len(prompt_ids)),
            queue_wait_ms=waited_ms,
        )


def parse_scores(
    data: dict[str, Any], candidate_ids: tuple[int, ...], expected_prompt_tokens: int
) -> ScoreResult:
    try:
        choices = data["choices"]
        if len(choices) != 1 or choices[0]["finish_reason"] != "length":
            raise ValueError("expected exactly one completed scoring token")
        lp = choices[0]["logprobs"]
        rows = lp["top_logprobs"]
        if len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError("expected one token logprob row")
        selected = {}
        for token_id in candidate_ids:
            value = rows[0][f"token_id:{token_id}"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("non-numeric candidate score")
            # vLLM serializes -inf as -9999; do not turn masked/missing data into
            # plausible confidence. With unmodified logits, valid scores are finite.
            if not math.isfinite(value) or value <= -9999 or value > 1e-5:
                raise ValueError("invalid raw log-probability")
            selected[token_id] = float(value)
        usage = data["usage"]
        prompt = usage["prompt_tokens"]
        completion = usage["completion_tokens"]
        if (
            type(prompt) is not int
            or prompt != expected_prompt_tokens
            or type(completion) is not int
            or completion != 1
        ):
            raise ValueError("unexpected token accounting")
        detail = usage.get("prompt_tokens_details") or {}
        cached = detail.get("cached_tokens")
        if cached is not None and (type(cached) is not int or not 0 <= cached <= prompt):
            raise ValueError("invalid cached token count")
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise BackendProtocolError(
            "vLLM did not return a complete, valid candidate score set; no probabilities were fabricated"
        ) from exc
    return ScoreResult(selected, prompt, completion, cached)
