"""Remote validation tools. These run only when explicitly invoked against a server."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from pathlib import Path

import httpx

from .metrics import classification_metrics, percentile
from .scheduler import gather_cancel_on_error
from .schemas import ClassifyRequest, ClassifyResponse


def read_request(path: str | Path) -> ClassifyRequest:
    return ClassifyRequest.model_validate_json(Path(path).read_text(encoding="utf-8"))


class GatewayClient:
    def __init__(self, url: str, timeout: float = 1200):
        headers = {}
        if key := os.environ.get("LLMJV_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        self.http = httpx.AsyncClient(
            base_url=url.rstrip("/"), headers=headers, timeout=timeout, trust_env=False
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.http.aclose()

    async def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        response = await self.http.post("/v1/classify", json=request.model_dump())
        if response.is_error:
            raise RuntimeError(f"gateway HTTP {response.status_code}: {response.text[:1000]}")
        result = ClassifyResponse.model_validate_json(response.content)
        if [answer.id for answer in result.answers] != [q.id for q in request.questions]:
            raise RuntimeError("gateway response does not cover the requested questions in order")
        for question, answer in zip(request.questions, result.answers, strict=True):
            if set(answer.probabilities) != {option.id for option in question.options}:
                raise RuntimeError("gateway response does not cover exactly the requested options")
        return result


async def benchmark(
    client: GatewayClient,
    request: ClassifyRequest,
    repeats: int,
    concurrency: int,
    warmup: int,
    cache_mode: str,
) -> dict:
    if repeats < 1 or concurrency < 1 or warmup < 0:
        raise ValueError("invalid benchmark repeat/concurrency/warmup count")
    if cache_mode not in {"warm", "cold"}:
        raise ValueError("cache_mode must be warm or cold")
    namespace = "benchmark-" + uuid.uuid4().hex
    base = request.model_copy(update={"cache_namespace": namespace})
    for _ in range(warmup):
        await client.classify(base)
    semaphore = asyncio.Semaphore(concurrency)

    async def once():
        async with semaphore:
            job = (
                base
                if cache_mode == "warm"
                else base.model_copy(update={"cache_namespace": "cold-" + uuid.uuid4().hex})
            )
            started = time.perf_counter()
            result = await client.classify(job)
            return (time.perf_counter() - started) * 1000, result

    started = time.perf_counter()
    runs = await gather_cancel_on_error([once() for _ in range(repeats)])
    wall = time.perf_counter() - started
    latencies = [elapsed for elapsed, _ in runs]
    usages = [result.usage for _, result in runs]
    cached = [usage.backend_cached_prompt_tokens for usage in usages]
    return {
        "mode": cache_mode,
        "repeats": repeats,
        "concurrency": concurrency,
        "warmup": warmup,
        "wall_seconds": wall,
        "requests_per_second": repeats / wall,
        "questions_per_second": repeats * len(request.questions) / wall,
        "latency_ms": {
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "backend_prompt_tokens": sum(u.backend_prompt_tokens for u in usages),
        "backend_cached_prompt_tokens": sum(cached) if all(x is not None for x in cached) else None,
        "backend_completion_tokens": sum(u.backend_completion_tokens for u in usages),
        "note": "Latency excludes client concurrency-queue wait. Cold mode isolates engine cache by salt; CPU tokenization may be warm.",
    }


async def evaluate(client: GatewayClient, path: str | Path, rotate: bool = False) -> dict:
    labeled = []
    stable = comparisons = samples = 0
    dataset = await asyncio.to_thread(Path(path).read_text, encoding="utf-8")
    for line in dataset.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        request = ClassifyRequest.model_validate(record["request"])
        gold = record["expected"]
        if not isinstance(gold, dict) or set(gold) != {q.id for q in request.questions}:
            raise ValueError("expected must map every question ID to exactly one option ID")
        for question in request.questions:
            if gold[question.id] not in {option.id for option in question.options}:
                raise ValueError(f"invalid ground truth for {question.id!r}")
        result = await client.classify(request)
        labeled.extend((answer, gold[answer.id]) for answer in result.answers)
        samples += 1
        if rotate:
            rotated = [
                q.model_copy(update={"options": q.options[1:] + q.options[:1]})
                for q in request.questions
            ]
            other = await client.classify(request.model_copy(update={"questions": rotated}))
            by_id = {a.id: a.selected for a in other.answers}
            comparisons += len(result.answers)
            stable += sum(a.selected == by_id[a.id] for a in result.answers)
    output = {"samples": samples, **classification_metrics(labeled)}
    if rotate:
        output["one_step_rotation_agreement"] = stable / comparisons
    return output


async def verify_cache(client: GatewayClient, request: ClassifyRequest, tolerance: float) -> dict:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    # Fresh salts isolate the *engine* cache without resetting a shared server.
    cold = {}
    cold_cached = []
    for question in request.questions:
        result = await client.classify(
            request.model_copy(
                update={
                    "questions": [question],
                    "execution": "serial",
                    "cache_namespace": "parity-cold-" + uuid.uuid4().hex,
                }
            )
        )
        cold[question.id] = result.answers[0]
        cold_cached.append(result.usage.backend_cached_prompt_tokens)
    shared = request.model_copy(
        update={"execution": "auto", "cache_namespace": "parity-warm-" + uuid.uuid4().hex}
    )
    primed = await client.classify(shared)
    warm = await client.classify(shared)
    delta = max(
        abs(answer.probabilities[key] - cold[answer.id].probabilities[key])
        for result in (primed, warm)
        for answer in result.answers
        for key in answer.probabilities
    )
    agreement = all(
        answer.selected == cold[answer.id].selected
        for result in (primed, warm)
        for answer in result.answers
    )
    observed_hit = (warm.usage.backend_cached_prompt_tokens or 0) > 0
    passed = delta <= tolerance and agreement and observed_hit
    return {
        "passed": passed,
        "max_probability_delta": delta,
        "tolerance": tolerance,
        "winner_agreement": agreement,
        "warm_cache_hit_observed": observed_hit,
        "cold_cached_tokens": cold_cached,
        "primed_usage": primed.usage.model_dump(),
        "warm_usage": warm.usage.model_dump(),
        "note": "A pass requires both numerical agreement and a reported cache hit. This checks the supplied cases only.",
    }
