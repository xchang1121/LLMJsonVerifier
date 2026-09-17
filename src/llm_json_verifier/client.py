"""Remote validation tools. These run only when explicitly invoked against a server."""

from __future__ import annotations

import asyncio
import math
import os
import random
import time
import uuid
from pathlib import Path

import httpx

from .datasets import parse_dataset
from .metrics import classification_metrics
from .runlog import RunLog, digest, json_digest, observe, outcome_summary, run_workers
from .schemas import ClassifyRequest, ClassifyResponse


def read_request(path: str | Path) -> ClassifyRequest:
    return ClassifyRequest.model_validate_json(Path(path).read_text(encoding="utf-8"))


class GatewayHTTPError(RuntimeError):
    def __init__(self, status: int):
        self.http_status = status
        super().__init__(f"gateway HTTP {status}")


class GatewayClient:
    def __init__(self, url: str, timeout: float = 1200):
        self.url = url.rstrip("/")
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
            raise GatewayHTTPError(response.status_code)
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
    records: str | Path | None = None,
) -> dict:
    if repeats < 1 or not 1 <= concurrency <= 256 or warmup < 0:
        raise ValueError("invalid benchmark repeat/concurrency/warmup count")
    if cache_mode not in {"warm", "cold"}:
        raise ValueError("cache_mode must be warm or cold")
    namespace = "benchmark-" + uuid.uuid4().hex
    base = request.model_copy(update={"cache_namespace": namespace})
    log = RunLog(
        "benchmark",
        {
            "url": getattr(client, "url", None),
            "request_sha256": json_digest(request.model_dump()),
            "mode": cache_mode,
            "repeats": repeats,
            "concurrency": concurrency,
            "warmup": warmup,
        },
        records,
    )
    try:
        warmups = [await observe(client, base, log, phase="warmup", index=i) for i in range(warmup)]

        async def once(index, _):
            job = (
                base
                if cache_mode == "warm"
                else base.model_copy(update={"cache_namespace": "cold-" + uuid.uuid4().hex})
            )
            return await observe(client, job, log, phase="measured", index=index)

        started = time.perf_counter()
        runs = await run_workers(range(repeats), min(concurrency, repeats), once)
        wall = time.perf_counter() - started
        outcomes = outcome_summary([event for event, _ in runs])
        usages = [result.usage for _, result in runs if result is not None]
        cached = [usage.backend_cached_prompt_tokens for usage in usages]
        return log.finish(
            {
                "mode": cache_mode,
                "repeats": repeats,
                "concurrency": concurrency,
                "warmup": warmup,
                "warmup_outcomes": outcome_summary([event for event, _ in warmups]),
                "wall_seconds": wall,
                **outcomes,
                "requests_per_second": outcomes["succeeded"] / wall,
                "questions_per_second": outcomes["succeeded"] * len(request.questions) / wall,
                "backend_prompt_tokens": sum(u.backend_prompt_tokens for u in usages),
                "backend_cached_prompt_tokens": sum(cached)
                if cached and all(x is not None for x in cached)
                else None,
                "backend_completion_tokens": sum(u.backend_completion_tokens for u in usages),
            }
        )
    finally:
        log.close()


async def evaluate(
    client: GatewayClient,
    path: str | Path,
    rotate: bool = False,
    records: str | Path | None = None,
    concurrency: int = 1,
    seed: int | None = None,
) -> dict:
    if not 1 <= concurrency <= 256:
        raise ValueError("concurrency must be between 1 and 256")
    dataset = await asyncio.to_thread(Path(path).read_bytes)
    cases = parse_dataset(dataset)  # Validate the entire file before spending any inference.
    if seed is not None:
        random.Random(seed).shuffle(cases)
    log = RunLog(
        "evaluate",
        {
            "url": getattr(client, "url", None),
            "dataset_sha256": digest(dataset),
            "dataset": str(path),
            "case_order": [case.id for case in cases],
            "rotate": rotate,
            "concurrency": concurrency,
            "seed": seed,
        },
        records,
    )
    try:

        async def once(index, case):
            metadata = {
                "index": index,
                "case_id": case.id,
                "tags": case.tags,
                "expected": case.expected,
            }
            primary = await observe(client, case.request, log, phase="primary", **metadata)
            rotation = None
            if rotate and primary[1] is not None:
                rotated = [
                    q.model_copy(update={"options": q.options[1:] + q.options[:1]})
                    for q in case.request.questions
                ]
                rotation = await observe(
                    client,
                    case.request.model_copy(update={"questions": rotated}),
                    log,
                    phase="rotation",
                    **metadata,
                )
            return primary, rotation

        started = time.perf_counter()
        runs = await run_workers(cases, min(concurrency, len(cases)), once)
        wall = time.perf_counter() - started
        labeled, mismatches, rotations = [], [], []
        correct_samples = stable = comparisons = 0
        for case, (primary, rotation) in zip(cases, runs, strict=True):
            result = primary[1]
            if result is None:
                continue
            labeled.extend((a, case.expected[a.id]) for a in result.answers)
            errors = [
                {
                    "case_id": case.id,
                    "question_id": a.id,
                    "expected": case.expected[a.id],
                    "selected": a.selected,
                }
                for a in result.answers
                if a.selected != case.expected[a.id]
            ]
            mismatches.extend(errors)
            correct_samples += not errors
            if rotation is not None:
                rotations.append(rotation[0])
                if rotation[1] is not None:
                    by_id = {a.id: a.selected for a in rotation[1].answers}
                    comparisons += len(result.answers)
                    stable += sum(a.selected == by_id[a.id] for a in result.answers)
        requested = sum(len(case.request.questions) for case in cases)
        metrics = (
            classification_metrics(labeled)
            if labeled
            else {
                "questions": 0,
                "accuracy": None,
                "nll": None,
                "brier": None,
                "ece": None,
            }
        )
        output = {
            "samples": len(cases),
            "wall_seconds": wall,
            **outcome_summary([primary[0] for primary, _ in runs]),
            **metrics,
            "requested_questions": requested,
            "question_coverage": len(labeled) / requested,
            "end_to_end_accuracy": sum(a.selected == gold for a, gold in labeled) / requested,
            "exact_match_rate": correct_samples / len(cases),
            "mismatches": mismatches,
        }
        if rotate:
            output.update(
                rotation_outcomes=outcome_summary(rotations),
                rotation_compared_questions=comparisons,
                rotation_question_coverage=comparisons / requested,
                one_step_rotation_agreement=stable / comparisons if comparisons else None,
            )
        return log.finish(output)
    finally:
        log.close()


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
