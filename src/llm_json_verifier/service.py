"""Compile, schedule and assemble classifications without parsing generated text."""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass

from .backend import ScoreResult, VLLMBackend
from .config import Settings
from .errors import BackendBusy, BackendProtocolError, BackendTimeout
from .probabilities import make_answer
from .prompts import PreparedQuestion, PromptCompiler
from .scheduler import PrefixPrimer, gather_cancel_on_error
from .schemas import ClassifyRequest, ClassifyResponse, Option, Question, Timing, Usage


@dataclass(frozen=True)
class _Work:
    question_index: int
    question: PreparedQuestion
    token_ids: tuple[int, ...]


class ClassificationService:
    def __init__(self, settings: Settings, compiler: PromptCompiler, backend: VLLMBackend):
        self.settings = settings
        self.compiler = compiler
        self.backend = backend
        self.primer = PrefixPrimer(
            settings.service.warm_hint_ttl_seconds, settings.service.warm_hint_entries
        )
        self._active = asyncio.Semaphore(settings.service.max_active_requests)

    async def close(self) -> None:
        await self.backend.close()

    async def verify_backend(self) -> dict:
        question = Question(
            id="probe",
            question="Does the document contain the word apple?",
            options=[
                Option(id="yes", description="It contains apple."),
                Option(id="no", description="It does not contain apple."),
            ],
        )
        request = ClassifyRequest(context="apple", questions=[question])
        batch = self.compiler.compile(request)
        job = batch.questions[0]
        await self.backend.check_tokenizer(
            self.compiler.render(request.context, question), job.prompt_ids
        )
        result = await self.backend.score(job.prompt_ids, job.candidate_ids, "llmjv-probe-v1")
        return {
            "tokenizer_matches": True,
            "candidate_coverage": len(result.logprobs),
            "backend_completion_tokens": result.completion_tokens,
        }

    async def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        try:
            await asyncio.wait_for(
                self._active.acquire(), timeout=self.settings.backend.queue_timeout_seconds
            )
        except TimeoutError as exc:
            raise BackendBusy("classification request queue is full") from exc
        try:
            async with asyncio.timeout(self.settings.service.request_timeout_seconds):
                return await self._classify(request)
        except TimeoutError as exc:
            raise BackendTimeout("classification exceeded its total deadline") from exc
        finally:
            self._active.release()

    async def _classify(self, request: ClassifyRequest) -> ClassifyResponse:
        started = time.perf_counter()
        batch = await asyncio.to_thread(self.compiler.compile, request)
        prepared_at = time.perf_counter()
        salt = hashlib.sha256(
            (self.compiler.fingerprint + "\0" + request.cache_namespace).encode()
        ).hexdigest()
        work = []
        size = self.settings.backend.logprob_chunk_size
        for index, question in enumerate(batch.questions):
            for start in range(0, len(question.candidate_ids), size):
                work.append(_Work(index, question, question.candidate_ids[start : start + size]))

        async def score(item: _Work) -> ScoreResult:
            return await self.backend.score(item.question.prompt_ids, item.token_ids, salt)

        primed_result = None
        if (
            request.execution == "auto"
            and self.settings.engine.enable_prefix_caching
            and len(work) > 1
            and batch.prefix_tokens >= self.settings.service.prime_min_prefix_tokens
        ):
            primed_result = await self.primer.prime(salt + batch.prefix_key, lambda: score(work[0]))

        remaining = work[1:] if primed_result is not None else work
        if request.execution == "serial":
            results = [await score(item) for item in remaining]
        else:
            results = await gather_cancel_on_error([score(item) for item in remaining])
        if primed_result is not None:
            results.insert(0, primed_result)

        collected: list[dict[int, float]] = [{} for _ in batch.questions]
        for item, result in zip(work, results, strict=True):
            target = collected[item.question_index]
            if target.keys() & result.logprobs.keys():
                raise BackendProtocolError("duplicate token scores across candidate chunks")
            target.update(result.logprobs)
        answers = []
        for question, scores in zip(batch.questions, collected, strict=True):
            if set(scores) != set(question.candidate_ids):
                raise BackendProtocolError("incomplete candidate scores after chunk assembly")
            answers.append(
                make_answer(
                    question.question,
                    [scores[t] for t in question.candidate_ids],
                    request.temperature,
                )
            )
        done = time.perf_counter()
        cached = [result.cached_prompt_tokens for result in results]
        return ClassifyResponse(
            request_id=str(uuid.uuid4()),
            model=self.settings.model.id,
            model_revision=self.settings.model.revision,
            answers=answers,
            usage=Usage(
                logical_prompt_tokens=batch.logical_prompt_tokens,
                backend_prompt_tokens=sum(result.prompt_tokens for result in results),
                backend_completion_tokens=sum(result.completion_tokens for result in results),
                backend_cached_prompt_tokens=(
                    sum(cached) if all(x is not None for x in cached) else None
                ),
                scoring_calls=len(results),
                prefix_tokens=batch.prefix_tokens,
                prefix_tokenization_cache_hit=batch.tokenization_cache_hit,
                primed=primed_result is not None,
            ),
            timing=Timing(
                preparation_ms=(prepared_at - started) * 1000,
                scoring_ms=(done - prepared_at) * 1000,
                total_ms=(done - started) * 1000,
            ),
        )
