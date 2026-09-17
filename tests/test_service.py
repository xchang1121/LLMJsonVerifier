import asyncio
import math

import pytest

from llm_json_verifier.backend import ScoreResult
from llm_json_verifier.errors import BackendProtocolError, BackendTimeout
from llm_json_verifier.prompts import PromptCompiler
from llm_json_verifier.schemas import Option
from llm_json_verifier.service import ClassificationService


class RecordingBackend:
    def __init__(self):
        self.events = []
        self.calls = []
        self.finished = 0

    async def score(self, prompt, ids, salt):
        index = len(self.calls)
        self.calls.append((prompt, ids, salt))
        self.events.append(("start", index))
        await asyncio.sleep(0.005)
        self.events.append(("finish", index))
        self.finished += 1
        # Global-vocabulary log probabilities; identical scale across chunks.
        return ScoreResult({token: -float(token) / 100 for token in ids}, len(prompt), 1, 0)

    async def close(self):
        pass


async def test_more_than_128_candidates_uses_one_global_normalization(
    settings, tokenizer, question, request_body
):
    options = [Option(id=str(i), description=f"Distinct candidate {i}") for i in range(256)]
    body = request_body.model_copy(
        update={"questions": [question.model_copy(update={"options": options})]}
    )
    compiler = PromptCompiler(tokenizer, settings)
    backend = RecordingBackend()
    result = await ClassificationService(settings, compiler, backend).classify(body)
    assert len(backend.calls) == 2
    assert backend.calls[0][0] == backend.calls[1][0]  # All options, not a chunk-specific prompt.
    assert [len(ids) for _, ids, _ in backend.calls] == [128, 128]
    scores = [-token / 100 for token in compiler.code_ids[:256]]
    expected = math.exp(scores[0] - max(scores)) / sum(math.exp(x - max(scores)) for x in scores)
    assert result.answers[0].probabilities["0"] == pytest.approx(expected)
    assert sum(result.answers[0].probabilities.values()) == pytest.approx(1)
    assert result.usage.backend_prompt_tokens == 2 * result.usage.logical_prompt_tokens
    assert result.usage.backend_completion_tokens == 2


async def test_cold_prefix_primes_with_real_work_then_fans_out(
    settings, tokenizer, question, request_body
):
    settings.service.prime_min_prefix_tokens = 0
    body = request_body.model_copy(
        update={"questions": [question.model_copy(update={"id": str(i)}) for i in range(4)]}
    )
    backend = RecordingBackend()
    service = ClassificationService(settings, PromptCompiler(tokenizer, settings), backend)
    first = await service.classify(body)
    assert backend.events[:2] == [("start", 0), ("finish", 0)]
    assert backend.events[2:5] == [("start", 1), ("start", 2), ("start", 3)]
    assert first.usage.primed and len(backend.calls) == 4  # No dummy request.
    second = await service.classify(body)
    assert not second.usage.primed
    assert second.usage.prefix_tokenization_cache_hit
    assert first.answers == second.answers
    isolated = await service.classify(body.model_copy(update={"cache_namespace": "another"}))
    assert isolated.usage.primed
    assert backend.calls[0][2] != backend.calls[-1][2]


async def test_serial_and_parallel_produce_same_ordered_output(
    settings, compiler, question, request_body
):
    service = ClassificationService(settings, compiler, RecordingBackend())
    questions = [question.model_copy(update={"id": str(i)}) for i in range(4)]
    serial = await service.classify(
        request_body.model_copy(update={"questions": questions, "execution": "serial"})
    )
    parallel = await service.classify(
        request_body.model_copy(update={"questions": questions, "execution": "parallel"})
    )
    assert serial.answers == parallel.answers
    assert [answer.id for answer in parallel.answers] == ["0", "1", "2", "3"]


async def test_missing_chunk_scores_fail_atomically(settings, compiler, request_body):
    class BrokenBackend(RecordingBackend):
        async def score(self, prompt, ids, salt):
            return ScoreResult({ids[0]: -1.0}, len(prompt), 1, None)

    with pytest.raises(BackendProtocolError, match="incomplete"):
        await ClassificationService(settings, compiler, BrokenBackend()).classify(request_body)


async def test_request_deadline_cancels_backend_and_releases_capacity(
    settings, compiler, request_body
):
    settings.service.max_active_requests = 1
    settings.service.request_timeout_seconds = 0.1
    cancelled = asyncio.Event()

    class SlowBackend(RecordingBackend):
        async def score(self, prompt, ids, salt):
            if not cancelled.is_set():
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return await super().score(prompt, ids, salt)

    service = ClassificationService(settings, compiler, SlowBackend())
    with pytest.raises(BackendTimeout):
        await service.classify(request_body)
    assert cancelled.is_set()
    assert (await service.classify(request_body)).answers
