import json

import httpx
import pytest

from llm_json_verifier.client import GatewayClient, benchmark, evaluate, verify_cache
from llm_json_verifier.probabilities import make_answer
from llm_json_verifier.schemas import ClassifyResponse, Timing, Usage


class FakeGateway:
    def __init__(self, report_hits=True, drift=False):
        self.requests = []
        self.namespaces = set()
        self.report_hits, self.drift = report_hits, drift

    async def classify(self, request):
        warm = request.cache_namespace in self.namespaces
        self.namespaces.add(request.cache_namespace)
        self.requests.append(request)
        scores = [-float(i + 1) for i in range(len(request.questions[0].options))]
        if self.drift and warm:
            scores.reverse()
        answers = [make_answer(q, scores, request.temperature) for q in request.questions]
        return ClassifyResponse(
            request_id="fake",
            model="fake",
            model_revision="fake",
            answers=answers,
            usage=Usage(
                logical_prompt_tokens=1000,
                backend_prompt_tokens=1000,
                backend_completion_tokens=len(answers),
                backend_cached_prompt_tokens=512 if warm and self.report_hits else 0,
                scoring_calls=len(answers),
                prefix_tokens=600,
                prefix_tokenization_cache_hit=warm,
                primed=False,
            ),
            timing=Timing(preparation_ms=0.0, scoring_ms=1.0, total_ms=1.0),
        )


@pytest.mark.parametrize(
    "hits,drift,passed", [(True, False, True), (False, False, False), (True, True, False)]
)
async def test_cache_verifier_needs_agreement_and_observed_hit(request_body, hits, drift, passed):
    client = FakeGateway(hits, drift)
    result = await verify_cache(client, request_body, 1e-3)
    assert result["passed"] is passed
    assert client.requests[0].cache_namespace != client.requests[1].cache_namespace
    assert client.requests[1].cache_namespace == client.requests[2].cache_namespace


async def test_evaluation_keeps_gold_out_of_request_and_checks_rotation(tmp_path, request_body):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps({"request": request_body.model_dump(), "expected": {"status": "yes"}}),
        encoding="utf-8",
    )
    client = FakeGateway()
    result = await evaluate(client, path, rotate=True)
    assert result["accuracy"] == 1
    assert result["one_step_rotation_agreement"] == 0
    assert len(client.requests) == 2
    assert "expected" not in client.requests[0].model_dump()
    assert [o.id for o in client.requests[1].questions[0].options] == ["no", "unknown", "yes"]


async def test_cold_benchmark_uses_unique_salts(request_body):
    client = FakeGateway()
    result = await benchmark(
        client, request_body, repeats=3, concurrency=2, warmup=0, cache_mode="cold"
    )
    assert len({request.cache_namespace for request in client.requests}) == 3
    assert result["backend_cached_prompt_tokens"] == 0
    assert result["backend_completion_tokens"] == 3
    assert result["questions_per_second"] > 0


async def test_remote_response_cannot_silently_drop_questions(request_body):
    data = (await FakeGateway().classify(request_body)).model_dump()
    data["answers"] = []
    async with GatewayClient("http://test") as client:
        await client.http.aclose()
        client.http = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=data)),
            base_url="http://test",
        )
        with pytest.raises(RuntimeError, match="cover"):
            await client.classify(request_body)
