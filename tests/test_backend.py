import asyncio
import copy
import json

import httpx
import pytest
from conftest import completion

from llm_json_verifier.backend import VLLMBackend, parse_scores
from llm_json_verifier.errors import BackendBusy, BackendProtocolError, BackendTimeout


async def test_exact_token_ids_and_unused_generated_text(settings):
    async def handle(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/completions"
        assert body["prompt"] == [100, 200]
        assert body["logprob_token_ids"] == [32, 33, 34]
        assert body["max_tokens"] == 1 and body["ignore_eos"]
        assert body["return_tokens_as_token_ids"]
        assert body["cache_salt"] == "isolation-key"
        assert "allowed_token_ids" not in body and "logit_bias" not in body
        return httpx.Response(
            200, json=completion(body["prompt"], body["logprob_token_ids"], [-8, -900, -4])
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://test"
    ) as client:
        result = await VLLMBackend(settings, client).score(
            (100, 200), (32, 33, 34), "isolation-key"
        )
    assert result.logprobs == {32: -8, 33: -900, 34: -4}


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "nan",
        "sentinel",
        "positive",
        "bool",
        "truncated",
        "usage",
        "text_keys",
        "bad_detail",
        "multi",
    ],
)
def test_malformed_responses_never_fabricate_probabilities(corruption):
    data = copy.deepcopy(completion((1, 2), (32, 33)))
    row = data["choices"][0]["logprobs"]["top_logprobs"][0]
    if corruption == "missing":
        row.pop("token_id:33")
    elif corruption in {"nan", "sentinel", "positive", "bool"}:
        row["token_id:33"] = {
            "nan": float("nan"),
            "sentinel": -9999,
            "positive": 0.1,
            "bool": True,
        }[corruption]
    elif corruption == "truncated":
        data["choices"][0]["finish_reason"] = "stop"
    elif corruption == "usage":
        data["usage"]["prompt_tokens"] = 1
    elif corruption == "text_keys":
        data["choices"][0]["logprobs"]["top_logprobs"] = [{"A": -1, "B": -2}]
    elif corruption == "bad_detail":
        data["usage"]["prompt_tokens_details"] = "bad"
    else:
        data["choices"].append(data["choices"][0])
    with pytest.raises(BackendProtocolError):
        parse_scores(data, (32, 33), 2)


async def test_bounded_concurrency(settings):
    settings.backend.max_in_flight = 2
    active = peak = 0

    async def handle(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        body = json.loads(request.content)
        return httpx.Response(200, json=completion(body["prompt"], body["logprob_token_ids"]))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://test"
    ) as client:
        backend = VLLMBackend(settings, client)
        await asyncio.gather(*(backend.score((1,), (32, 33), "salt") for _ in range(8)))
    assert peak == 2 and active == 0


@pytest.mark.parametrize("failure", ["timeout", "busy"])
async def test_failure_releases_capacity(settings, failure):
    settings.backend.max_in_flight = 1
    count = 0

    def handle(request):
        nonlocal count
        count += 1
        if count == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("timeout")
            return httpx.Response(503)
        return httpx.Response(200, json=completion((1,), (32, 33)))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://test"
    ) as client:
        backend = VLLMBackend(settings, client)
        with pytest.raises(BackendTimeout if failure == "timeout" else BackendBusy):
            await backend.score((1,), (32, 33), "salt")
        assert (await backend.score((1,), (32, 33), "salt")).completion_tokens == 1


async def test_tokenizer_mismatch_detected_and_empty_health_valid(settings):
    def handle(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": settings.model.served_name}]})
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(200, json={"tokens": [999]})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://test"
    ) as client:
        backend = VLLMBackend(settings, client)
        assert await backend._request("GET", "/health") == {}
        with pytest.raises(BackendProtocolError, match="disagree"):
            await backend.check_tokenizer("test", (1, 2))
