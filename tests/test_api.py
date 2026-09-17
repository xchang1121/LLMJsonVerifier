import json

import httpx
import pytest
from conftest import completion
from fastapi.testclient import TestClient

from llm_json_verifier.api import create_app
from llm_json_verifier.backend import VLLMBackend
from llm_json_verifier.schemas import ClassifyResponse
from llm_json_verifier.service import ClassificationService


def api_client(settings, compiler):
    async def handle(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": settings.model.served_name}]})
        if request.url.path == "/health":
            return httpx.Response(200)
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": compiler.tokenizer.encode(body["prompt"])})
        return httpx.Response(200, json=completion(body["prompt"], body["logprob_token_ids"]))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://engine")
    service = ClassificationService(settings, compiler, VLLMBackend(settings, http))
    # Let the service own/close this otherwise externally constructed test client.
    service.backend._owns_client = True
    return TestClient(create_app(settings, service))


def test_api_end_to_end_with_real_http_protocol_adapter(settings, compiler, request_body):
    with api_client(settings, compiler) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        assert client.get("/v1/info").json()["available_codes"] == 702
        response = client.post("/v1/classify", json=request_body.model_dump())
        assert response.status_code == 200, response.text
        result = ClassifyResponse.model_validate(response.json())
        assert result.answers[0].selected == "yes"
        assert set(result.answers[0].probabilities) == {"yes", "no", "unknown"}
        assert result.usage.backend_completion_tokens == 1


@pytest.mark.parametrize("bad", ["extra", "duplicate", "empty", "temperature", "ambiguous"])
def test_bad_requests_rejected(settings, compiler, request_body, bad):
    body = request_body.model_dump()
    if bad == "extra":
        body["unexpected"] = "SECRET_DOCUMENT"
    elif bad == "duplicate":
        body["questions"] *= 2
    elif bad == "empty":
        body["context"] = "  "
    elif bad == "temperature":
        body["temperature"] = "1.0"
    else:
        body["questions"][0]["options"][1]["description"] = body["questions"][0]["options"][0][
            "description"
        ]
    with api_client(settings, compiler) as client:
        response = client.post("/v1/classify", json=body)
    assert response.status_code == 422
    assert "SECRET_DOCUMENT" not in response.text


def test_auth_and_body_limit(settings, compiler, request_body, monkeypatch):
    monkeypatch.setenv("LLMJV_API_KEY", "test-secret")
    settings.service.max_request_bytes = 1024
    with api_client(settings, compiler) as client:
        assert client.get("/v1/info").status_code == 401
        assert (
            client.get("/v1/info", headers={"Authorization": "Bearer test-secret"}).status_code
            == 200
        )
        response = client.post(
            "/v1/classify",
            content=b"x" * 1025,
            headers={"Authorization": "Bearer test-secret", "Content-Type": "application/json"},
        )
        assert response.status_code == 413
