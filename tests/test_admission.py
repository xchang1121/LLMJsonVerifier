import asyncio
import json

import httpx
import pytest
from conftest import completion

from llm_json_verifier.admission import AdmissionGate
from llm_json_verifier.api import create_app
from llm_json_verifier.backend import VLLMBackend
from llm_json_verifier.errors import BackendBusy, BackendTimeout
from llm_json_verifier.service import ClassificationService


async def test_queue_is_bounded_fifo_and_cancel_removes_waiter():
    gate = AdmissionGate(1, 2, 1, "test")
    await gate.acquire()
    first = asyncio.create_task(gate.acquire())
    second = asyncio.create_task(gate.acquire())
    await asyncio.sleep(0)
    assert gate.active == 1 and gate.queued == 2
    with pytest.raises(BackendBusy, match="full"):
        await gate.acquire()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    gate.release()
    assert await second >= 0
    assert gate.active == 1 and gate.queued == 0
    gate.release()
    assert gate.active == 0


async def test_cancel_after_grant_passes_reserved_slot_forward():
    gate = AdmissionGate(1, 2, 1, "test")
    await gate.acquire()
    first = asyncio.create_task(gate.acquire())
    second = asyncio.create_task(gate.acquire())
    await asyncio.sleep(0)
    gate.release()  # first owns the slot, but has not yet resumed.
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.wait_for(second, 1)
    assert gate.active == 1 and gate.queued == 0
    gate.release()
    assert gate.active == 0


async def test_queue_timeout_and_zero_queue_capacity():
    gate = AdmissionGate(1, 1, 0.01, "test")
    await gate.acquire()
    with pytest.raises(BackendBusy, match="deadline"):
        await gate.acquire()
    assert gate.queued == 0 and gate.active == 1
    gate.release()
    no_wait = AdmissionGate(1, 0, 1, "test")
    async with no_wait.slot():
        with pytest.raises(BackendBusy, match="full"):
            await no_wait.acquire()
    assert no_wait.active == 0


async def test_backend_rejects_full_queue_before_http(settings):
    settings.backend.max_in_flight = 1
    settings.backend.max_queued_scores = 0
    entered, release = asyncio.Event(), asyncio.Event()

    async def handle(request):
        entered.set()
        await release.wait()
        return httpx.Response(200, json=completion((1,), (32, 33)))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://test"
    ) as http:
        backend = VLLMBackend(settings, http)
        task = asyncio.create_task(backend.score((1,), (32, 33), "test"))
        await asyncio.wait_for(entered.wait(), 1)
        with pytest.raises(BackendBusy):
            await backend.score((1,), (32, 33), "test")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert backend.admission.active == 0
        release.set()
        assert (await backend.score((1,), (32, 33), "test")).queue_wait_ms == 0


async def test_service_close_cancels_active_and_waiting_requests(settings, compiler, request_body):
    settings.service.max_active_requests = 1
    entered = asyncio.Event()

    async def handle(request):
        entered.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://test"
    ) as http:
        service = ClassificationService(settings, compiler, VLLMBackend(settings, http))
        active = asyncio.create_task(service.classify(request_body))
        await asyncio.wait_for(entered.wait(), 1)
        queued = asyncio.create_task(service.classify(request_body))
        await asyncio.sleep(0)
        assert service.admission.queued == 1
        await asyncio.wait_for(service.close(), 1)
        assert active.cancelled() and queued.cancelled()
        assert service.admission.active == service.admission.queued == 0
        assert not service._tasks
        with pytest.raises(BackendBusy, match="shutting down"):
            await service.classify(request_body)


async def test_total_deadline_includes_admission_wait(settings, compiler, request_body):
    settings.service.max_active_requests = 1
    settings.service.request_timeout_seconds = 0.02
    async with httpx.AsyncClient(base_url="http://test") as http:
        service = ClassificationService(settings, compiler, VLLMBackend(settings, http))
        await service.admission.acquire()
        with pytest.raises(BackendTimeout):
            await service.classify(request_body)
        assert service.admission.queued == 0
        service.admission.release()
        await service.close()


async def test_http_disconnect_cancels_upstream_and_releases_capacity(
    settings, compiler, request_body
):
    settings.backend.verify_on_startup = False
    settings.service.max_active_requests = 1
    entered, canceled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def handle(request):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            canceled.set()
            raise
        body = json.loads(request.content)
        return httpx.Response(200, json=completion(body["prompt"], body["logprob_token_ids"]))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://test"
    ) as http:
        service = ClassificationService(settings, compiler, VLLMBackend(settings, http))
        app = create_app(settings, service)
        incoming = asyncio.Queue()
        await incoming.put(
            {
                "type": "http.request",
                "body": request_body.model_dump_json().encode(),
                "more_body": False,
            }
        )
        sent = []

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "method": "POST",
            "path": "/v1/classify",
            "raw_path": b"/v1/classify",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "scheme": "http",
            "http_version": "1.1",
            "server": ("test", 80),
            "client": ("test", 123),
        }
        async with app.router.lifespan_context(app):
            request = asyncio.create_task(app(scope, incoming.get, send))
            await asyncio.wait_for(entered.wait(), 1)
            await incoming.put({"type": "http.disconnect"})
            await asyncio.wait_for(request, 1)
            assert canceled.is_set()
            assert service.admission.active == service.backend.admission.active == 0
            assert not sent
            release.set()
            result = await service.classify(request_body)
            assert result.answers[0].selected == "yes"
