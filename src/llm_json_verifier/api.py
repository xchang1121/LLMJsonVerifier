"""FastAPI gateway. It can run on CPU/Windows while vLLM runs on a GPU host."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from . import __version__
from .backend import VLLMBackend
from .config import Settings, load_settings
from .errors import VerifierError
from .jsonio import strict_json_loads
from .prompts import load_compiler
from .schemas import (
    ClassifyRequest,
    ClassifyResponse,
    SchemaClassifyRequest,
    SchemaClassifyResponse,
)
from .service import ClassificationService

logger = logging.getLogger(__name__)


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            size += len(body)
            if size > self.max_bytes:
                await JSONResponse(
                    status_code=413,
                    content={
                        "error": {"code": "body_too_large", "message": "request body exceeds limit"}
                    },
                )(scope, receive, send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break
        delivered = False
        disconnected = asyncio.Event()
        body = b"".join(chunks)
        try:
            strict_json_loads(body)
        except (ValueError, RecursionError):
            await JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "code": "invalid_json",
                        "message": "body must be finite JSON with unique object keys",
                    }
                },
            )(scope, receive, send)
            return

        async def replay() -> dict:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def watch_disconnect():
            # Only this coroutine reads the original channel after the body.
            while (await receive())["type"] != "http.disconnect":
                pass
            disconnected.set()

        application = asyncio.create_task(self.app(scope, replay, send))
        watcher = asyncio.create_task(watch_disconnect())
        try:
            done, _ = await asyncio.wait(
                (application, watcher), return_when=asyncio.FIRST_COMPLETED
            )
            if application in done:
                await application
            else:
                await watcher
                application.cancel()
        finally:
            for task in (application, watcher):
                if not task.done():
                    task.cancel()
            await asyncio.gather(application, watcher, return_exceptions=True)


def create_app(
    settings: Settings | None = None, service: ClassificationService | None = None
) -> FastAPI:
    settings = settings or load_settings()
    api_key = os.environ.get("LLMJV_API_KEY")

    async def authorize(authorization: str | None = Header(default=None)) -> None:
        if api_key and not secrets.compare_digest(authorization or "", f"Bearer {api_key}"):
            raise HTTPException(status_code=401, detail="invalid API key")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active = service
        if active is None:
            compiler = await asyncio.to_thread(load_compiler, settings)
            active = ClassificationService(settings, compiler, VLLMBackend(settings))
        app.state.service = active
        app.state.ready = False
        try:
            if settings.backend.verify_on_startup:
                await active.verify_backend()
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            await active.close()

    app = FastAPI(
        title="LLMJsonVerifier",
        version=__version__,
        lifespan=lifespan,
        description="Closed-set classification; probabilities are not calibrated correctness estimates.",
    )
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.service.max_request_bytes)

    @app.exception_handler(VerifierError)
    async def verifier_error(_request, exc: VerifierError):
        return JSONResponse(
            status_code=exc.status_code, content={"error": {"code": exc.code, "message": str(exc)}}
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, exc: RequestValidationError):
        issues = [
            {"location": list(e["loc"]), "message": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "request failed validation",
                    "issues": issues,
                }
            },
        )

    @app.get("/healthz")
    async def health():
        return {"status": "alive", "version": __version__}

    @app.get("/readyz")
    async def ready():
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="not ready")
        await app.state.service.backend._request("GET", "/health")
        return {"status": "ready"}

    @app.get("/v1/info", dependencies=[Depends(authorize)])
    async def info():
        return {
            "version": __version__,
            **app.state.service.compiler.registry_summary(),
            "max_model_len": settings.model.max_model_len,
            "max_questions": settings.service.max_questions,
            "admission": {
                "active": app.state.service.admission.active,
                "queued": app.state.service.admission.queued,
                "max_active": settings.service.max_active_requests,
                "max_queued": settings.service.max_queued_requests,
            },
            "probability_kind": "candidate_conditional_uncalibrated",
        }

    @app.post("/v1/classify", response_model=ClassifyResponse, dependencies=[Depends(authorize)])
    async def classify(body: ClassifyRequest):
        result = await app.state.service.classify(body)
        logger.info(
            "request=%s questions=%d calls=%d cached_tokens=%s elapsed_ms=%.1f",
            result.request_id,
            len(result.answers),
            result.usage.scoring_calls,
            result.usage.backend_cached_prompt_tokens,
            result.timing.total_ms,
        )
        return result

    @app.post(
        "/v1/classify-schema",
        response_model=SchemaClassifyResponse,
        dependencies=[Depends(authorize)],
    )
    async def classify_schema(body: SchemaClassifyRequest):
        result = await app.state.service.classify_schema(body)
        logger.info(
            "request=%s fields=%d calls=%d elapsed_ms=%.1f",
            result.request_id,
            len(result.fields),
            result.usage.scoring_calls,
            result.timing.total_ms,
        )
        return result

    return app
