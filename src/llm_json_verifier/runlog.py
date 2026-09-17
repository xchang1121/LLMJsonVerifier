"""Append-only experiment records and per-attempt outcomes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx

from . import __version__
from .metrics import percentile


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_digest(value) -> str:
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode())


class RunLog:
    """Flush each completed attempt so interrupted runs retain their observations."""

    def __init__(self, operation: str, metadata: dict, path: str | Path | None = None):
        self.run_id = uuid.uuid4().hex
        self.path = str(path) if path is not None else None
        self.handle = None
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.handle = destination.open("x", encoding="utf-8", newline="\n")
        self.write(
            {
                "kind": "run",
                "format_version": 1,
                "run_id": self.run_id,
                "operation": operation,
                "created_at": datetime.now(UTC).isoformat(),
                "gateway_client_version": __version__,
                **metadata,
            }
        )

    def write(self, event: dict) -> None:
        if self.handle:
            self.handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            self.handle.flush()

    def finish(self, summary: dict) -> dict:
        summary.update(run_id=self.run_id, records_file=self.path)
        self.write({"kind": "summary", **summary})
        return summary

    def close(self) -> None:
        if self.handle:
            self.handle.close()


async def observe(client, request, log: RunLog, **metadata):
    event = {
        "kind": "request",
        "run_id": log.run_id,
        **metadata,
        "sent_at": datetime.now(UTC).isoformat(),
        "request": request.model_dump(),
    }
    started = time.perf_counter()
    result = None
    try:
        result = await client.classify(request)
        event.update(status="succeeded", response=result.model_dump())
    except asyncio.CancelledError:
        event["status"] = "canceled"
        raise
    except Exception as exc:
        event.update(status="failed", error={"type": type(exc).__name__})
        if isinstance(exc, httpx.HTTPStatusError):
            event["error"]["http_status"] = exc.response.status_code
        elif getattr(exc, "http_status", None) is not None:
            event["error"]["http_status"] = exc.http_status
    finally:
        event["elapsed_ms"] = (time.perf_counter() - started) * 1000
        # File I/O occurs after timing the request. The whole-run throughput
        # includes recording overhead when a records file is enabled.
        log.write(event)
    return {
        key: value for key, value in event.items() if key not in {"request", "response"}
    }, result


async def run_workers(items, concurrency: int, work):
    """Bound task creation as well as HTTP concurrency, preserving input order."""
    iterator = iter(enumerate(items))
    results = {}

    async def worker():
        for index, item in iterator:
            results[index] = await work(index, item)

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return [results[index] for index in sorted(results)]


def latency_summary(values: list[float]) -> dict:
    return {
        name: percentile(values, quantile) if values else None
        for name, quantile in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
    }


def outcome_summary(events: list[dict]) -> dict:
    succeeded = [event for event in events if event["status"] == "succeeded"]
    return {
        "attempted": len(events),
        "succeeded": len(succeeded),
        "failed": sum(event["status"] == "failed" for event in events),
        "canceled": sum(event["status"] == "canceled" for event in events),
        "latency_ms": latency_summary([event["elapsed_ms"] for event in succeeded]),
        "all_outcomes_latency_ms": latency_summary([event["elapsed_ms"] for event in events]),
    }
