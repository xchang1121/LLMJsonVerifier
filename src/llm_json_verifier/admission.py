"""FIFO admission with explicit bounds on active work and pending waiters."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager

from .errors import BackendBusy


class AdmissionGate:
    """Owned by one event loop; every granted slot must be released once."""

    def __init__(self, capacity: int, max_queued: int, timeout: float, name: str):
        if capacity < 1 or max_queued < 0 or timeout <= 0:
            raise ValueError("invalid admission limits")
        self.capacity, self.max_queued, self.timeout, self.name = (
            capacity,
            max_queued,
            timeout,
            name,
        )
        self.active = 0
        self._waiters: deque[asyncio.Future] = deque()

    @property
    def queued(self) -> int:
        return len(self._waiters)

    async def acquire(self) -> float:
        started = time.perf_counter()
        if self.active < self.capacity and not self._waiters:
            self.active += 1
            return 0.0
        if self.queued >= self.max_queued:
            raise BackendBusy(f"{self.name} queue is full")
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            async with asyncio.timeout(self.timeout):
                await waiter
        except BaseException as exc:
            if waiter.done() and not waiter.cancelled():
                # A release granted this slot, then its owner was canceled before
                # resuming. Pass that reserved slot to the next waiter.
                self.release()
            else:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass  # release() already skipped the canceled waiter.
            if isinstance(exc, TimeoutError):
                raise BackendBusy(f"{self.name} queue wait exceeded its deadline") from exc
            raise
        return (time.perf_counter() - started) * 1000

    def release(self) -> None:
        if self.active <= 0:
            raise RuntimeError("admission slot released twice")
        self.active -= 1
        while self._waiters and self.active < self.capacity:
            waiter = self._waiters.popleft()
            if not waiter.done():
                self.active += 1
                waiter.set_result(None)

    @asynccontextmanager
    async def slot(self):
        waited_ms = await self.acquire()
        try:
            yield waited_ms
        finally:
            self.release()
