"""Prime cold prefixes using useful work, then fan out bounded scoring calls."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")


@dataclass
class _Lease:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class PrefixPrimer:
    """Hints about engine cache warmth, not a claim that blocks cannot be evicted.

    Concurrent cold requests for the same namespace/document share the priming
    barrier. No dummy generation is needed: the leader scores its first question.
    Cancellation or failure never marks the prefix warm.
    """

    def __init__(self, ttl: float, capacity: int):
        self.ttl = ttl
        self.capacity = capacity
        self._warm: OrderedDict[str, float] = OrderedDict()
        self._leases: dict[str, _Lease] = {}

    def _is_warm(self, key: str) -> bool:
        expiry = self._warm.get(key)
        if expiry is None:
            return False
        if expiry <= time.monotonic():
            del self._warm[key]
            return False
        self._warm.move_to_end(key)
        return True

    async def prime(self, key: str, first: Callable[[], Awaitable[T]]) -> T | None:
        # All dictionary mutations occur on the one owning event loop. There is
        # no await between lease lookup and increment, so the lease cannot race.
        if self._is_warm(key):
            return None
        lease = self._leases.setdefault(key, _Lease())
        lease.users += 1
        try:
            async with lease.lock:
                if self._is_warm(key):
                    return None
                result = await first()
                self._warm[key] = time.monotonic() + self.ttl
                self._warm.move_to_end(key)
                while len(self._warm) > self.capacity:
                    self._warm.popitem(last=False)
                return result
        finally:
            lease.users -= 1
            if lease.users == 0:
                del self._leases[key]


async def gather_cancel_on_error(calls: list[Awaitable[T]]) -> list[T]:
    tasks = [asyncio.ensure_future(call) for call in calls]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
