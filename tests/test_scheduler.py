import asyncio

import pytest

from llm_json_verifier.scheduler import PrefixPrimer, gather_cancel_on_error


async def test_single_priming_call_for_concurrent_followers():
    primer = PrefixPrimer(120, 8)
    called = 0

    async def work():
        nonlocal called
        called += 1
        await asyncio.sleep(0.01)
        return "done"

    results = await asyncio.gather(*(primer.prime("same", work) for _ in range(20)))
    assert results.count("done") == 1 and results.count(None) == 19
    assert called == 1 and primer._leases == {}


async def test_cancelled_leader_does_not_poison_waiters():
    primer = PrefixPrimer(120, 8)
    started = asyncio.Event()

    async def long_job():
        started.set()
        await asyncio.Event().wait()

    async def replacement():
        return "replacement"

    leader = asyncio.create_task(primer.prime("same", long_job))
    await started.wait()
    follower = asyncio.create_task(primer.prime("same", replacement))
    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader
    assert await asyncio.wait_for(follower, 0.5) == "replacement"
    assert primer._leases == {}


async def test_failure_cancels_and_drains_siblings():
    cancelled = asyncio.Event()
    started = asyncio.Event()

    async def sibling():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def broken():
        await started.wait()
        raise RuntimeError("broken")

    with pytest.raises(RuntimeError, match="broken"):
        await gather_cancel_on_error([sibling(), broken()])
    assert cancelled.is_set()


async def test_ttl_and_capacity():
    primer = PrefixPrimer(0, 1)

    async def work():
        return 1

    assert await primer.prime("a", work) == 1
    assert await primer.prime("a", work) == 1
    await primer.prime("b", work)
    assert len(primer._warm) == 1
