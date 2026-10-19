"""Fixtures of the LLM-layer tests."""

from __future__ import annotations

import asyncio

import pytest

_REAL_SLEEP = asyncio.sleep


class Sleeps(list[float]):
    """The delays slept, in order; :meth:`clock` advances by each of them."""

    now: float = 0.0

    def clock(self) -> float:
        """Return the fake time: the sum of the delays slept so far."""
        return self.now


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> Sleeps:
    """Record every ``asyncio.sleep`` delay and only yield instead of waiting.

    Pass ``sleeps.clock`` to rate limiters so their waits end in fake time.
    """
    recorded = Sleeps()

    async def fake_sleep(delay: float, result: object = None) -> object:
        recorded.append(delay)
        recorded.now += max(0.0, delay)
        await _REAL_SLEEP(0)
        return result

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return recorded
