"""Repeat a pass until no work is pending or a deadline has passed.

:func:`poll_until_settled` runs a pass, asks how much work it left pending,
and sleeps between passes. It returns once a pass leaves nothing pending or
the deadline has passed. The deadline is soft: no pass starts after it, but
a pass that is running finishes. A pass that raises a retryable error counts
as pending and the loop goes on; any other exception, ``KeyboardInterrupt``
and ``CancelledError`` propagate. The clock and the sleep are injectable,
so tests run the loop without waiting.

:func:`wait_options` returns the ``--wait``, ``--poll-minutes`` and
``--wait-hours`` options of a ``batch check``, so every tool declares them
alike.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from .options import Kind, Option
from .retry import classify_error

__all__ = [
    "DEFAULT_POLL_MINUTES",
    "PollResult",
    "Stop",
    "WaitLimits",
    "WaitTick",
    "poll_until_settled",
    "wait_options",
]

DEFAULT_POLL_MINUTES = 15.0


class Stop(StrEnum):
    """Why the loop ended: nothing pending, or the deadline passed."""

    SETTLED = "settled"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class WaitLimits:
    """Seconds between passes and seconds after which no pass starts."""

    interval: float
    timeout: float

    def __post_init__(self) -> None:
        for name, value in (("interval", self.interval), ("timeout", self.timeout)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite number >= 0, not {value}")


@dataclass(frozen=True)
class WaitTick:
    """Record the polling state before a sleep.

    Attributes:
        passes: Passes run so far.
        pending: Work the passes left pending.
        elapsed: Seconds since the first pass began.
        next_in: Seconds until the next pass.
        timeout: Seconds after which polling gives up.
    """

    passes: int
    pending: int
    elapsed: float
    next_in: float
    timeout: float


@dataclass(frozen=True)
class PollResult[T]:
    """How the loop ended.

    *last* is the result of the last pass that returned (None when every
    pass raised a retryable error), *passes* counts the passes run and
    *elapsed* the seconds since the first one began.
    """

    last: T | None
    stop: Stop
    passes: int
    elapsed: float


def _retryable(exc: BaseException) -> bool:
    return classify_error(exc).retryable


def _ignore(*_: object) -> None:
    return None


async def poll_until_settled[T](
    run_pass: Callable[[int], Awaitable[T]],
    pending: Callable[[T], int],
    limits: WaitLimits,
    *,
    on_wait: Callable[[WaitTick], None] = _ignore,
    on_retry: Callable[[int, Exception], None] = _ignore,
    retryable: Callable[[BaseException], bool] = _retryable,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> PollResult[T]:
    """Run *run_pass* until *pending* of its result is 0 or the timeout passes.

    *run_pass* receives the 1-based number of the pass. A pass whose
    exception *retryable* accepts is reported to *on_retry* and counts as
    pending. Before each sleep *on_wait* receives a :class:`WaitTick`; the
    sleep lasts the interval, or the time left until the timeout when that
    is shorter, so the last pass starts at the latest on the deadline.
    """
    start = clock()
    passes = 0
    last: T | None = None
    while True:
        passes += 1
        try:
            last = await run_pass(passes)
        except Exception as exc:
            if not retryable(exc):
                raise
            on_retry(passes, exc)
            left = 1
        else:
            left = pending(last)
        elapsed = clock() - start
        if left <= 0:
            return PollResult(last, Stop.SETTLED, passes, elapsed)
        remaining = limits.timeout - elapsed
        if remaining <= 0:
            return PollResult(last, Stop.TIMEOUT, passes, elapsed)
        delay = min(limits.interval, remaining)
        on_wait(WaitTick(passes, left, elapsed, delay, limits.timeout))
        await sleep(delay)


def _positive(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("must be a positive number")
    return value


def wait_options(*, group: str, default_hours: float) -> tuple[Option, Option, Option]:
    """Return the ``--wait``, ``--poll-minutes`` and ``--wait-hours`` options.

    They fill the spec fields ``wait``, ``poll_minutes`` and ``wait_hours``
    and take no settings default. *default_hours* is the tool's timeout.
    """
    return (
        Option(
            "wait",
            "--wait",
            "check again every --poll-minutes until no batch work is pending, "
            "or until --wait-hours have passed",
            group=group,
            kind=Kind.FLAG,
            default=False,
            target="wait",
            settable=False,
        ),
        Option(
            "poll_minutes",
            "--poll-minutes",
            "minutes between two checks with --wait",
            group=group,
            type=_positive,
            metavar="N",
            default=DEFAULT_POLL_MINUTES,
            default_help=f"{DEFAULT_POLL_MINUTES:g}",
            target="poll_minutes",
            settable=False,
        ),
        Option(
            "wait_hours",
            "--wait-hours",
            "hours after which --wait starts no further check; a running check "
            "finishes",
            group=group,
            type=_positive,
            metavar="N",
            default=float(default_hours),
            default_help=f"{default_hours:g}",
            target="wait_hours",
            settable=False,
        ),
    )
