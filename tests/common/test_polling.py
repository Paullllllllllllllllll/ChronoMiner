"""The shared poll loop: passes, sleeps, the soft deadline and pass errors.

A fake clock whose ``sleep`` advances it stands in for time, so no test
waits.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Sequence

import pytest

from chronominer.common.options import OptionError, OptionTable
from chronominer.common.polling import (
    PollResult,
    Stop,
    WaitLimits,
    WaitTick,
    poll_until_settled,
    wait_options,
)


class Clock:
    """A monotonic clock in seconds whose ``sleep`` advances it."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class ConnectionReset(ConnectionError):
    """A connection failure: retryable."""


class BadRequest(Exception):
    """A provider error with a client status: not retryable."""

    status_code = 400


def scripted(
    pending: Sequence[int | Exception],
    clock: Clock,
    *,
    duration: float = 0.0,
) -> tuple[Callable[[int], Awaitable[int]], list[int]]:
    """Return a pass that leaves *pending* work in turn, and its call log.

    Each pass takes *duration* seconds of the clock; an exception in the
    script is raised by its pass.
    """
    calls: list[int] = []
    script = list(pending)

    async def run_pass(number: int) -> int:
        calls.append(number)
        clock.now += duration
        outcome = script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return run_pass, calls


def poll(
    run_pass: Callable[[int], Awaitable[int]],
    clock: Clock,
    limits: WaitLimits,
    **hooks: object,
) -> PollResult[int]:
    return asyncio.run(
        poll_until_settled(
            run_pass,
            lambda left: left,
            limits,
            clock=clock,
            sleep=clock.sleep,
            **hooks,  # type: ignore[arg-type]
        )
    )


def test_a_first_pass_with_nothing_pending_settles_without_sleeping() -> None:
    clock = Clock()
    run_pass, calls = scripted([0], clock)

    result = poll(run_pass, clock, WaitLimits(900, 3600))

    assert (result.stop, result.passes, result.last) == (Stop.SETTLED, 1, 0)
    assert calls == [1]
    assert clock.sleeps == []


def test_pending_passes_sleep_the_interval_until_settled() -> None:
    clock = Clock()
    run_pass, calls = scripted([3, 1, 0], clock)
    ticks: list[WaitTick] = []

    result = poll(run_pass, clock, WaitLimits(900, 3600), on_wait=ticks.append)

    assert (result.stop, result.passes) == (Stop.SETTLED, 3)
    assert calls == [1, 2, 3]
    assert clock.sleeps == [900, 900]
    assert result.elapsed == 1800
    assert [(tick.passes, tick.pending, tick.next_in) for tick in ticks] == [
        (1, 3, 900),
        (2, 1, 900),
    ]


def test_the_last_pass_starts_on_the_deadline_and_none_after_it() -> None:
    clock = Clock()
    run_pass, calls = scripted([1, 1, 1, 1, 1], clock)

    result = poll(run_pass, clock, WaitLimits(900, 2000))

    assert (result.stop, result.passes) == (Stop.TIMEOUT, 4)
    assert clock.sleeps == [900, 900, 200]
    assert calls == [1, 2, 3, 4]


def test_a_running_pass_finishes_after_the_deadline() -> None:
    clock = Clock()
    run_pass, calls = scripted([1, 0], clock, duration=50)

    result = poll(run_pass, clock, WaitLimits(900, 960))

    assert (result.stop, result.passes, result.last) == (Stop.SETTLED, 2, 0)
    assert clock.sleeps == [900]
    assert result.elapsed == 1000


def test_a_pass_that_ends_past_the_deadline_stops_the_wait() -> None:
    clock = Clock()
    run_pass, calls = scripted([1, 1], clock, duration=600)

    result = poll(run_pass, clock, WaitLimits(900, 500))

    assert (result.stop, result.passes) == (Stop.TIMEOUT, 1)
    assert clock.sleeps == []


def test_a_retryable_pass_error_counts_as_pending() -> None:
    clock = Clock()
    error = ConnectionReset("connection reset by peer")
    run_pass, calls = scripted([error, 0], clock)
    retries: list[tuple[int, Exception]] = []

    result = poll(
        run_pass,
        clock,
        WaitLimits(60, 3600),
        on_retry=lambda number, exc: retries.append((number, exc)),
    )

    assert (result.stop, result.passes, result.last) == (Stop.SETTLED, 2, 0)
    assert retries == [(1, error)]
    assert clock.sleeps == [60]


def test_every_pass_failing_retryably_times_out_without_a_result() -> None:
    clock = Clock()
    run_pass, _ = scripted([TimeoutError("timed out")] * 3, clock)

    result = poll(run_pass, clock, WaitLimits(60, 120))

    assert (result.stop, result.passes, result.last) == (Stop.TIMEOUT, 3, None)


def test_a_non_retryable_pass_error_propagates() -> None:
    clock = Clock()
    run_pass, calls = scripted([1, BadRequest("unknown batch")], clock)

    with pytest.raises(BadRequest):
        poll(run_pass, clock, WaitLimits(60, 3600))

    assert calls == [1, 2]


def test_the_caller_decides_which_errors_are_retried() -> None:
    clock = Clock()
    run_pass, _ = scripted([ConnectionReset("reset")], clock)

    with pytest.raises(ConnectionReset):
        poll(run_pass, clock, WaitLimits(60, 3600), retryable=lambda _: False)


def test_an_interrupt_during_the_sleep_propagates() -> None:
    clock = Clock()
    run_pass, calls = scripted([1, 1], clock)

    async def interrupted(seconds: float) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(
            poll_until_settled(
                run_pass,
                lambda left: left,
                WaitLimits(60, 3600),
                clock=clock,
                sleep=interrupted,
            )
        )
    assert calls == [1]


def test_zero_limits_poll_once() -> None:
    clock = Clock()
    run_pass, calls = scripted([1], clock)

    result = poll(run_pass, clock, WaitLimits(0, 0))

    assert (result.stop, result.passes) == (Stop.TIMEOUT, 1)


@pytest.mark.parametrize(
    ("interval", "timeout"), [(-1, 10), (10, -1), (float("nan"), 10)]
)
def test_invalid_limits_are_refused(interval: float, timeout: float) -> None:
    with pytest.raises(ValueError):
        WaitLimits(interval, timeout)


def _parse(argv: list[str]) -> dict[str, object]:
    table = OptionTable(wait_options(group="batch", default_hours=26))
    parser = argparse.ArgumentParser(exit_on_error=False)
    table.add_arguments(parser)
    values = table.values(parser.parse_args(argv))
    resolved, _ = table.by_targets(table.resolve(values))
    return resolved


def test_the_wait_options_default_to_one_pass() -> None:
    assert _parse([]) == {"wait": False, "poll_minutes": 15.0, "wait_hours": 26.0}


def test_the_wait_options_read_their_values() -> None:
    assert _parse(["--wait", "--poll-minutes", "5", "--wait-hours=1.5"]) == {
        "wait": True,
        "poll_minutes": 5.0,
        "wait_hours": 1.5,
    }


@pytest.mark.parametrize("value", ["0", "-2", "nan", "inf", "soon"])
def test_the_wait_values_must_be_positive_numbers(value: str) -> None:
    table = OptionTable(wait_options(group="batch", default_hours=26))
    with pytest.raises(OptionError, match="--poll-minutes"):
        table["poll_minutes"].convert(value)
