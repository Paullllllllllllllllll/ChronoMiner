"""Waiting for leases their run no longer renews.

- Without held leases nothing is read and nothing waits.
- An unrenewed lease is waited for until it is gone, the first read just
  after its expiry when that comes before the renewal check.
- A renewed lease, or one another holder took, ends the wait for its item.
- The announcement carries the phrase the CLIs forward to stderr.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import pytest

from chronominer.common.stale_leases import (
    RENEWAL_CHECK,
    HeldLease,
    wait_for_stale_leases,
)

NOW = 1_000_000.0


class _Clock:
    def __init__(self) -> None:
        self.now = NOW
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _reads(
    *rounds: Sequence[HeldLease],
) -> tuple[list[int], Callable[[], Sequence[HeldLease]]]:
    calls = [0]

    def read() -> Sequence[HeldLease]:
        calls[0] += 1
        return rounds[min(calls[0], len(rounds)) - 1]

    return calls, read


def _wait(
    held: Sequence[HeldLease], *rounds: Sequence[HeldLease]
) -> tuple[int, _Clock]:
    clock = _Clock()
    calls, read = _reads(*rounds)
    wait_for_stale_leases(held, read, clock=clock, sleep=clock.sleep)
    return calls[0], clock


def test_without_held_leases_nothing_waits() -> None:
    reads, clock = _wait(())
    assert reads == 0 and clock.sleeps == []


def test_the_wait_is_announced_with_the_phrase_the_cli_shows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    lease = HeldLease("a.txt", "dead", NOW + 10)

    with caplog.at_level(logging.WARNING):
        _wait([lease], [])

    (record,) = caplog.records
    assert "waiting until" in record.getMessage().lower()


def test_an_unrenewed_lease_is_waited_for_until_it_is_gone() -> None:
    lease = HeldLease("a.txt", "dead", NOW + 200)

    reads, clock = _wait([lease], [lease], [lease], [])

    assert reads == 3
    assert clock.sleeps == [RENEWAL_CHECK, RENEWAL_CHECK, 200 + 1 - 2 * RENEWAL_CHECK]
    assert clock.now == NOW + 201


def test_a_lease_close_to_expiry_is_read_just_after_it() -> None:
    lease = HeldLease("a.txt", "dead", NOW + 10)

    reads, clock = _wait([lease], [])

    assert (reads, clock.sleeps) == (1, [11.0])


def test_a_renewed_lease_ends_the_wait() -> None:
    lease = HeldLease("a.txt", "live", NOW + 600)

    reads, clock = _wait([lease], [HeldLease("a.txt", "live", NOW + 660)])

    assert (reads, clock.sleeps) == (1, [RENEWAL_CHECK])


def test_a_lease_another_holder_took_ends_the_wait() -> None:
    lease = HeldLease("a.txt", "dead", NOW + 30)

    reads, _clock = _wait([lease], [HeldLease("a.txt", "next", NOW + 30)])

    assert reads == 1


def test_each_item_is_judged_on_its_own() -> None:
    live = HeldLease("live.txt", "live", NOW + 600)
    dead = HeldLease("dead.txt", "dead", NOW + 300)
    renewed = HeldLease("live.txt", "live", NOW + 690)

    reads, clock = _wait([live, dead], *[[renewed, dead]] * 3, [renewed])

    assert reads == 4
    assert clock.now == NOW + 301
