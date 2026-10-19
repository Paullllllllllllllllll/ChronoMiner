"""batch check --wait: passes, sleeps, the deadline and the store-based totals.

Run on Windows with:

    uv --directory <repo> run pytest tests/batch/test_check_wait.py -q

A fake clock whose ``sleep`` advances it replaces the wait's clock and
sleep, and can change the fake provider batches between passes. The store
keeps real time, so its leases stay valid.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chronominer.batch import check as check_module
from chronominer.batch import wait as wait_module
from chronominer.batch.check import (
    INTERRUPTED,
    CheckResult,
    WaitLedger,
    check_until_settled,
    final_totals,
)
from chronominer.batch.clients import BatchClients
from chronominer.batch.finalize import FAILED, FINALIZED_OUTCOME, PENDING
from chronominer.batch.submit import submit_batch
from chronominer.batch.wait import wait_for_batches
from chronominer.common.batch import BatchHandle, BatchStatus
from chronominer.common.polling import PollResult, Stop, WaitLimits
from chronominer.output.store import open_store
from chronominer.settings import RetrySettings, Settings
from chronominer.spec import WAIT_HOURS, resolve_batch_check
from tests.batch.conftest import World, leased_document, page_images, run_spec
from tests.batch.test_reconcile import row, submit_lost
from tests.recording_events import RecordingEvents

OUTPUT = "doc_output.json"
LIMITS = WaitLimits(900, 26 * 3600)


class ApiError(Exception):
    """A provider error carrying an HTTP status."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"Error code: {status_code} - provider error")
        self.status_code = status_code


class Clock:
    """A monotonic clock whose ``sleep`` advances it and runs a scripted step.

    *steps* maps the number of a sleep (1-based) to a callable run during it.
    """

    def __init__(self, steps: dict[int, Callable[[], None]] | None = None) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []
        self.steps = steps or {}

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        step = self.steps.get(len(self.sleeps))
        if step is not None:
            step()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """Run the wait of batch check on a fake clock."""
    fake = Clock()
    monkeypatch.setattr(check_module, "_clock", fake)
    monkeypatch.setattr(check_module, "_sleep", fake.sleep)
    return fake


@pytest.fixture
def closed(monkeypatch: pytest.MonkeyPatch) -> list[BatchClients]:
    """Record every batch clients object the wait closes."""
    seen: list[BatchClients] = []
    real = BatchClients.close

    def close(self: BatchClients) -> None:
        seen.append(self)
        real(self)

    monkeypatch.setattr(BatchClients, "close", close)
    return seen


async def submit(root: Path, provider: str = "openai") -> str:
    """Submit the two page chunks of document ``doc``; return the batch id."""
    root.mkdir(parents=True, exist_ok=True)
    async with open_store(root) as store:
        await store.begin_run("extract")
        handle = await leased_document(store, provider=provider)
        outcome = await submit_batch(
            page_images(2),
            document=handle,
            spec=run_spec(root, provider),
            settings=Settings(),
        )
    (submission_id,) = outcome.submission_ids
    async with open_store(root) as store:
        batch_id = store.submission(submission_id).batch_id
    assert batch_id is not None
    return batch_id


def complete(world: World, batch_id: str) -> Callable[[], None]:
    def step() -> None:
        world.batches[batch_id].status = BatchStatus.COMPLETED

    return step


def wait(
    root: Path,
    settings: Settings | None = None,
    limits: WaitLimits = LIMITS,
    **options: Any,
) -> tuple[PollResult[CheckResult], WaitLedger]:
    ledger = WaitLedger()
    polled = asyncio.run(
        check_until_settled(root, settings or Settings(), limits, ledger, **options)
    )
    return polled, ledger


def test_one_wait_finalizes_and_renders(
    world: World, tmp_path: Path, clock: Clock, closed: list[BatchClients]
) -> None:
    world.status = BatchStatus.IN_PROGRESS
    batch_id = asyncio.run(submit(tmp_path))
    clock.steps[2] = complete(world, batch_id)
    shown: list[tuple[int, bool]] = []

    polled, ledger = wait(
        tmp_path, on_pass=lambda number, _r, pending: shown.append((number, pending))
    )

    assert (polled.stop, polled.passes, ledger.polls) == (Stop.SETTLED, 3, 3)
    assert clock.sleeps == [900, 900]
    assert ledger.stopped == "settled"
    assert ledger.elapsed() == 1800
    assert (tmp_path / OUTPUT).is_file()
    stats = final_totals(ledger)
    assert (stats.finalized, stats.pending, stats.failed) == (1, 0, 0)
    assert shown == [(1, True), (2, False), (3, False)]
    assert len(closed) == 1


def test_adopt_acts_in_the_first_pass_only(
    sdk: dict[str, Any], tmp_path: Path, clock: Clock
) -> None:
    anthropic = sdk["anthropic"]
    submission_id = asyncio.run(submit_lost(tmp_path, "anthropic", sdk))
    (lost,) = anthropic.records
    clock.steps[1] = lambda: anthropic.complete(lost.id)

    polled, ledger = wait(tmp_path, adopt=lost.id)

    assert (polled.stop, polled.passes) == (Stop.SETTLED, 2)
    assert [check.outcome for check in ledger.checks] == [PENDING, FINALIZED_OUTCOME]
    assert "adopted batch" in (ledger.checks[0].detail or "")
    assert row(tmp_path, submission_id) == ("finalized", lost.id)


def test_an_unresolved_and_a_pending_submission_wait_for_the_pending_one(
    sdk: dict[str, Any], tmp_path: Path, clock: Clock
) -> None:
    (tmp_path / "a").mkdir()
    asyncio.run(submit_lost(tmp_path / "a", "anthropic", sdk))
    batch_id = asyncio.run(submit(tmp_path / "b"))
    clock.steps[1] = lambda: sdk["openai"].complete(batch_id)

    polled, ledger = wait(tmp_path)

    assert (polled.stop, polled.passes) == (Stop.SETTLED, 2)
    stats = final_totals(ledger)
    assert (stats.finalized, stats.unresolved, stats.pending) == (1, 1, 0)
    assert stats.had_failure


def test_an_unresolved_submission_alone_stops_after_one_pass(
    sdk: dict[str, Any], tmp_path: Path, clock: Clock
) -> None:
    asyncio.run(submit_lost(tmp_path, "anthropic", sdk))

    polled, ledger = wait(tmp_path)

    assert (polled.stop, polled.passes) == (Stop.SETTLED, 1)
    assert clock.sleeps == []
    assert final_totals(ledger).unresolved == 1


def test_a_non_retryable_status_error_fails_without_waiting(
    world: World, tmp_path: Path, clock: Clock
) -> None:
    asyncio.run(submit(tmp_path))
    world.errors[("openai", "status")] = [ApiError(401)]

    polled, ledger = wait(tmp_path, Settings(retry=RetrySettings(max_attempts=1)))

    assert (polled.stop, polled.passes) == (Stop.SETTLED, 1)
    stats = final_totals(ledger)
    assert (stats.failed, stats.pending) == (1, 0)
    assert not (tmp_path / OUTPUT).exists()


def test_a_failure_stays_unless_the_store_shows_it_finalized(
    world: World, tmp_path: Path, clock: Clock
) -> None:
    world.status = BatchStatus.IN_PROGRESS
    batch_ids = [asyncio.run(submit(tmp_path / name)) for name in "abc"]
    world.errors[("openai", "status")] = [ApiError(401), ApiError(401)]

    def between() -> None:
        world.errors[("openai", "status")] = [ApiError(401)]
        for batch_id in batch_ids[1:]:
            complete(world, batch_id)()

    clock.steps[1] = between

    polled, ledger = wait(tmp_path, Settings(retry=RetrySettings(max_attempts=1)))

    assert (polled.stop, polled.passes) == (Stop.SETTLED, 2)
    assert [(c.database.parent.name, c.outcome) for c in ledger.checks] == [
        ("a", FAILED),
        ("b", FAILED),
        ("c", PENDING),
        ("a", FAILED),
        ("b", FINALIZED_OUTCOME),
        ("c", FINALIZED_OUTCOME),
    ]
    stats = final_totals(ledger)
    assert (stats.failed, stats.finalized, stats.pending) == (1, 2, 0)
    assert row(tmp_path / "a", ledger.checks[0].submission_id)[0] == "submitted"


def test_a_pass_that_raises_keeps_what_it_reported(
    world: World, tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(submit(tmp_path / "a"))
    asyncio.run(submit(tmp_path / "b"))
    real = check_module._check_database
    raised: list[int] = []

    async def flaky(database: Any, *args: Any, **kwargs: Any) -> None:
        if database.root.name == "b" and not raised:
            raised.append(1)
            raise ConnectionError("connection reset by peer")
        await real(database, *args, **kwargs)

    monkeypatch.setattr(check_module, "_check_database", flaky)
    retries: list[tuple[int, float | None]] = []

    polled, ledger = wait(
        tmp_path,
        on_retry=lambda number, _exc, next_in: retries.append((number, next_in)),
    )

    assert (polled.stop, polled.passes, ledger.polls) == (Stop.SETTLED, 2, 2)
    assert retries == [(1, 900)]
    assert [check.database.parent.name for check in ledger.checks] == ["a", "b"]
    stats = final_totals(ledger)
    assert (stats.finalized, stats.pending) == (2, 0)


def test_the_deadline_leaves_the_submission_submitted(
    world: World, tmp_path: Path, clock: Clock, closed: list[BatchClients]
) -> None:
    world.status = BatchStatus.IN_PROGRESS
    asyncio.run(submit(tmp_path))
    shown: list[bool] = []

    polled, ledger = wait(
        tmp_path,
        limits=WaitLimits(900, 2000),
        on_pass=lambda _n, _r, pending: shown.append(pending),
    )

    assert (polled.stop, polled.passes) == (Stop.TIMEOUT, 4)
    assert clock.sleeps == [900, 900, 200]
    assert shown == [True, False, False, True]
    (check,) = ledger.seen.values()
    assert row(tmp_path, check.submission_id)[0] == "submitted"
    assert final_totals(ledger).pending == 1
    assert len(closed) == 1


def test_the_default_wait_outlasts_the_longest_batch_window() -> None:
    spec = resolve_batch_check({"wait": True}).spec
    assert spec.wait_hours == WAIT_HOURS == 50
    assert spec.wait_hours > 48


def test_an_expired_batch_ends_the_default_wait_failed(
    world: World, tmp_path: Path, clock: Clock
) -> None:
    world.status = BatchStatus.IN_PROGRESS
    batch_id = asyncio.run(submit(tmp_path))

    def expire() -> None:
        world.batches[batch_id].status = BatchStatus.EXPIRED

    clock.steps[2] = expire

    polled, ledger = wait(tmp_path, limits=WaitLimits(900, WAIT_HOURS * 3600))

    assert (polled.stop, polled.passes) == (Stop.SETTLED, 3)
    assert clock.sleeps == [900, 900]
    stats = final_totals(ledger)
    assert (stats.failed, stats.pending) == (1, 0)
    assert row(tmp_path, ledger.checks[-1].submission_id)[0] == "failed"


def test_an_interrupt_keeps_the_totals_and_closes_the_clients(
    world: World, tmp_path: Path, clock: Clock, closed: list[BatchClients]
) -> None:
    world.status = BatchStatus.IN_PROGRESS
    asyncio.run(submit(tmp_path))

    def interrupt() -> None:
        raise KeyboardInterrupt

    clock.steps[1] = interrupt
    ledger = WaitLedger()

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(check_until_settled(tmp_path, Settings(), LIMITS, ledger))
    ledger.stopped = INTERRUPTED

    assert ledger.polls == 1
    assert final_totals(ledger).pending == 1
    assert len(closed) == 1


@pytest.fixture
def wait_clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """Run the repair wait on a fake clock."""
    fake = Clock()
    monkeypatch.setattr(wait_module, "_clock", fake)
    monkeypatch.setattr(wait_module, "_sleep", fake.sleep)
    return fake


async def test_the_repair_wait_stops_at_its_timeout(
    world: World, wait_clock: Clock
) -> None:
    batch_id = world.add("openai", status=BatchStatus.IN_PROGRESS)
    events = RecordingEvents()

    outcome = await wait_for_batches(
        [BatchHandle("openai", batch_id)],
        BatchClients(Settings(), "extract"),
        events,
        poll_seconds=10,
        timeout_seconds=25,
    )

    assert (outcome.running, outcome.error) == ([batch_id], None)
    assert wait_clock.sleeps == [10, 10, 5]
    assert world.count("openai", "status") == 4
    assert [event.seconds for event in events.of("waiting")] == [10, 10, 5]
    assert events.of("waiting")[2].reason == "1 batch(es) still running (20s elapsed)"
