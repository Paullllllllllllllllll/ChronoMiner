"""Lease expiry is judged when a write runs, by the clock inside its transaction.

Run on Windows with:

    uv --directory <repo> run pytest tests/common/test_store_busy_lease.py -q

A second ``sqlite3`` connection holds a read transaction, so the store's
writer cannot commit until it ends. The clock is injected and advanced past
the lease TTL while the writes wait.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from chronominer.common.sqlite import read_rows
from chronominer.common.store import (
    LEASE_TTL,
    Lease,
    LeaseLostError,
    PlannedUnit,
    ResponseRow,
    Store,
    UnitResult,
    WriterOptions,
)
from tests.common.store_child import FakeClock

WAIT = 10.0


def _result(idx: int, content: str) -> UnitResult:
    return UnitResult(
        kind="page",
        idx=idx,
        status="done",
        model="model-a",
        provider="openai",
        content=content,
    )


def _store(db: Path, clock: FakeClock, busy: list[float]) -> Store:
    options = WriterOptions(
        busy_timeout=0.05, on_busy=lambda _path, waited: busy.append(waited)
    )
    return Store(db, "tool", clock=clock, writer=options)


async def _lease(store: Store, units: int = 3) -> Lease:
    await store.begin_run("run")
    doc_id = await store.ensure_document("doc")
    lease = await store.acquire(doc_id)
    assert lease is not None
    planned = [PlannedUnit(idx) for idx in range(units)]
    await store.plan_units(lease, "page", planned, model="model-a", provider="openai")
    return lease


def _reader(db: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db, isolation_level=None)
    connection.execute("BEGIN")
    connection.execute("SELECT count(*) FROM units").fetchall()
    return connection


async def _until(condition: Callable[[], bool]) -> None:
    async def poll() -> None:
        while not condition():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), WAIT)


def _start(coroutine: Awaitable[object]) -> asyncio.Task[object]:
    async def run() -> object:
        return await coroutine

    return asyncio.create_task(run())


def test_a_renewal_and_a_write_that_start_after_expiry_fail_but_responses_stay(
    tmp_path: Path,
) -> None:
    """The first write checked its lease inside a transaction begun before the
    expiry and holds the write lock while its commit waits, so it commits;
    the renewal and the write queued behind it start after the expiry."""
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()
    busy: list[float] = []

    async def main() -> list[object]:
        async with _store(db, clock, busy) as store:
            lease = await _lease(store)
            reader = _reader(db)
            try:
                first = _start(store.write_unit(lease, _result(0, "before")))
                renewal = _start(store.renew(lease))
                second = _start(store.write_unit(lease, _result(1, "after")))
                row = ResponseRow(
                    provider="openai", model="model-a", raw="{}", raw_kind="http"
                )
                response = _start(store.append_response(row))
                await asyncio.sleep(0)
                await _until(lambda: bool(busy))
                clock.advance(LEASE_TTL + 60.0)
            finally:
                reader.execute("COMMIT")
                reader.close()
            return list(
                await asyncio.gather(
                    first, renewal, second, response, return_exceptions=True
                )
            )

    first, renewal, second, response = asyncio.run(main())

    assert busy
    assert first is None
    assert isinstance(renewal, LeaseLostError)
    assert isinstance(second, LeaseLostError)
    assert isinstance(response, int)
    contents = {row["idx"]: row["content"] for row in read_rows(db, "units")}
    assert contents == {0: "before", 1: None, 2: None}
    assert len(read_rows(db, "responses")) == 1


def test_a_write_submitted_after_expiry_without_a_renewal_still_fails(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()
    busy: list[float] = []

    async def main() -> None:
        async with _store(db, clock, busy) as store:
            lease = await _lease(store)
            clock.advance(LEASE_TTL + 1.0)
            with pytest.raises(LeaseLostError, match="expired"):
                await store.write_unit(lease, _result(0, "late"))
            with pytest.raises(LeaseLostError):
                await store.renew(lease)

    asyncio.run(main())

    assert [row["content"] for row in read_rows(db, "units")] == [None] * 3


def test_a_takeover_committed_first_fails_the_queued_writes(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()
    other_clock = FakeClock(clock.now + LEASE_TTL + 60.0)
    busy: list[float] = []
    other_busy: list[float] = []

    async def main() -> tuple[object, object, object]:
        async with (
            _store(db, clock, busy) as store,
            _store(db, other_clock, other_busy) as other,
        ):
            lease = await _lease(store)
            await other.begin_run("other")
            reader = _reader(db)
            try:
                takeover = _start(other.acquire(lease.doc_id))
                await _until(lambda: bool(other_busy))
                write = _start(store.write_unit(lease, _result(0, "queued")))
                renewal = _start(store.renew(lease))
                await _until(lambda: bool(busy))
            finally:
                reader.execute("COMMIT")
                reader.close()
            results = await asyncio.gather(
                takeover, write, renewal, return_exceptions=True
            )
            return results[0], results[1], results[2]

    taken, write, renewal = asyncio.run(main())

    assert isinstance(taken, Lease)
    assert taken.generation == 2
    assert isinstance(write, LeaseLostError)
    assert isinstance(renewal, LeaseLostError)
    assert [row["content"] for row in read_rows(db, "units")] == [None] * 3
