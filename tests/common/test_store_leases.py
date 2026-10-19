"""Document leases: acquisition, expiry, late commits, heartbeat and races.

Run on Windows with:

    uv --directory <repo> run pytest tests/common/test_store_leases.py -q

The race test starts two copies of ``tests/common/store_child.py`` with the
test's interpreter and the repository root as working directory.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.sqlite import read, read_rows
from chronominer.common.store import (
    LEASE_TTL,
    Lease,
    LeaseLostError,
    PlannedUnit,
    ResponseRow,
    Store,
    UnitKey,
    UnitResult,
)
from tests.common.store_child import FakeClock

REPO_ROOT = Path(__file__).resolve().parents[2]
CHILD = "tests/common/store_child.py"
CHILD_TIMEOUT = 120.0
MINUTE = 60.0


def _result(idx: int, content: str) -> UnitResult:
    return UnitResult(
        kind="page",
        idx=idx,
        status="done",
        model="model-a",
        provider="openai",
        content=content,
    )


async def _document(store: Store, name: str = "doc") -> int:
    return await store.ensure_document(name)


def test_an_expired_holder_cannot_commit_late_but_keeps_its_responses(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()

    async def main() -> None:
        async with (
            Store(db, "tool", clock=clock) as first,
            Store(db, "tool", clock=clock) as second,
        ):
            await first.begin_run("first")
            await second.begin_run("second")
            doc_id = await _document(first)
            lease = await first.acquire(doc_id)
            assert lease is not None
            assert lease.generation == 1
            await first.plan_units(
                lease, "page", [PlannedUnit(0)], model="model-a", provider="openai"
            )
            await first.write_unit(lease, _result(0, "first draft"))
            assert await second.acquire(doc_id) is None

            clock.advance(11 * MINUTE)

            before = read_rows(db, "units")
            with pytest.raises(LeaseLostError, match="expired"):
                await first.write_unit(lease, _result(0, "late"))
            assert read_rows(db, "units") == before
            with pytest.raises(LeaseLostError):
                await first.renew(lease)
            assert first.held_leases() == {}
            response_id = await first.append_response(
                ResponseRow(
                    provider="openai",
                    model="model-a",
                    unit=UnitKey(doc_id, "page", 0),
                    raw='{"late": true}',
                    raw_kind="http",
                )
            )
            assert response_id > 0

            taken = await second.acquire(doc_id)
            assert taken is not None
            assert taken.generation == 2
            with pytest.raises(LeaseLostError, match="another holder"):
                await first.write_unit(lease, _result(0, "later"))
            assert await first.acquire(doc_id) is None
            await second.write_unit(taken, _result(0, "second"))

    asyncio.run(main())

    assert [row["content"] for row in read_rows(db, "units")] == ["second"]
    responses = read_rows(db, "responses")
    assert [row["raw"] for row in responses] == ['{"late": true}']


def test_release_frees_the_lease_for_another_holder(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool") as first, Store(db, "tool") as second:
            await first.begin_run("first")
            await second.begin_run("second")
            doc_id = await _document(first)
            lease = await first.acquire(doc_id)
            assert lease is not None
            assert await second.acquire(doc_id) is None
            await first.release(lease)
            taken = await second.acquire(doc_id)
            assert taken is not None
            assert taken.generation == 2
            with pytest.raises(KeyError):
                await second.acquire(999)

    asyncio.run(main())


def test_closing_a_store_releases_its_leases(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool") as first:
            await first.begin_run("first")
            doc_id = await _document(first)
            assert await first.acquire(doc_id) is not None
        async with Store(db, "tool") as second:
            await second.begin_run("second")
            assert await second.acquire(doc_id) is not None

    asyncio.run(main())


def test_closing_releases_a_lease_whose_acquisition_was_cancelled(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool") as first:
            await first.begin_run("first")
            doc_id = await _document(first)
            acquiring = asyncio.create_task(first.acquire(doc_id))
            await asyncio.sleep(0)
            acquiring.cancel()
            with pytest.raises(asyncio.CancelledError):
                await acquiring
            assert first.held_leases() == {}
        async with Store(db, "tool") as second:
            await second.begin_run("second")
            assert await second.acquire(doc_id) is not None

    asyncio.run(main())
    (row,) = read_rows(db, "documents")
    assert row["lease_run_id"] is None


def test_the_heartbeat_renews_at_its_interval_and_stops_cleanly(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()

    async def main() -> tuple[int, int, list[float]]:
        async with Store(db, "tool", clock=clock) as store:
            await store.begin_run("run")
            doc_id = await _document(store)
            lease = await store.acquire(doc_id)
            assert lease is not None
            await store.plan_units(
                lease, "page", [PlannedUnit(0)], model="model-a", provider="openai"
            )
            until: list[float] = []
            keeper = store.heartbeat(interval=0.05)
            async with keeper:
                assert keeper.running
                for _ in range(8):
                    clock.advance(4 * MINUTE)
                    await asyncio.sleep(0.15)
                    until.append(store.held_leases()[doc_id].until)
            assert not keeper.running
            renewals = keeper.renewals
            await asyncio.sleep(0.2)
            await store.write_unit(lease, _result(0, "kept alive"))
            return renewals, keeper.renewals, until

    renewals, after_stop, until = asyncio.run(main())

    assert renewals >= 8
    assert after_stop == renewals
    assert until == sorted(until)
    assert clock.now - FakeClock().now > LEASE_TTL
    assert until[-1] > clock.now
    assert [row["content"] for row in read_rows(db, "units")] == ["kept alive"]


def test_the_heartbeat_reports_a_lease_that_expired_between_beats(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()
    lost: list[Lease] = []

    async def main() -> list[Lease]:
        async with Store(db, "tool", clock=clock) as store:
            await store.begin_run("run")
            doc_id = await _document(store)
            lease = await store.acquire(doc_id)
            assert lease is not None
            keeper = store.heartbeat(interval=0.05, on_lost=lost.append)
            clock.advance(11 * MINUTE)
            async with keeper:
                await asyncio.sleep(0.2)
            assert store.held_leases() == {}
            return keeper.lost

    keeper_lost = asyncio.run(main())

    assert len(lost) == 1
    assert keeper_lost == lost


def test_the_heartbeat_skips_a_lease_released_during_its_beat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "tool.sqlite"
    lost: list[Lease] = []

    async def main() -> tuple[list[Lease], int]:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            first = await store.acquire(await _document(store, "first"))
            second = await store.acquire(await _document(store, "second"))
            assert first is not None
            assert second is not None
            renew = store.renew

            async def renew_then_release(lease: Lease) -> Lease:
                renewed = await renew(lease)
                if lease == first:
                    await store.release(second)
                return renewed

            monkeypatch.setattr(store, "renew", renew_then_release)
            keeper = store.heartbeat(on_lost=lost.append)
            await keeper.renew_all()
            return keeper.lost, keeper.renewals

    keeper_lost, renewals = asyncio.run(main())

    assert lost == []
    assert keeper_lost == []
    assert renewals == 1


def test_the_heartbeat_skips_a_lease_whose_release_is_in_flight(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    lost: list[Lease] = []

    async def main() -> tuple[dict[int, Lease], list[Lease]]:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            lease = await store.acquire(await _document(store))
            assert lease is not None
            keeper = store.heartbeat(on_lost=lost.append)
            release = asyncio.create_task(store.release(lease))
            await asyncio.sleep(0)
            await keeper.renew_all()
            await release
            return store.held_leases(), keeper.lost

    held, keeper_lost = asyncio.run(main())

    assert held == {}
    assert lost == []
    assert keeper_lost == []
    assert [row["lease_run_id"] for row in read_rows(db, "documents")] == [None]


@contextlib.contextmanager
def _child(*args: str) -> Iterator[subprocess.Popen[str]]:
    process = subprocess.Popen(
        [sys.executable, CHILD, *args],
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    watchdog = threading.Timer(CHILD_TIMEOUT, process.kill)
    watchdog.start()
    try:
        yield process
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.kill()
        process.communicate()


def _line(process: subprocess.Popen[str]) -> str:
    assert process.stdout is not None
    line = process.stdout.readline().strip()
    if not line:
        process.kill()
        _out, err = process.communicate()
        pytest.fail(f"child printed nothing:\n{err}")
    return line


def _send(process: subprocess.Popen[str], text: str) -> None:
    assert process.stdin is not None
    process.stdin.write(text + "\n")
    process.stdin.flush()


def test_two_processes_acquiring_one_document_get_exactly_one_lease(
    tmp_path: Path,
) -> None:
    db = tmp_path / "race.sqlite"

    async def seed() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("seed")
            await _document(store, "shared")

    asyncio.run(seed())

    with _child("acquire", str(db), "shared") as first:
        assert _line(first) == "READY"
        with _child("acquire", str(db), "shared") as second:
            assert _line(second) == "READY"
            for process in (first, second):
                _send(process, "go")
            results = sorted([_line(first), _line(second)])
            for process in (first, second):
                _send(process, "exit")
            assert first.wait(CHILD_TIMEOUT) == 0
            assert second.wait(CHILD_TIMEOUT) == 0

    assert results == ["ACQUIRED 1", "REFUSED"]
    rows: list[dict[str, Any]] = read_rows(db, "documents")
    assert [row["lease_generation"] for row in rows] == [1]
    assert read(db, lambda c: c.execute("PRAGMA integrity_check").fetchone()[0]) == (
        "ok"
    )
