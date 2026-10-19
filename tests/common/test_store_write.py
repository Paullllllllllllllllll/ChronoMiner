"""The public leased write and renderers that return bytes."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from chronominer.common.sqlite import read_rows
from chronominer.common.store import (
    Component,
    Lease,
    LeaseLostError,
    PlannedUnit,
    RenderTarget,
    Store,
    UnitResult,
)
from tests.common.store_child import FakeClock

MINUTE = 60.0
EXTRA = Component(
    "tool",
    ("CREATE TABLE extra (doc_id INTEGER NOT NULL, text TEXT NOT NULL);",),
)
BINARY = b"PK\x03\x04\x00\xff\xfe binary body\r\n"


def _insert(text: str, lease: Lease) -> Callable[[sqlite3.Connection], int]:
    def func(connection: sqlite3.Connection) -> int:
        connection.execute(
            "INSERT INTO extra (doc_id, text) VALUES (?, ?)", (lease.doc_id, text)
        )
        (count,) = connection.execute("SELECT count(*) FROM extra").fetchone()
        return int(count)

    return func


async def _lease(store: Store) -> Lease:
    doc_id = await store.ensure_document("doc")
    lease = await store.acquire(doc_id)
    assert lease is not None
    return lease


def test_write_commits_the_function_under_the_lease(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> list[int]:
        async with Store(db, "tool", components=(EXTRA,)) as store:
            await store.begin_run("run")
            lease = await _lease(store)
            first = await store.write(lease, _insert("one", lease))
            second = await store.write(lease, _insert("two", lease))
            return [first, second]

    assert asyncio.run(main()) == [1, 2]
    assert [row["text"] for row in read_rows(db, "extra")] == ["one", "two"]


def test_write_rolls_back_when_the_function_raises(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    def failing(connection: sqlite3.Connection) -> None:
        connection.execute("INSERT INTO extra (doc_id, text) VALUES (1, 'lost')")
        raise ValueError("refused")

    async def main() -> None:
        async with Store(db, "tool", components=(EXTRA,)) as store:
            await store.begin_run("run")
            lease = await _lease(store)
            with pytest.raises(ValueError, match="refused"):
                await store.write(lease, failing)
            assert lease.doc_id in store.held_leases()

    asyncio.run(main())
    assert read_rows(db, "extra") == []


def test_write_refuses_an_expired_or_taken_lease(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()
    called: list[str] = []

    def func(connection: sqlite3.Connection) -> None:
        called.append("ran")
        connection.execute("INSERT INTO extra (doc_id, text) VALUES (1, 'late')")

    async def main() -> None:
        async with (
            Store(db, "tool", components=(EXTRA,), clock=clock) as first,
            Store(db, "tool", components=(EXTRA,), clock=clock) as second,
        ):
            await first.begin_run("first")
            await second.begin_run("second")
            lease = await _lease(first)
            clock.advance(11 * MINUTE)
            with pytest.raises(LeaseLostError, match="expired"):
                await first.write(lease, func)
            assert lease.doc_id not in first.held_leases()

            again = await first.acquire(lease.doc_id)
            assert again is not None
            clock.advance(11 * MINUTE)
            assert await second.acquire(lease.doc_id) is not None
            with pytest.raises(LeaseLostError, match="another holder"):
                await first.write(again, func)

    asyncio.run(main())
    assert called == []
    assert read_rows(db, "extra") == []


def test_a_unit_and_its_extra_row_commit_or_roll_back_together(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"

    def failing(connection: sqlite3.Connection) -> None:
        connection.execute("INSERT INTO extra (doc_id, text) VALUES (1, 'lost')")
        raise ValueError("refused")

    async def main() -> None:
        async with Store(db, "tool", components=(EXTRA,)) as store:
            await store.begin_run("run")
            lease = await _lease(store)
            await store.plan_units(
                lease, "page", [PlannedUnit(0)], model="m", provider="p"
            )
            done = UnitResult("page", 0, "done", "m", "p", content="kept")
            await store.write_unit(lease, done, extra=_insert("row", lease))
            failed = UnitResult("page", 0, "done", "m", "p", content="lost")
            with pytest.raises(ValueError, match="refused"):
                await store.write_unit(lease, failed, extra=failing)

    asyncio.run(main())
    assert [row["text"] for row in read_rows(db, "extra")] == ["row"]
    assert [row["content"] for row in read_rows(db, "units")] == ["kept"]


def test_a_bytes_rendering_is_written_as_it_is(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    target = tmp_path / "out" / "doc.docx"
    text_target = tmp_path / "out" / "doc.txt"

    def binary(_connection: sqlite3.Connection, _doc_id: int) -> bytes:
        return BINARY

    def text(_connection: sqlite3.Connection, _doc_id: int) -> str:
        return "text\n"

    async def main() -> list[list[Path]]:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            lease = await _lease(store)
            spec = [
                RenderTarget(target, "docx", binary),
                RenderTarget(text_target, "txt", text),
            ]
            first = await store.render(lease, spec)
            second = await store.render(lease, spec)
            return [first.written, second.unchanged]

    written, unchanged = asyncio.run(main())

    assert written == [target, text_target]
    assert unchanged == [target, text_target]
    assert target.read_bytes() == BINARY
    assert text_target.read_bytes() == b"text\n"
    digests = {row["path"]: row["sha256"] for row in read_rows(db, "outputs")}
    assert digests["out/doc.docx"] == hashlib.sha256(BINARY).hexdigest()
