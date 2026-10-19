"""The SQLite layer under a busy reader, races, a killed writer and a full disk.

Run on Windows with:

    uv --directory <repo> run pytest tests/common/test_sqlite_robustness.py -q

The subprocess tests start ``tests/common/sqlite_child.py`` with the test's
interpreter and the repository root as working directory.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.sqlite import (
    DatabaseFullError,
    SqliteWriter,
    TableSpec,
    read,
    read_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHILD = "tests/common/sqlite_child.py"
ROWS = TableSpec(
    name="rows",
    ddl=(
        "CREATE TABLE IF NOT EXISTS rows ("
        "id INTEGER PRIMARY KEY, label TEXT NOT NULL, seq INTEGER NOT NULL, "
        "payload BLOB)"
    ),
    key=("id",),
)
CHILD_TIMEOUT = 120.0


def _insert(label: str, seq: int) -> Any:
    def write(connection: sqlite3.Connection) -> None:
        connection.execute("INSERT INTO rows (label, seq) VALUES (?, ?)", (label, seq))

    return write


def _labels(db: Path) -> list[tuple[str, int]]:
    return [
        (row["label"], row["seq"]) for row in read_rows(db, "rows", order_by=["id"])
    ]


def _integrity(db: Path) -> str:
    return str(read(db, lambda c: c.execute("PRAGMA integrity_check").fetchone()[0]))


def _seed(db: Path, label: str, count: int) -> None:
    async def main() -> None:
        async with SqliteWriter(db, (ROWS,)) as writer:
            for seq in range(count):
                await writer.transaction(_insert(label, seq))

    asyncio.run(main())


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


def _expect(process: subprocess.Popen[str], word: str) -> None:
    assert process.stdout is not None
    line = process.stdout.readline().strip()
    if line != word:
        process.kill()
        _out, err = process.communicate()
        pytest.fail(f"child printed {line!r} instead of {word!r}:\n{err}")


def test_a_busy_reader_delays_the_commit_with_warnings_and_no_error(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    warnings: list[tuple[Path, float]] = []

    async def main() -> float:
        loop = asyncio.get_running_loop()
        async with SqliteWriter(
            db,
            (ROWS,),
            busy_timeout=0.2,
            warn_interval=0.5,
            on_busy=lambda path, waited: warnings.append((path, waited)),
        ) as writer:
            await writer.transaction(_insert("before", 0))
            reader = sqlite3.connect(db, isolation_level=None)
            reader.execute("BEGIN")
            assert reader.execute("SELECT count(*) FROM rows").fetchone()[0] == 1
            loop.call_later(2.0, lambda: (reader.execute("COMMIT"), reader.close()))
            started = time.monotonic()
            await writer.transaction(_insert("during", 0))
            return time.monotonic() - started

    waited = asyncio.run(main())

    assert waited >= 1.5
    assert _labels(db) == [("before", 0), ("during", 0)]
    assert warnings
    assert all(path == db for path, _ in warnings)
    times = [seconds for _, seconds in warnings]
    assert all(later - earlier >= 0.45 for earlier, later in pairwise(times))


def test_capacity_wait_pauses_until_the_pending_write_committed(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    seen_at_capacity: list[list[tuple[str, int]]] = []

    async def main() -> None:
        async with SqliteWriter(
            db,
            (ROWS,),
            busy_timeout=0.1,
            warn_interval=10.0,
            queue_size=1,
            on_busy=lambda path, waited: None,
        ) as writer:
            await writer.wait_capacity()
            reader = sqlite3.connect(db, isolation_level=None)
            reader.execute("BEGIN")
            reader.execute("SELECT count(*) FROM rows").fetchone()
            first = asyncio.create_task(writer.transaction(_insert("first", 0)))
            await asyncio.sleep(0.3)
            assert writer.pending == 1

            async def second() -> None:
                await writer.wait_capacity()
                seen_at_capacity.append(_labels(db))
                await writer.transaction(_insert("second", 0))

            follower = asyncio.create_task(second())
            await asyncio.sleep(0.5)
            assert not first.done()
            assert not seen_at_capacity
            reader.execute("COMMIT")
            reader.close()
            await asyncio.gather(first, follower)

    asyncio.run(main())

    assert seen_at_capacity == [[("first", 0)]]
    assert _labels(db) == [("first", 0), ("second", 0)]


def test_two_processes_commit_400_transactions_into_one_database(
    tmp_path: Path,
) -> None:
    db = tmp_path / "race.sqlite"
    with _child("race", str(db), "a", "200") as first:
        _expect(first, "READY")
        with _child("race", str(db), "b", "200") as second:
            _expect(second, "READY")
            for process in (first, second):
                assert process.stdin is not None
                process.stdin.write("go\n")
                process.stdin.flush()
            _expect(first, "DONE")
            _expect(second, "DONE")
            assert first.wait(CHILD_TIMEOUT) == 0
            assert second.wait(CHILD_TIMEOUT) == 0

    rows = _labels(db)
    assert len(rows) == 400
    assert sorted(rows) == sorted(
        [("a", seq) for seq in range(200)] + [("b", seq) for seq in range(200)]
    )
    assert _integrity(db) == "ok"
    assert read(db, lambda c: c.execute("PRAGMA journal_mode").fetchone()[0]) == (
        "delete"
    )


def _kill_inside_transaction(db: Path) -> Path:
    journal = db.with_name(db.name + "-journal")
    with _child("hot", str(db)) as process:
        _expect(process, "READY")
        process.kill()
        process.wait(CHILD_TIMEOUT)
    return journal


def test_writer_open_rolls_back_the_hot_journal_of_a_killed_child(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    _seed(db, "before", 10)
    size_before_child = db.stat().st_size

    journal = _kill_inside_transaction(db)

    assert journal.is_file()
    assert db.stat().st_size > size_before_child

    async def reopen() -> None:
        async with SqliteWriter(db, (ROWS,)):
            pass

    asyncio.run(reopen())

    assert not journal.exists()
    assert _integrity(db) == "ok"
    rows = _labels(db)
    assert [label for label, _ in rows] == ["before"] * 10 + ["committed"] * 5
    assert not [label for label, _ in rows if label == "killed"]


def test_read_helper_recovers_the_hot_journal_of_a_killed_child(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    _seed(db, "before", 3)

    journal = _kill_inside_transaction(db)

    assert journal.is_file()
    rows = _labels(db)
    assert [label for label, _ in rows] == ["before"] * 3 + ["committed"] * 5
    assert not journal.exists()
    assert _integrity(db) == "ok"


def test_a_full_database_raises_the_typed_error_and_keeps_earlier_rows(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"

    def overflow(connection: sqlite3.Connection) -> None:
        (pages,) = connection.execute("PRAGMA page_count").fetchone()
        connection.execute(f"PRAGMA max_page_count = {pages + 2}")
        for seq in range(200):
            connection.execute(
                "INSERT INTO rows (label, seq, payload) "
                "VALUES ('overflow', ?, randomblob(4000))",
                (seq,),
            )

    async def main() -> None:
        async with SqliteWriter(db, (ROWS,)) as writer:
            for seq in range(10):
                await writer.transaction(_insert("kept", seq))
            with pytest.raises(DatabaseFullError) as raised:
                await writer.transaction(overflow)
            assert isinstance(raised.value, sqlite3.OperationalError)
            assert isinstance(raised.value.__cause__, sqlite3.Error)

    asyncio.run(main())

    assert _labels(db) == [("kept", seq) for seq in range(10)]
    assert _integrity(db) == "ok"
