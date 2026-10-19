"""A reader holding a transaction during a run: the run waits, warns, completes.

Run on Windows with (``<repo>`` is the repository root):

    uv --directory <repo> run pytest tests/core/test_run_busy.py -q

When the first chunk reaches the scripted model, a second thread opens the
run's database, starts a read transaction and holds it for about 3 s. Under
the rollback journal every commit of the run waits until that read ends. The
busy timeout and the warning interval are patched short, so the run warns
while it waits; it must still finish every chunk without a failed unit.

The second test blocks the run's reads. When the item starts, a reader holds
a transaction for three busy-timeout cycles and another program's write queues
behind it; a writer waiting to commit refuses new reads, so the item's first
read is busy for longer than one busy timeout. The busy timeout and warning
interval keep the production ratio (1 s and 2 s against 30 s and 60 s); the
read must wait, warn and finish the item without a failed unit.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import chronominer.run as core
from chronominer.common.contract import ExitCode, exit_code_for
from chronominer.common.sqlite import read_rows
from chronominer.common.testing import FakeRequest
from chronominer.events import ItemStarted
from chronominer.output.store import DATABASE_NAME
from tests.conftest import make_settings, make_spec
from tests.llm import helpers
from tests.recording_events import RecordingEvents

CHUNKS = 10
HOLD_SECONDS = 3.0
RATIO_BUSY_TIMEOUT = 1.0
RATIO_WARN_INTERVAL = 2.0
QUEUED_HOLD_SECONDS = 3 * RATIO_BUSY_TIMEOUT
OTHER_NOTE = "written by another program"
ANSWER: dict[str, Any] = {
    "contains_no_content_of_requested_type": False,
    "entries": [],
}


def _text(path: Path, chunks: int) -> Path:
    """Write a text with *chunks* line ranges beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{n}. John Carter, mason, 2s 6d\n" for n in range(chunks)),
        encoding="utf-8",
    )
    path.with_name(f"{path.stem}_line_ranges.txt").write_text(
        "".join(f"({n}, {n})\n" for n in range(1, chunks + 1)), encoding="utf-8"
    )
    return path


class Reader(threading.Thread):
    """Hold a read transaction on *db* for *seconds* once started."""

    def __init__(self, db: Path, seconds: float) -> None:
        super().__init__(daemon=True)
        self.db = db
        self.seconds = seconds
        self.go = threading.Event()
        self.holding = threading.Event()
        self.released_at: float | None = None
        self.error: BaseException | None = None

    def run(self) -> None:
        self.go.wait(30)
        try:
            connection = sqlite3.connect(
                f"{self.db.resolve().as_uri()}?mode=ro", uri=True, timeout=30
            )
            try:
                connection.execute("BEGIN")
                connection.execute("SELECT count(*) FROM units").fetchone()
                self.holding.set()
                time.sleep(self.seconds)
                connection.execute("COMMIT")
            finally:
                connection.close()
        except BaseException as exc:  # noqa: BLE001 - the test asserts on it
            self.error = exc
            self.holding.set()
        self.released_at = time.monotonic()


def test_a_reader_holding_a_transaction_delays_but_never_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(core, "BUSY_TIMEOUT", 0.1)
    monkeypatch.setattr(core, "BUSY_WARN_INTERVAL", 0.3)
    out = tmp_path / "out"
    db = out / DATABASE_NAME
    reader = Reader(db, HOLD_SECONDS)
    reader.start()

    def answer(request: FakeRequest) -> Any:
        if not reader.go.is_set():
            reader.go.set()
            assert reader.holding.wait(10)
        return ANSWER

    helpers.install(monkeypatch, [answer])
    spec = make_spec(
        _text(tmp_path / "in" / "ledger.txt", CHUNKS),
        output=out,
        chunking="line-ranges",
        concurrency=2,
    )
    run_plan = core.plan(spec, make_settings(rate_limits=((100_000, 1),)))
    events = RecordingEvents()

    started = time.monotonic()
    result = asyncio.run(core.run(run_plan, events))
    finished = time.monotonic()
    reader.join(10)

    assert reader.error is None
    assert reader.released_at is not None
    busy = [message for message in events.of("warning") if "is busy" in message]
    assert busy, events.of("warning")
    assert all(str(db) in message for message in busy)
    assert finished - started >= HOLD_SECONDS - 0.5
    assert exit_code_for(failed=result.failed) == ExitCode.OK
    assert (result.complete, result.failed) == (1, 0)
    units = read_rows(db, "units", order_by=["idx"])
    assert [row["status"] for row in units if row["kind"] == "chunk"] == [
        "done"
    ] * CHUNKS
    assert events.of("item_finished")[0].success


class OtherWriter(threading.Thread):
    """Another program's write, queued behind the reader until it releases."""

    def __init__(self, db: Path) -> None:
        super().__init__(daemon=True)
        self.db = db
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            connection = sqlite3.connect(self.db, timeout=60, isolation_level=None)
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO notes (level, message, created_at) "
                    "VALUES ('info', ?, '2026-10-11T00:00:00.000+00:00')",
                    (OTHER_NOTE,),
                )
                connection.execute("COMMIT")
            finally:
                connection.close()
        except BaseException as exc:  # noqa: BLE001 - the test asserts on it
            self.error = exc


def _reads_refused(db: Path) -> bool:
    """Return True while a writer waiting to commit refuses new reads."""
    probe = sqlite3.connect(
        f"{db.resolve().as_uri()}?mode=ro", uri=True, timeout=0, isolation_level=None
    )
    try:
        probe.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except sqlite3.OperationalError as exc:
        return "locked" in str(exc)
    finally:
        probe.close()
    return False


class BlockingEvents(RecordingEvents):
    """When the item starts, block new reads of *db* for a few busy timeouts."""

    def __init__(self, db: Path) -> None:
        super().__init__()
        self.reader = Reader(db, QUEUED_HOLD_SECONDS)
        self.writer = OtherWriter(db)
        self.blocked_at: float | None = None

    def item_started(self, event: ItemStarted) -> None:
        super().item_started(event)
        self.reader.start()
        self.reader.go.set()
        assert self.reader.holding.wait(10)
        self.writer.start()
        deadline = time.monotonic() + 10
        while not _reads_refused(self.reader.db):
            assert time.monotonic() < deadline, "the other write never queued"
            time.sleep(0.01)
        self.blocked_at = time.monotonic()


def test_a_read_behind_a_queued_write_waits_and_the_item_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(core, "BUSY_TIMEOUT", RATIO_BUSY_TIMEOUT)
    monkeypatch.setattr(core, "BUSY_WARN_INTERVAL", RATIO_WARN_INTERVAL)
    out = tmp_path / "out"
    db = out / DATABASE_NAME
    helpers.install(monkeypatch, [ANSWER])
    spec = make_spec(
        _text(tmp_path / "in" / "ledger.txt", 2),
        output=out,
        chunking="line-ranges",
        concurrency=2,
        context="none",
    )
    run_plan = core.plan(spec, make_settings(rate_limits=((100_000, 1),)))
    events = BlockingEvents(db)

    result = asyncio.run(core.run(run_plan, events))
    finished = time.monotonic()
    events.reader.join(10)
    events.writer.join(10)

    assert (events.reader.error, events.writer.error) == (None, None)
    assert events.blocked_at is not None and events.reader.released_at is not None
    assert finished - events.blocked_at >= QUEUED_HOLD_SECONDS - 0.5
    busy = [message for message in events.of("warning") if "is busy" in message]
    assert busy, events.of("warning")
    assert all(str(db) in message for message in busy)
    assert exit_code_for(failed=result.failed) == ExitCode.OK
    assert (result.complete, result.failed) == (1, 0)
    units = read_rows(db, "units", order_by=["idx"])
    assert [row["status"] for row in units if row["kind"] == "chunk"] == [
        "done",
        "done",
    ]
    assert events.of("item_finished")[0].success
    notes = [row["message"] for row in read_rows(db, "notes")]
    assert OTHER_NOTE in notes
