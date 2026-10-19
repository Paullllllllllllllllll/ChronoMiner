"""The store-backed unit pipeline: one committed row per chunk, then progress.

Run on Windows with:

    uv --directory <repo> run pytest tests/extract/test_units.py -q

Each test runs the pipeline on a fake handler against a real store in
``tmp_path`` and reads the rows back through ``mode=ro`` connections.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.sqlite import DatabaseFullError, read, read_rows
from chronominer.common.store import (
    LeaseLostError,
    UnitKey,
    WriterOptions,
    current_unit,
)
from chronominer.events import ChunkDone
from chronominer.extract import ChunkCounter, Item
from chronominer.extract.units import (
    DocumentHandle,
    EntrySchema,
    UnitsFailedError,
    run_unit_stream,
)
from tests.conftest import make_job, make_spec
from tests.extract.conftest import (
    MODEL,
    PROVIDER,
    TOOL,
    FakePayload,
    make_store,
    open_handle,
    payloads,
    stream,
    units,
)
from tests.recording_events import RecordingEvents

USAGE = {"input_tokens": 1500, "output_tokens": 40, "total_tokens": 1540}

SCHEMA = {
    "name": "Prices",
    "schema": {
        "type": "object",
        "properties": {
            "contains_no_content_of_requested_type": {"type": "boolean"},
            "entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "price": {"type": "number"},
                    },
                },
            },
        },
    },
}


def _entries_text(index: int) -> str:
    entries = [{"item": f"wheat {index}", "price": index + 0.5}]
    return json.dumps(
        {"contains_no_content_of_requested_type": False, "entries": entries}
    )


def _answer(payload: FakePayload, text: str | None, **raw: Any) -> Any:
    response = {"output_text": text or "", "usage": dict(USAGE), **raw}
    return (payload, payload.image_name, text, response, payload.index)


class _CheckingEvents(RecordingEvents):
    """Read the reported unit's row through a read-only connection first."""

    def __init__(self, db: Path) -> None:
        super().__init__()
        self.db = db
        self.seen: list[tuple[int, str, int]] = []

    def chunk_done(self, event: ChunkDone) -> None:
        def query(connection: sqlite3.Connection) -> tuple[str, int]:
            (status,) = connection.execute(
                "SELECT status FROM units WHERE idx = ?", (event.chunk,)
            ).fetchone()
            (settled,) = connection.execute(
                "SELECT count(*) FROM units WHERE status != 'pending'"
            ).fetchone()
            return str(status), int(settled)

        status, settled = read(self.db, query)
        self.seen.append((event.chunk, status, settled))
        super().chunk_done(event)


def test_five_units_are_five_commits_each_visible_before_its_progress(
    tmp_path: Path,
) -> None:
    db = tmp_path / "chronominer.sqlite"
    events = _CheckingEvents(db)
    contexts: list[UnitKey | None] = []
    writes: list[int] = []

    async def handler(payload: FakePayload, extractor: Any) -> Any:
        assert extractor == "client"
        contexts.append(current_unit())
        if payload.index == 2:
            return _answer(payload, None, error="model refused the page")
        return _answer(payload, _entries_text(payload.index))

    async def main() -> DocumentHandle:
        async with make_store(db) as store:
            handle = await open_handle(store, pages=5)
            write_unit = store.write_unit

            async def counting(lease: Any, result: Any, **kwargs: Any) -> None:
                writes.append(result.idx)
                await write_unit(lease, result, **kwargs)

            store.write_unit = counting  # type: ignore[method-assign]
            counter = ChunkCounter(events, handle.name, total=5)
            with pytest.raises(UnitsFailedError) as raised:
                await run_unit_stream(
                    handle,
                    stream(payloads(5)),
                    "client",
                    handler,
                    concurrency=1,
                    on_unit=counter,
                    entries=EntrySchema(SCHEMA),
                )
            assert events.seen and len(events.of("chunk_done")) == 5
            error = raised.value
            assert str(error) == "1 of 5 chunk(s) failed to extract for 'doc'"
            assert (error.failed_chunks, error.total_chunks) == (1, 5)
            assert error.write_errors == []
            return handle

    handle = asyncio.run(main())

    assert writes == [0, 1, 2, 3, 4]
    assert events.seen == [
        (0, "done", 1),
        (1, "done", 2),
        (2, "failed", 3),
        (3, "done", 4),
        (4, "done", 5),
    ]
    assert [e.status for e in events.of("chunk_done")] == ["ok"] * 2 + ["failed"] + [
        "ok"
    ] * 2
    assert contexts == [UnitKey(handle.doc_id, "chunk", i) for i in range(5)]
    rows = units(db)
    assert [row["status"] for row in rows] == ["done"] * 2 + ["failed"] + ["done"] * 2
    failed = rows[2]
    assert failed["content"] is None
    assert failed["error"] == "model refused the page"
    assert failed["outcome"] == "failed"
    done = rows[0]
    assert done["content"] == _entries_text(0)
    assert (done["label"], done["source_index"], done["outcome"]) == (
        "page_0001",
        0,
        "ok",
    )
    assert (done["tool"], done["model"], done["provider"]) == (TOOL, MODEL, PROVIDER)
    assert json.loads(done["usage"]) == USAGE
    assert json.loads(done["provenance"]) == {
        "custom_id": "doc-page-1",
        "chunk_range": None,
    }
    assert (done["attempts"], done["response_id"], done["error"]) == (1, None, None)
    assert {row["run_id"] for row in rows} == {handle.run_id}


def test_the_entries_of_a_done_chunk_commit_with_it(tmp_path: Path) -> None:
    db = tmp_path / "chronominer.sqlite"

    async def handler(payload: FakePayload, _extractor: Any) -> Any:
        if payload.index == 1:
            return _answer(payload, "")
        return _answer(payload, f"Here is the JSON:\n{_entries_text(payload.index)}")

    async def main() -> None:
        async with make_store(db) as store:
            handle = await open_handle(store, pages=3)
            with pytest.raises(UnitsFailedError):
                await run_unit_stream(
                    handle,
                    stream(payloads(3)),
                    None,
                    handler,
                    entries=EntrySchema(SCHEMA),
                )

    asyncio.run(main())

    rows = read_rows(db, "entries", order_by=["chunk_index", "entry_index"])
    assert [(row["chunk_index"], row["item"], row["price"]) for row in rows] == [
        (1, "wheat 0", 0.5),
        (3, "wheat 2", 2.5),
    ]
    assert {row["schema"] for row in rows} == {"Prices"}
    statuses = [(row["status"], row["error"]) for row in units(db)]
    assert statuses[1] == (
        "failed",
        "empty model output (possible truncation or refusal)",
    )


def test_no_content_and_unparsed_answers_are_done_without_entries(
    tmp_path: Path,
) -> None:
    db = tmp_path / "chronominer.sqlite"
    answers = {
        0: json.dumps({"contains_no_content_of_requested_type": True, "entries": []}),
        1: '{"entries": [{"item": "rye"',
    }

    async def handler(payload: FakePayload, _extractor: Any) -> Any:
        return _answer(payload, answers[payload.index])

    async def main() -> None:
        async with make_store(db) as store:
            handle = await open_handle(store, pages=2)
            await run_unit_stream(
                handle,
                stream(payloads(2)),
                None,
                handler,
                entries=EntrySchema(SCHEMA),
            )

    asyncio.run(main())

    rows = units(db)
    assert [(row["status"], row["outcome"]) for row in rows] == [
        ("done", "no_content"),
        ("done", "unparsed"),
    ]
    assert rows[1]["content"] == answers[1]
    assert read_rows(db, "entries", order_by=["chunk_index"]) == []


def test_a_full_write_queue_pauses_provider_calls_until_the_commit_lands(
    tmp_path: Path,
) -> None:
    """A reader holding a transaction blocks the commit; the run waits and warns."""
    db = tmp_path / "chronominer.sqlite"
    log: list[str] = []
    warnings: list[float] = []
    readers: list[sqlite3.Connection] = []
    options = WriterOptions(
        busy_timeout=0.05,
        warn_interval=0.0,
        queue_size=1,
        on_busy=lambda _path, waited: warnings.append(waited),
    )

    async def main() -> None:
        async with make_store(db, writer=options) as store:
            handle = await open_handle(store, pages=2)
            first, second = payloads(2)
            gate = asyncio.Event()

            async def producer() -> AsyncIterator[FakePayload]:
                yield first
                await gate.wait()
                yield second

            async def handler(payload: FakePayload, _extractor: Any) -> Any:
                log.append(f"call {payload.index}")
                if payload.index == 0:
                    reader = sqlite3.connect(
                        f"{db.resolve().as_uri()}?mode=ro",
                        uri=True,
                        isolation_level=None,
                    )
                    reader.execute("BEGIN")
                    reader.execute("SELECT count(*) FROM units").fetchone()
                    readers.append(reader)
                    gate.set()
                return _answer(payload, f"Text {payload.index}")

            async def release() -> None:
                while not warnings:
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.3)
                log.append("release")
                readers[0].execute("COMMIT")
                readers[0].close()

            async with asyncio.timeout(30):
                processed, _ = await asyncio.gather(
                    run_unit_stream(handle, producer(), None, handler, concurrency=2),
                    release(),
                )
            assert processed == 2

    asyncio.run(main())

    assert log == ["call 0", "release", "call 1"]
    assert warnings
    assert [row["status"] for row in units(db)] == ["done", "done"]


def test_a_full_database_fails_the_unit_loudly_and_keeps_earlier_units(
    tmp_path: Path,
) -> None:
    db = tmp_path / "chronominer.sqlite"
    events = RecordingEvents()

    async def main() -> UnitsFailedError:
        async with make_store(db) as store:
            handle = await open_handle(store, pages=3)

            async def handler(payload: FakePayload, _extractor: Any) -> Any:
                if payload.index < 2:
                    return _answer(payload, f"Text {payload.index}")
                await store._write(
                    lambda c: c.execute(
                        "PRAGMA max_page_count = "
                        f"{c.execute('PRAGMA page_count').fetchone()[0] + 1}"
                    )
                )
                return _answer(payload, "x" * 200_000)

            with pytest.raises(UnitsFailedError) as raised:
                await run_unit_stream(
                    handle,
                    stream(payloads(3)),
                    None,
                    handler,
                    concurrency=1,
                    on_unit=ChunkCounter(events, handle.name, total=3),
                )
            return raised.value

    error = asyncio.run(main())

    assert isinstance(error.__cause__, DatabaseFullError)
    assert [type(e) for e in error.write_errors] == [DatabaseFullError]
    assert "could not be stored: DatabaseFullError" in str(error)
    assert [event.chunk for event in events.of("chunk_done")] == [0, 1]
    rows = units(db)
    assert [row["status"] for row in rows] == ["done", "done", "pending"]
    assert [row["content"] for row in rows[:2]] == ["Text 0", "Text 1"]
    assert read(db, lambda c: c.execute("PRAGMA integrity_check").fetchone()[0]) == (
        "ok"
    )


def test_a_lost_lease_stops_the_document_and_is_raised(tmp_path: Path) -> None:
    db = tmp_path / "chronominer.sqlite"
    now = [1_000.0]
    calls: list[int] = []

    async def main() -> None:
        async with make_store(db, clock=lambda: now[0]) as store:
            handle = await open_handle(store, pages=3)

            async def handler(payload: FakePayload, _extractor: Any) -> Any:
                calls.append(payload.index)
                if payload.index == 1:
                    now[0] += store.lease_ttl + 1
                return _answer(payload, f"Text {payload.index}")

            with pytest.raises(LeaseLostError):
                await run_unit_stream(
                    handle, stream(payloads(3)), None, handler, concurrency=1
                )

    asyncio.run(main())

    assert calls == [0, 1]
    assert [row["status"] for row in units(db)] == ["done", "pending", "pending"]


def test_todo_and_reset_follow_the_store(tmp_path: Path) -> None:
    db = tmp_path / "chronominer.sqlite"

    async def main() -> list[list[int]]:
        async with make_store(db) as store:
            handle = await open_handle(store, pages=4)
            await handle.write(0, status="done", content="a")
            await handle.write(1, status="failed", content=None)
            seen = [handle.todo(), handle.todo(retry_errors=True)]
            assert await handle.reset([0]) == 1
            seen.append(handle.todo())
            return seen

    assert asyncio.run(main()) == [[2, 3], [1, 2, 3], [0, 2, 3]]


def test_items_and_jobs_default_to_no_store(tmp_path: Path) -> None:
    job = make_job(
        make_spec(tmp_path / "doc.pdf"), output_dir=None, input_root=tmp_path
    )
    item = Item(tmp_path / "doc.pdf", "pdf")

    assert job.store is None
    assert job.retry_errors is True
    assert item.document is None
    assert job.locate(tmp_path / "doc.pdf").output == tmp_path / "doc_output.json"
