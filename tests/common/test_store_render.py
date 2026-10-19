"""The outputs registry and the render driver.

Run on Windows with:

    uv --directory <repo> run pytest tests/common/test_store_render.py -q

The locked-file test holds the target open with ``open()``, which on Windows
shares the file for reading and writing but not for deletion, so the
replacement fails.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.sqlite import read_rows
from chronominer.common.store import (
    Lease,
    LeaseLostError,
    PlannedUnit,
    RenderTarget,
    Store,
    UnitResult,
)
from tests.common.store_child import FakeClock

MINUTE = 60.0


def _text(connection: sqlite3.Connection, doc_id: int) -> str:
    rows = connection.execute(
        "SELECT content FROM units WHERE doc_id = ? ORDER BY idx", (doc_id,)
    ).fetchall()
    return "\n\n".join(str(row[0] or "") for row in rows) + "\n"


def _result(idx: int, content: str) -> UnitResult:
    return UnitResult(
        kind="page",
        idx=idx,
        status="done",
        model="model-a",
        provider="openai",
        content=content,
    )


async def _document(store: Store, contents: list[str]) -> Lease:
    doc_id = await store.ensure_document("doc")
    lease = await store.acquire(doc_id)
    assert lease is not None
    units = [PlannedUnit(idx) for idx in range(len(contents))]
    await store.plan_units(lease, "page", units, model="model-a", provider="openai")
    for idx, content in enumerate(contents):
        await store.write_unit(lease, _result(idx, content))
    return lease


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _temps(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.glob(".*.tmp"))


def test_render_writes_registers_and_skips_unchanged_files(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    target = tmp_path / "out" / "doc.txt"

    async def main() -> list[Any]:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            lease = await _document(store, ["one", "two"])
            spec = [RenderTarget(target, "txt", _text)]
            first = await store.render(lease, spec)
            second = await store.render(lease, spec)
            outputs = store.outputs(lease.doc_id)
            return [first, second, outputs]

    first, second, outputs = asyncio.run(main())

    assert first.written == [target]
    assert second.unchanged == [target]
    assert second.written == []
    assert target.read_bytes() == b"one\n\ntwo\n"
    assert [(row.path, row.format, row.sha256, row.stale) for row in outputs] == [
        ("out/doc.txt", "txt", _sha(target), 0)
    ]
    assert _temps(target.parent) == []


def test_a_hand_edited_file_stays_unless_forced(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    target = tmp_path / "doc.txt"
    edited = b"edited by hand\n"

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            lease = await _document(store, ["one"])
            spec = [RenderTarget(target, "txt", _text)]
            await store.render(lease, spec)
            target.write_bytes(edited)
            await store.write_unit(lease, _result(0, "changed"))

            kept = await store.render(lease, spec)
            assert kept.kept == [target]
            assert kept.written == []
            assert "changed outside the tool" in kept.warnings[0]
            assert target.read_bytes() == edited

            forced = await store.render(lease, spec, force=True)
            assert forced.written == [target]
            assert target.read_bytes() == b"changed\n"
            assert store.outputs(lease.doc_id)[0].sha256 == _sha(target)

            await store.write_unit(lease, _result(0, "again"))
            assert (await store.render(lease, spec)).written == [target]
            assert target.read_bytes() == b"again\n"

    asyncio.run(main())


def test_a_file_edited_while_the_render_waits_stays_unless_forced(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    target = tmp_path / "doc.txt"
    edited = b"edited while the render waited\n"

    def edit() -> None:
        target.write_bytes(edited)

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            lease = await _document(store, ["one"])
            spec = [RenderTarget(target, "txt", _text)]
            await store.render(lease, spec)
            recorded = store.outputs(lease.doc_id)[0].sha256
            await store.write_unit(lease, _result(0, "changed"))

            kept = await store.render(lease, spec, hook=edit)
            assert kept.kept == [target]
            assert kept.written == []
            assert "changed while it was rendered" in kept.warnings[0]
            assert target.read_bytes() == edited
            assert store.outputs(lease.doc_id)[0].sha256 == recorded
            assert _temps(tmp_path) == []

            forced = await store.render(lease, spec, force=True, hook=edit)
            assert forced.written == [target]

    asyncio.run(main())
    assert target.read_bytes() == b"changed\n"


def test_an_existing_file_without_a_row_needs_force(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    target = tmp_path / "doc.txt"
    target.write_bytes(b"legacy\n")

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            lease = await _document(store, ["one"])
            spec = [RenderTarget(target, "txt", _text)]
            assert (await store.render(lease, spec)).kept == [target]
            await store.register_output(lease, target, format="txt")
            assert (await store.render(lease, spec)).written == [target]

    asyncio.run(main())
    assert target.read_bytes() == b"one\n"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file sharing")
def test_a_locked_target_marks_the_output_stale_and_keeps_the_data(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    target = tmp_path / "doc.txt"

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            lease = await _document(store, ["one"])
            spec = [RenderTarget(target, "txt", _text)]
            await store.render(lease, spec)
            await store.write_unit(lease, _result(0, "new data"))
            with open(target, "rb"):
                report = await store.render(lease, spec)
            assert report.stale == [target]
            assert report.written == []
            assert "marked stale" in report.warnings[0]
            assert target.read_bytes() == b"one\n"
            assert store.outputs(lease.doc_id)[0].stale == 1
            assert store.units(lease.doc_id)[0].content == "new data"
            assert _temps(tmp_path) == []

            retry = await store.render(lease, spec)
            assert retry.written == [target]
            assert store.outputs(lease.doc_id)[0].stale == 0

    asyncio.run(main())
    assert target.read_bytes() == b"new data\n"


def test_a_takeover_before_the_render_transaction_replaces_nothing(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    target = tmp_path / "doc.txt"
    clock = FakeClock()

    async def main() -> None:
        async with (
            Store(db, "tool", clock=clock) as first,
            Store(db, "tool", clock=clock) as second,
        ):
            await first.begin_run("first")
            await second.begin_run("second")
            lease = await _document(first, ["one"])
            spec = [RenderTarget(target, "txt", _text)]
            await first.render(lease, spec)
            clock.advance(11 * MINUTE)
            assert await second.acquire(lease.doc_id) is not None
            await second.write_unit(
                (second.held_leases()[lease.doc_id]), _result(0, "second")
            )
            with pytest.raises(LeaseLostError):
                await first.render(lease, spec)

    asyncio.run(main())

    assert target.read_bytes() == b"one\n"
    assert _temps(tmp_path) == []
    assert [row["sha256"] for row in read_rows(db, "outputs")] == [_sha(target)]


def test_a_takeover_waits_for_the_render_holding_the_write_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "tool.sqlite"
    target = tmp_path / "doc.txt"
    clock = FakeClock()
    locked = threading.Event()
    proceed = threading.Event()
    replaced: list[str] = []
    real_replace = os.replace

    def counting_replace(source: Any, destination: Any) -> None:
        if Path(destination) == target:
            replaced.append(Path(source).name)
        real_replace(source, destination)

    def pause() -> None:
        locked.set()
        assert proceed.wait(30)

    async def main() -> Lease | None:
        async with (
            Store(db, "tool", clock=clock) as first,
            Store(db, "tool", clock=clock) as second,
        ):
            await first.begin_run("first")
            await second.begin_run("second")
            lease = await _document(first, ["one"])
            spec = [RenderTarget(target, "txt", _text)]
            monkeypatch.setattr(os, "replace", counting_replace)
            render = asyncio.create_task(first.render(lease, spec, hook=pause))
            await asyncio.to_thread(locked.wait, 30)
            clock.advance(11 * MINUTE)
            takeover = asyncio.create_task(second.acquire(lease.doc_id))
            await asyncio.sleep(0.5)
            assert not takeover.done()
            assert replaced == []
            proceed.set()
            report = await render
            taken = await takeover
            assert report.written == [target]
            return taken

    taken = asyncio.run(main())

    assert taken is not None
    assert taken.generation == 2
    assert len(replaced) == 1
    assert target.read_bytes() == b"one\n"
    outputs = read_rows(db, "outputs")
    assert [(row["sha256"], row["stale"]) for row in outputs] == [(_sha(target), 0)]
