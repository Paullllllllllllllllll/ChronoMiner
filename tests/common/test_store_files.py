"""Unreadable render targets and blockwise file hashing.

Run on Windows with:

    uv --directory <repo> run pytest tests/common/test_store_files.py -q

The Windows test holds the render target open with share mode 0 through
``_winapi.CreateFile``, so no other handle can read or replace it.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chronominer.common import store as store_module
from chronominer.common.sqlite import read_rows
from chronominer.common.store import (
    Lease,
    PlannedUnit,
    RenderReport,
    RenderTarget,
    Store,
    UnitResult,
    file_sha256,
)


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


async def _document(store: Store) -> Lease:
    doc_id = await store.ensure_document("doc")
    lease = await store.acquire(doc_id)
    assert lease is not None
    await store.plan_units(
        lease, "page", [PlannedUnit(0)], model="model-a", provider="openai"
    )
    await store.write_unit(lease, _result(0, "first"))
    return lease


def _units(db: Path) -> list[dict[str, Any]]:
    return read_rows(db, "units")


def _check_unreadable(
    tmp_path: Path, block: Callable[[Path], None], unblock: Callable[[], None]
) -> None:
    db = tmp_path / "tool.sqlite"
    target = tmp_path / "out" / "doc.txt"

    async def main() -> tuple[RenderReport, list[Any], RenderReport]:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            lease = await _document(store)
            spec = [RenderTarget(target, "txt", _text)]
            await store.render(lease, spec)
            await store.write_unit(lease, _result(0, "second"))
            before = _units(db)
            block(target)
            try:
                blocked = await store.render(lease, spec)
            finally:
                unblock()
            assert _units(db) == before
            outputs = store.outputs(lease.doc_id)
            later = await store.render(lease, spec)
            return blocked, outputs, later

    blocked, outputs, later = asyncio.run(main())
    assert blocked.stale == [target]
    assert blocked.written == []
    assert len(blocked.warnings) == 1
    assert str(target) in blocked.warnings[0]
    assert "data is stored" in blocked.warnings[0]
    assert [row.stale for row in outputs] == [1]
    assert later.written == [target]
    assert target.read_text(encoding="utf-8") == "second\n"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows share modes")
def test_a_target_held_with_share_mode_zero_goes_stale_and_is_kept(
    tmp_path: Path,
) -> None:
    import _winapi

    handles: list[int] = []

    def block(path: Path) -> None:
        handles.append(
            _winapi.CreateFile(
                str(path), _winapi.GENERIC_READ, 0, 0, _winapi.OPEN_EXISTING, 0, 0
            )
        )

    def unblock() -> None:
        while handles:
            _winapi.CloseHandle(handles.pop())

    _check_unreadable(tmp_path, block, unblock)


def test_a_target_the_reader_cannot_read_goes_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = store_module.file_sha256
    denied: list[Path] = []

    def guarded(path: Path) -> str | None:
        if Path(path) in denied:
            raise PermissionError(13, "Permission denied", str(path))
        return original(path)

    monkeypatch.setattr(store_module, "file_sha256", guarded)
    _check_unreadable(tmp_path, denied.append, denied.clear)


def test_file_sha256_hashes_in_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "big.bin"
    data = bytes(range(256)) * (3 * 1024 * 4)
    path.write_bytes(data)
    assert len(data) == 3 * 1024 * 1024

    def refuse(_self: Path) -> bytes:
        raise AssertionError("read_bytes must not be used")

    monkeypatch.setattr(Path, "read_bytes", refuse)
    assert file_sha256(path) == hashlib.sha256(data).hexdigest()
    assert file_sha256(tmp_path / "missing.bin") is None


def test_file_sha256_propagates_a_denied_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "denied.bin"
    path.write_bytes(b"secret")

    def deny(self: Path, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "open", deny)
    with pytest.raises(PermissionError):
        file_sha256(path)
