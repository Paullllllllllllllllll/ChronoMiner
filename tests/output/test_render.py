"""The render driver: render_item, render_ranges, render_database and preview.

Run on Windows with:

    uv --directory <repo> run pytest tests/output/test_render.py -q

The renderers' own formats are tested in ``test_output_json.py``,
``test_csv.py`` and ``test_line_ranges_render.py``. The locked-file test
holds the target open with ``open()``, which on Windows shares it for
reading and writing but not for deletion, so the replacement fails.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.store import Lease, UnitResult
from chronominer.output.render import (
    FORMAT_CHOICES,
    document_targets,
    output_formats,
    preview_database,
    render_database,
    render_item,
    render_ranges,
    render_targets,
)
from chronominer.output.store import KIND_CHUNK, KIND_RANGE, database_path, open_store
from tests.output.test_store import MODEL, NAME, PROVIDER, Unit, answer, chunk
from tests.output.test_store import seed_document as seed


def _items(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        json.loads(record["response"]["output_text"])["entries"][0]["item"]
        for record in data["records"]
    ]


def test_formats_offer_csv_and_always_render_json(tmp_path: Path) -> None:
    output = tmp_path / "my.file_output.json"

    targets = render_targets(output, ("csv",))

    assert FORMAT_CHOICES == ("csv",)
    assert output_formats(()) == ("json",)
    assert output_formats(("sqlite", "csv")) == ("json", "csv")
    assert [(target.path.name, target.format) for target in targets] == [
        ("my.file_output.json", "json"),
        ("my.file_output.csv", "csv"),
    ]


def test_pending_and_failed_chunks_are_left_out_of_the_records(
    tmp_path: Path,
) -> None:
    output = tmp_path / f"{NAME}_output.json"

    async def main() -> None:
        async with open_store(tmp_path) as store:
            await store.begin_run("extract")
            lease = await seed(
                store, [chunk(0, "pending"), chunk(1), chunk(2, "failed")]
            )
            report = await render_item(store, NAME, ("csv",), output, lease)
            assert report.written == [output, output.with_suffix(".csv")]

    asyncio.run(main())

    data = json.loads(output.read_text(encoding="utf-8"))
    assert [record["chunk_index"] for record in data["records"]] == [2]
    assert data["_chronominer_metadata"]["failed_chunks"] == [3]
    assert data["_chronominer_metadata"]["partial"] is True
    assert output.read_bytes().endswith(b"}\n")


def test_a_document_without_rendered_units_gets_no_file(tmp_path: Path) -> None:
    output = tmp_path / f"{NAME}_output.json"

    async def main() -> None:
        async with open_store(tmp_path) as store:
            await store.begin_run("extract")
            lease = await seed(store, [chunk(0, "pending")])
            report = await render_item(store, lease.doc_id, (), output, lease)
            assert report.written == report.stale == []
            ranges = await render_ranges(store, NAME, tmp_path / "r.txt", lease)
            assert ranges.written == []

    asyncio.run(main())
    assert not output.exists()
    assert not (tmp_path / "r.txt").exists()


def test_render_item_needs_the_lease_of_its_document(tmp_path: Path) -> None:
    async def main() -> None:
        async with open_store(tmp_path) as store:
            await store.begin_run("extract")
            lease = await seed(store, [chunk(0)])
            other = Lease(lease.doc_id + 1, lease.holder, 1, lease.until)
            with pytest.raises(ValueError, match="lease"):
                await render_item(store, NAME, (), tmp_path / "d.json", other)
            with pytest.raises(KeyError):
                await render_item(store, "missing", (), tmp_path / "d.json", lease)

    asyncio.run(main())


def _rewrite(item: str) -> UnitResult:
    return UnitResult(
        kind=KIND_CHUNK,
        idx=0,
        status="done",
        model=MODEL,
        provider=PROVIDER,
        content=answer({"item": item}),
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file sharing")
def test_a_locked_output_is_marked_stale_and_replaced_later(tmp_path: Path) -> None:
    output = tmp_path / f"{NAME}_output.json"

    async def main() -> None:
        async with open_store(tmp_path) as store:
            await store.begin_run("extract")
            lease = await seed(store, [chunk(0)])
            first = await render_item(store, NAME, (), output, lease)
            assert first.written == [output]
            await store.write_unit(lease, _rewrite("corrected"))
            with open(output, "rb"):
                locked = await render_item(store, NAME, (), output, lease)
            assert locked.stale == [output]
            assert locked.written == []
            assert "marked stale" in locked.warnings[0]
            assert _items(output) == ["item 0"]
            assert [row.stale for row in store.outputs(lease.doc_id)] == [1]

            later = await render_item(store, NAME, (), output, lease)
            assert later.written == [output]
            assert [row.stale for row in store.outputs(lease.doc_id)] == [0]

    asyncio.run(main())
    assert _items(output) == ["corrected"]


def _seeded(root: Path) -> Path:
    """Seed document NAME, render its _output.json and return that file."""
    output = root / f"{NAME}_output.json"

    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("extract")
            lease = await seed(store, [chunk(0), chunk(1)])
            await render_item(store, NAME, (), output, lease)

    asyncio.run(main())
    return output


def _change_chunk(root: Path, item: str, *, stale: bool = False) -> None:
    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("extract")
            row = store.document(NAME)
            assert row is not None
            lease = await store.acquire(row.doc_id)
            assert lease is not None
            await store.write_unit(lease, _rewrite(item))

    asyncio.run(main())
    if stale:
        connection = sqlite3.connect(database_path(root))
        try:
            connection.execute("UPDATE outputs SET stale = 1")
            connection.commit()
        finally:
            connection.close()


def test_render_database_rerenders_a_stale_output(tmp_path: Path) -> None:
    output = _seeded(tmp_path)
    _change_chunk(tmp_path, "corrected", stale=True)

    (result,) = asyncio.run(render_database(database_path(tmp_path)))

    assert result.written == [output]
    assert not result.failed
    assert _items(output) == ["corrected", "item 1"]
    reader = open_store(tmp_path)
    (doc,) = reader.documents()
    assert [row.stale for row in reader.outputs(doc.doc_id)] == [0]


def test_render_database_leaves_a_hand_edited_file_without_force(
    tmp_path: Path,
) -> None:
    output = _seeded(tmp_path)
    edited = json.loads(output.read_text(encoding="utf-8"))
    edited["_chronominer_metadata"]["postprocessing"] = {"by": "downstream"}
    output.write_text(json.dumps(edited), encoding="utf-8")
    _change_chunk(tmp_path, "corrected")

    (kept,) = asyncio.run(render_database(database_path(tmp_path)))
    assert kept.kept == [output]
    assert "postprocessing" in output.read_text(encoding="utf-8")

    (forced,) = asyncio.run(render_database(database_path(tmp_path), force=True))
    assert forced.written == [output]
    assert _items(output) == ["corrected", "item 1"]


def test_render_database_adds_the_csv_beside_the_registered_file(
    tmp_path: Path,
) -> None:
    output = _seeded(tmp_path)

    (result,) = asyncio.run(render_database(database_path(tmp_path), formats=("csv",)))

    assert result.unchanged == [output]
    assert result.written == [output.with_suffix(".csv")]
    (again,) = asyncio.run(render_database(database_path(tmp_path)))
    assert again.unchanged == [output, output.with_suffix(".csv")]


def test_render_database_skips_a_document_another_run_holds(tmp_path: Path) -> None:
    _seeded(tmp_path)

    async def hold() -> None:
        store = open_store(tmp_path)
        await store.open()
        await store.begin_run("extract")
        row = store.document(NAME)
        assert row is not None
        assert await store.acquire(row.doc_id) is not None
        await store._require_writer().close()

    asyncio.run(hold())

    (result,) = asyncio.run(render_database(database_path(tmp_path)))

    assert result.skipped is not None and result.failed
    assert result.written == []


def test_preview_database_writes_nothing(tmp_path: Path) -> None:
    output = _seeded(tmp_path)
    _change_chunk(tmp_path, "corrected")
    database = database_path(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    (result,) = preview_database(database, formats=("csv",))

    assert result.written == [output, output.with_suffix(".csv")]
    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


def test_render_database_rerenders_registered_line_ranges(tmp_path: Path) -> None:
    ranges = tmp_path / "docs" / f"{NAME}_line_ranges.txt"

    async def main() -> None:
        async with open_store(tmp_path) as store:
            await store.begin_run("ranges")
            lease = await seed(
                store, [Unit(0, content="[1, 17]", kind=KIND_RANGE)], name=NAME
            )
            report = await render_ranges(store, NAME, ranges, lease)
            assert report.written == [ranges]

    asyncio.run(main())
    ranges.unlink()

    (result,) = asyncio.run(render_database(database_path(tmp_path)))

    assert result.written == [ranges]
    assert ranges.read_text(encoding="utf-8") == "(1, 17)\n"


def test_document_targets_follow_the_registered_outputs(tmp_path: Path) -> None:
    targets = document_targets(tmp_path, "a/b", [])
    assert [(t.path, t.format) for t in targets] == [
        (tmp_path / "a" / "b_output.json", "json")
    ]
    assert document_targets(tmp_path, "a/b", [], chunks=False) == []
