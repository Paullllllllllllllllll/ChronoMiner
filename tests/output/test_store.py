"""ChronoMiner's store component: views, naming and image settings.

Run on Windows with:

    uv --directory <repo> run pytest tests/output/test_store.py -q

The view oracles query ``chunks`` and ``ranges`` through a ``mode=ro``
connection over a store seeded through the store API and compare the
documented columns and rows, failed and pending units included.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.native import image_settings_fingerprint
from chronominer.common.store import (
    Lease,
    PlannedUnit,
    Store,
    UnitResult,
    VersionlessDatabaseError,
)
from chronominer.output.store import (
    COMPONENT,
    DATABASE_NAME,
    KIND_CHUNK,
    KIND_RANGE,
    TOOL,
    VIEWS,
    database_path,
    document_metadata,
    document_name,
    image_settings_record,
    metadata_writer,
    open_store,
    output_path,
    read_only_connection,
    read_store,
)
from chronominer.ranges.decision import BoundaryDecision, RangeResult
from chronominer.ranges.store import (
    lease_document,
    plan_ranges,
    replace_ranges,
    write_failure,
    write_result,
)

MODEL = "gpt-5.6-luna"
PROVIDER = "openai"
NAME = "ledger"
USAGE = {"input_tokens": 1500, "output_tokens": 400, "total_tokens": 1900}

CHUNKS_COLUMNS = (
    "document",
    "chunk_index",
    "source_index",
    "label",
    "status",
    "model",
    "provider",
    "usage",
    "output_text",
    "error",
    "error_type",
    "custom_id",
    "stale",
    "run_id",
)

RANGES_COLUMNS = (
    "document",
    "range_index",
    "original_start",
    "original_end",
    "adjusted_start",
    "adjusted_end",
    "should_delete",
    "status",
)


def answer(*entries: dict[str, Any]) -> str:
    """Return a model answer holding *entries*."""
    return json.dumps(
        {"contains_no_content_of_requested_type": False, "entries": list(entries)}
    )


@dataclass(frozen=True)
class Unit:
    """One unit to seed: its state and content."""

    idx: int
    status: str = "done"
    content: str | None = None
    label: str | None = None
    source_index: int | str | None = None
    model: str = MODEL
    kind: str = KIND_CHUNK
    error: str | None = None
    usage: dict[str, int] | None = None
    provenance: dict[str, Any] | None = None


def chunk(idx: int, status: str = "done", **changes: Any) -> Unit:
    """Return a text chunk unit of 17 lines with the content its status implies."""
    start = idx * 17 + 1
    values: dict[str, Any] = {
        "idx": idx,
        "status": status,
        "content": answer({"item": f"item {idx}"}) if status == "done" else None,
        "source_index": f"{start}-{start + 16}",
        "error": "Error code: 400" if status == "failed" else None,
        "usage": USAGE if status == "done" else None,
        "provenance": {
            "custom_id": f"{NAME}-chunk-{idx + 1}",
            "chunk_range": [start, start + 16],
        },
    }
    values.update(changes)
    return Unit(**values)


def result(unit: Unit) -> UnitResult:
    """Return the unit result that writes *unit*."""
    return UnitResult(
        kind=unit.kind,
        idx=unit.idx,
        status=unit.status,
        model=unit.model,
        provider=PROVIDER,
        content=unit.content,
        label=unit.label,
        source_index=unit.source_index,
        usage=unit.usage,
        error=unit.error,
        error_type="api_error" if unit.status == "failed" else None,
        provenance=unit.provenance,
    )


async def seed_document(
    store: Store,
    units: Sequence[Unit],
    *,
    name: str = NAME,
    kind: str = "text",
    image_settings: dict[str, Any] | None = None,
    order: Sequence[int] | None = None,
) -> Lease:
    """Create document *name*, plan *units* as pending and write the others.

    *order* gives the positions in *units* to write, in that order.
    """
    doc_id = await store.ensure_document(name, input_path=f"{name}.txt", kind=kind)
    lease = await store.acquire(doc_id)
    assert lease is not None
    for unit_kind in dict.fromkeys(unit.kind for unit in units):
        planned = [
            PlannedUnit(unit.idx, label=unit.label, source_index=unit.source_index)
            for unit in units
            if unit.kind == unit_kind
        ]
        await store.plan_units(
            lease, unit_kind, planned, model=MODEL, provider=PROVIDER
        )
    if image_settings is not None:
        await store.set_image_settings(lease, image_settings)
    positions = order if order is not None else range(len(units))
    for position in positions:
        unit = units[position]
        if unit.status != "pending":
            await store.write_unit(lease, result(unit))
    return lease


def build_store(
    root: Path,
    units: Sequence[Unit],
    *,
    name: str = NAME,
    image_settings: dict[str, Any] | None = None,
) -> Path:
    """Seed one document in *root*'s store; return the database path."""

    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("extract")
            await seed_document(store, units, name=name, image_settings=image_settings)
            await store.end_run(0)

    asyncio.run(main())
    return database_path(root)


def query(database: Path, sql: str) -> tuple[tuple[str, ...], list[tuple[Any, ...]]]:
    """Run *sql* on a ``mode=ro`` connection; return the columns and rows."""
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        cursor = connection.execute(sql)
        columns = tuple(description[0] for description in cursor.description)
        return columns, cursor.fetchall()
    finally:
        connection.close()


def test_the_component_adds_tables_and_views_at_version_1(tmp_path: Path) -> None:
    database = build_store(tmp_path, [chunk(0)])

    _, meta = query(database, "SELECT component, version FROM meta ORDER BY 1")
    _, views = query(
        database, "SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name"
    )
    _, tables = query(
        database,
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name IN ('entry_rows', 'document_meta') ORDER BY name",
    )

    assert database.name == DATABASE_NAME
    assert COMPONENT.name == TOOL
    assert meta == [("chronominer", 1), ("core", 2)]
    assert sorted(name for (name,) in views) == sorted(VIEWS)
    assert tables == [("document_meta",), ("entry_rows",)]


def test_chunks_returns_every_chunk_unit_in_order(tmp_path: Path) -> None:
    text = [chunk(0), chunk(1, "failed"), chunk(2), chunk(3, "pending")]
    images = [
        Unit(
            0,
            content=answer(),
            label="a.png",
            source_index="a.png",
            provenance={"custom_id": "scans-page-1"},
        )
    ]
    ranges = [Unit(0, content="[1, 17]", kind=KIND_RANGE)]

    async def main() -> str:
        async with open_store(tmp_path) as store:
            run_id = await store.begin_run("extract")
            await seed_document(store, [*text, *ranges], order=[2, 1, 0, 3, 4])
            await seed_document(store, images, name="scans", kind="images")
            return run_id

    run_id = asyncio.run(main())
    columns, rows = query(database_path(tmp_path), "SELECT * FROM chunks")

    assert columns == CHUNKS_COLUMNS
    usage = json.dumps(USAGE)
    first, third = answer({"item": "item 0"}), answer({"item": "item 2"})
    assert rows == [
        (NAME, 1, "1-17", None, "done", MODEL, PROVIDER, usage, first, None, None,
         "ledger-chunk-1", 0, run_id),
        (NAME, 2, "18-34", None, "failed", MODEL, PROVIDER, None, None,
         "Error code: 400", "api_error", "ledger-chunk-2", 0, run_id),
        (NAME, 3, "35-51", None, "done", MODEL, PROVIDER, usage, third, None, None,
         "ledger-chunk-3", 0, run_id),
        (NAME, 4, "52-68", None, "pending", MODEL, PROVIDER, None, None, None, None,
         None, 0, run_id),
        ("scans", 1, "a.png", "a.png", "done", MODEL, PROVIDER, None, answer(),
         None, None, "scans-page-1", 0, run_id),
    ]  # fmt: skip


def decided(
    index: int, original: tuple[int, int], adjusted: tuple[int, int], **changes: Any
) -> RangeResult:
    """Return a boundary decision on range *index* (1-based)."""
    return RangeResult(
        index=index,
        original=original,
        adjusted=adjusted,
        should_delete=changes.get("should_delete", False),
        decision=BoundaryDecision.from_payload({"certainty": 90}),
    )


def test_ranges_returns_every_range_unit_with_its_decision(tmp_path: Path) -> None:
    starting = [(1, 17), (18, 34), (35, 51), (52, 68), (69, 85)]
    for name in (NAME, "notes"):
        (tmp_path / f"{name}.txt").write_text("line\n" * 85, encoding="utf-8")
    calls: dict[str, Any] = {"model": MODEL, "provider": PROVIDER}

    async def main() -> None:
        async with open_store(tmp_path) as store:
            await store.begin_run("ranges")
            await seed_document(store, [chunk(0)], name="book")
            lease = await lease_document(store, tmp_path / f"{NAME}.txt", NAME)
            await plan_ranges(store, lease, starting, **calls)
            await write_result(store, lease, decided(1, (1, 17), (3, 17)), **calls)
            deleted = decided(2, (18, 34), (18, 34), should_delete=True)
            await write_result(store, lease, deleted, **calls)
            await write_failure(store, lease, 2, (35, 51), TimeoutError(), **calls)
            degraded = decided(4, (52, 68), (52, 68))
            await write_result(store, lease, degraded, degraded=True, **calls)
            await store.release(lease)
            await replace_ranges(store, tmp_path / "notes.txt", [(1, 40)], model=MODEL)

    asyncio.run(main())
    columns, rows = query(database_path(tmp_path), "SELECT * FROM ranges")

    assert columns == RANGES_COLUMNS
    assert rows == [
        (NAME, 1, 1, 17, 3, 17, 0, "done"),
        (NAME, 2, 18, 34, 18, 34, 1, "done"),
        (NAME, 3, 35, 51, None, None, None, "failed"),
        (NAME, 4, 52, 68, 52, 68, 0, "failed"),
        (NAME, 5, 69, 85, None, None, None, "pending"),
        ("notes", 1, 1, 40, None, None, None, "done"),
    ]


def test_read_helpers_open_short_read_only_connections(tmp_path: Path) -> None:
    assert read_store(tmp_path) is None
    with pytest.raises(FileNotFoundError), read_only_connection(tmp_path):
        pass

    build_store(tmp_path, [chunk(0), chunk(1)])
    store = read_store(tmp_path)
    assert store is not None
    document = store.document(NAME)
    assert document is not None and document.status == "complete"
    with read_only_connection(tmp_path) as connection:
        count = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM notes")
    assert count == 2


def test_store_helpers_refuse_a_versionless_database_as_not_a_store(
    tmp_path: Path,
) -> None:
    connection = sqlite3.connect(tmp_path / DATABASE_NAME)
    connection.execute("CREATE TABLE documents (name TEXT PRIMARY KEY)")
    connection.execute("CREATE TABLE pages (document TEXT, page_index INTEGER)")
    connection.commit()
    connection.close()
    refused = "is not a chronominer store"

    with pytest.raises(VersionlessDatabaseError, match=refused) as caught:
        read_store(tmp_path)
    assert "migrate" not in str(caught.value)
    with (
        pytest.raises(VersionlessDatabaseError, match=refused),
        read_only_connection(tmp_path),
    ):
        pass

    async def main() -> None:
        async with open_store(tmp_path):
            pass

    with pytest.raises(VersionlessDatabaseError, match=refused):
        asyncio.run(main())


@pytest.mark.parametrize(
    ("output", "root", "expected"),
    [
        ("out/ledger_output.json", "out", "ledger"),
        ("out/sub/deep/beta_output.json", "out", "sub/deep/beta"),
        ("docs/ledger_line_ranges.txt", "docs", "ledger"),
        ("docs/ledger.txt", "docs", "ledger"),
        ("elsewhere/my.file_output.json", "out", "my.file"),
    ],
)
def test_document_name_is_the_output_relative_to_the_root(
    tmp_path: Path, output: str, root: str, expected: str
) -> None:
    assert document_name(tmp_path / output, tmp_path / root) == expected


def test_output_path_inverts_document_name(tmp_path: Path) -> None:
    path = output_path(tmp_path, "sub/deep/beta")

    assert path == tmp_path / "sub" / "deep" / "beta_output.json"
    assert document_name(path, tmp_path) == "sub/deep/beta"


def test_image_settings_record_matches_the_provenance_fingerprint() -> None:
    settings = {
        "jpeg_quality": 95,
        "high_target_box": (768, 1536),
        "model_name": MODEL,
        "provider": PROVIDER,
    }

    record = image_settings_record(settings, "openai", 24_000_000)

    assert record["high_target_box"] == [768, 1536]
    assert record["model_type"] == "openai"
    assert image_settings_fingerprint(record) == image_settings_fingerprint(
        {**settings, "max_pixels_per_page": 24_000_000, "model_type": "openai"}
    )


def test_metadata_writer_merges_and_removes_keys(tmp_path: Path) -> None:
    async def main() -> None:
        async with open_store(tmp_path) as store:
            await store.begin_run("extract")
            lease = await seed_document(store, [chunk(0)])
            sha = await store.store_artifact("schema", '{"name": "Flat"}')
            await store.write(
                lease,
                metadata_writer(
                    store,
                    lease.doc_id,
                    {"schema_name": "Flat", "chunk_slice": {"first_n": 2}},
                    schema_sha256=sha,
                ),
            )
            await store.write(
                lease,
                metadata_writer(
                    store, lease.doc_id, {"chunk_slice": None, "model_name": MODEL}
                ),
            )

    asyncio.run(main())

    with read_only_connection(tmp_path) as connection:
        values, sha = document_metadata(connection, 1)
    assert values == {"schema_name": "Flat", "model_name": MODEL}
    assert sha is not None and len(sha) == 64
