"""The entries content table and the entries view.

Run on Windows with:

    uv --directory <repo> run pytest tests/output/test_entries.py -q

Entries are written with their unit through ``Store.write_unit(extra=...)``
and read back through the ``entries`` view on a ``mode=ro`` connection:
typed columns for scalar fields, JSON text for nested ones, columns added
per schema, and no rows for reset, failed or stale units.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.store import Lease, Store, UnitResult
from chronominer.output.entries import (
    EntryField,
    entries_writer,
    entry_fields,
    parse_entries,
)
from chronominer.output.store import KIND_CHUNK, database_path, open_store
from tests.output.test_store import MODEL, NAME, PROVIDER, chunk, query, seed_document

REPO = Path(__file__).resolve().parents[2]
NESTED = json.loads(
    (REPO / "tests" / "characterization" / "schemas" / "nested_entries.json").read_text(
        encoding="utf-8"
    )
)
PROBATE = json.loads(
    (REPO / "schemas" / "probate_inventories.json").read_text(encoding="utf-8")
)

GRANARY = {
    "name": "Granary of Basel",
    "place": {"town": "Basel", "region": None},
    "tags": ["grain", "storage"],
    "source_page": "12",
    "notes": None,
}
MILL = {
    "name": "Mill at Liestal",
    "place": {"town": "Liestal", "region": "Basel-Land"},
    "tags": [],
    "source_page": None,
    "notes": "Rebuilt after the flood.",
}
INVENTORY = {
    "decedent": "Hans Meyer",
    "decedent_occupation": "baker",
    "date": "1712-03-04",
    "place": "Basel",
    "currency": "lb",
    "items": [{"description": "one oak table", "quantity": 1, "appraised_value": 2.5}],
    "source_page": "f. 3r",
    "notes": None,
}


async def write_chunk(
    store: Store,
    lease: Lease,
    idx: int,
    entries: Sequence[Mapping[str, Any]],
    *,
    schema: Mapping[str, Any] | None = NESTED,
    status: str = "done",
) -> None:
    """Write chunk *idx* with *entries* in one transaction."""
    content = json.dumps({"entries": list(entries)}) if status == "done" else None
    await store.write_unit(
        lease,
        UnitResult(
            kind=KIND_CHUNK,
            idx=idx,
            status=status,
            model=MODEL,
            provider=PROVIDER,
            content=content,
        ),
        extra=entries_writer(
            store,
            lease.doc_id,
            idx,
            entries,
            model=MODEL,
            provider=PROVIDER,
            schema=schema,
        ),
    )


def _seed(root: Path, *steps: Any) -> str:
    """Plan three chunks of NAME and run each step on the store and lease."""

    async def main() -> str:
        async with open_store(root) as store:
            run_id = await store.begin_run("extract")
            lease = await seed_document(
                store, [chunk(0, "pending"), chunk(1, "pending"), chunk(2, "pending")]
            )
            for step in steps:
                await step(store, lease)
            return run_id

    return asyncio.run(main())


def _entries(root: Path, columns: str = "*") -> tuple[tuple[str, ...], list[Any]]:
    return query(database_path(root), f"SELECT {columns} FROM entries")


def test_entry_fields_follow_the_schema_with_column_types() -> None:
    assert entry_fields(NESTED) == (
        EntryField("name", "TEXT"),
        EntryField("place", "TEXT"),
        EntryField("tags", "TEXT"),
        EntryField("source_page", "TEXT"),
        EntryField("notes", "TEXT"),
    )
    assert [(f.name, f.sql_type) for f in entry_fields({"schema": {
        "type": "object",
        "properties": {"entries": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "price": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "paid": {"type": "boolean"},
                "mixed": {"type": ["string", "number"]},
            },
        }}},
    }})] == [("count", "INTEGER"), ("price", "REAL"), ("paid", "INTEGER"),
             ("mixed", "")]  # fmt: skip
    assert entry_fields(None) == ()


def test_the_view_returns_typed_and_nested_columns(tmp_path: Path) -> None:
    async def step(store: Store, lease: Lease) -> None:
        await write_chunk(store, lease, 0, [GRANARY, MILL])
        await write_chunk(store, lease, 2, [GRANARY])

    run_id = _seed(tmp_path, step)
    columns, rows = _entries(tmp_path)

    assert columns == (
        "document", "chunk_index", "entry_index", "schema", "name", "place",
        "tags", "source_page", "notes", "model", "provider", "run_id",
    )  # fmt: skip
    place = json.dumps(GRANARY["place"], ensure_ascii=False)
    assert rows[0] == (
        NAME, 1, 0, "NestedEntries", "Granary of Basel", place,
        '["grain", "storage"]', "12", None, MODEL, PROVIDER, run_id,
    )  # fmt: skip
    assert [(row[1], row[2], row[4]) for row in rows] == [
        (1, 0, "Granary of Basel"),
        (1, 1, "Mill at Liestal"),
        (3, 0, "Granary of Basel"),
    ]
    _, nested = _entries(tmp_path, "json_extract(place, '$.town')")
    assert [town for (town,) in nested] == ["Basel", "Liestal", "Basel"]


def test_scalar_columns_keep_their_types(tmp_path: Path) -> None:
    schema = {
        "name": "Typed",
        "schema": {
            "type": "object",
            "properties": {"entries": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "price": {"type": "number"},
                    "paid": {"type": "boolean"},
                    "label": {"type": "string"},
                },
            }}},
        },
    }  # fmt: skip
    entry = {"count": 3, "price": 1.5, "paid": True, "label": "007"}

    async def step(store: Store, lease: Lease) -> None:
        await write_chunk(store, lease, 0, [entry], schema=schema)

    _seed(tmp_path, step)
    _, rows = _entries(
        tmp_path,
        "typeof(count), typeof(price), typeof(paid), typeof(label), paid, label",
    )

    assert rows == [("integer", "real", "integer", "text", 1, "007")]


def test_a_second_schema_adds_its_columns(tmp_path: Path) -> None:
    async def step(store: Store, lease: Lease) -> None:
        await write_chunk(store, lease, 0, [GRANARY])
        await write_chunk(store, lease, 1, [INVENTORY], schema=PROBATE)

    _seed(tmp_path, step)
    columns, rows = _entries(tmp_path, "schema, name, decedent, items, place")

    assert columns == ("schema", "name", "decedent", "items", "place")
    assert rows[0][:3] == ("NestedEntries", "Granary of Basel", None)
    assert rows[1][:3] == ("ProbateInventories", None, "Hans Meyer")
    assert json.loads(rows[1][3]) == INVENTORY["items"]
    assert rows[1][4] == "Basel"


def test_fields_named_like_fixed_columns_are_renamed(tmp_path: Path) -> None:
    entry = {"model": "Ford T", "Document": "deed", "price": 3}

    async def step(store: Store, lease: Lease) -> None:
        await write_chunk(store, lease, 0, [entry], schema=None)

    _seed(tmp_path, step)
    _, rows = _entries(tmp_path, "model, field_model, field_Document, price")

    assert rows == [(MODEL, "Ford T", "deed", 3)]


def test_rewriting_a_unit_replaces_its_entries(tmp_path: Path) -> None:
    async def step(store: Store, lease: Lease) -> None:
        await write_chunk(store, lease, 0, [GRANARY, MILL])
        await write_chunk(store, lease, 0, [MILL])

    _seed(tmp_path, step)
    _, rows = _entries(tmp_path, "entry_index, name")

    assert rows == [(0, "Mill at Liestal")]


def test_reset_failed_and_stale_units_hide_their_entries(tmp_path: Path) -> None:
    async def step(store: Store, lease: Lease) -> None:
        for idx in range(3):
            await write_chunk(store, lease, idx, [GRANARY])
        await store.reset_units(lease, kind=KIND_CHUNK, idxs=[0])
        await store.write_unit(
            lease,
            UnitResult(KIND_CHUNK, 1, "failed", MODEL, PROVIDER, error="timeout"),
        )
        await store.set_units_stale(lease, KIND_CHUNK, [2])

    _seed(tmp_path, step)

    assert _entries(tmp_path)[1] == []
    _, stored = query(database_path(tmp_path), "SELECT count(*) FROM entry_rows")
    assert stored == [(3,)]


def test_entries_of_an_earlier_run_do_not_surface_after_a_plain_rewrite(
    tmp_path: Path,
) -> None:
    async def first(store: Store, lease: Lease) -> None:
        await write_chunk(store, lease, 0, [GRANARY])

    _seed(tmp_path, first)

    async def main() -> None:
        async with open_store(tmp_path) as store:
            await store.begin_run("extract")
            row = store.document(NAME)
            assert row is not None
            lease = await store.acquire(row.doc_id)
            assert lease is not None
            await store.write_unit(
                lease, UnitResult(KIND_CHUNK, 0, "done", MODEL, PROVIDER, content="{}")
            )

    asyncio.run(main())

    assert _entries(tmp_path)[1] == []


def test_entries_roll_back_with_their_unit(tmp_path: Path) -> None:
    async def step(store: Store, lease: Lease) -> None:
        def fail(connection: sqlite3.Connection) -> None:
            entries_writer(
                store,
                lease.doc_id,
                0,
                [GRANARY],
                model=MODEL,
                provider=PROVIDER,
                schema=NESTED,
            )(connection)
            raise RuntimeError("disk full")

        with pytest.raises(RuntimeError):
            await store.write_unit(
                lease,
                UnitResult(KIND_CHUNK, 0, "done", MODEL, PROVIDER, content="{}"),
                extra=fail,
            )

    _seed(tmp_path, step)
    _, stored = query(database_path(tmp_path), "SELECT count(*) FROM entry_rows")
    _, units = query(database_path(tmp_path), "SELECT status FROM chunks")

    assert stored == [(0,)]
    assert units == [("pending",), ("pending",), ("pending",)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"entries": [{"a": 1}, null, "x"]}', [{"a": 1}]),
        ('```json\n{"entries": [{"a": 1}]}\n```', [{"a": 1}]),
        ('Here you go: {"entries": [{"a": 2}]} done', [{"a": 2}]),
        ('{"contains_no_content_of_requested_type": true, "entries": [{"a": 1}]}', []),
        ('{"entries": null}', []),
        ("not json", []),
        (None, []),
    ],
)
def test_parse_entries_tolerates_loosely_shaped_answers(
    text: str | None, expected: list[dict[str, Any]]
) -> None:
    assert parse_entries(text) == expected
