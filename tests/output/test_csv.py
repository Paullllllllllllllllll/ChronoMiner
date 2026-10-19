"""The generic CSV renderer: one row per entry, columns from the schema.

Run on Windows with:

    uv --directory <repo> run pytest tests/output/test_csv.py -q

Entries are written with their units and the schema stored as an artifact;
``render_item`` with ``--formats csv`` writes the file, whose bytes are
compared with the documented format.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chronominer.common.store import Lease, Store
from chronominer.output.csv_view import csv_columns
from chronominer.output.render import render_item
from chronominer.output.store import metadata_writer, open_store
from tests.output.test_entries import (
    GRANARY,
    INVENTORY,
    MILL,
    NESTED,
    PROBATE,
    write_chunk,
)
from tests.output.test_store import NAME, chunk, seed_document

BOM = "﻿"


def _render_csv(
    root: Path,
    chunks: Sequence[tuple[int, Sequence[Mapping[str, Any]], str]],
    schema: Mapping[str, Any] | None,
) -> str:
    output = root / f"{NAME}_output.json"

    async def store_schema(store: Store, lease: Lease) -> None:
        sha = None
        if schema is not None:
            sha = await store.store_artifact("schema", json.dumps(schema))
        await store.write(
            lease, metadata_writer(store, lease.doc_id, {}, schema_sha256=sha)
        )

    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("extract")
            lease = await seed_document(
                store, [chunk(index, "pending") for index in range(3)]
            )
            await store_schema(store, lease)
            for idx, entries, status in chunks:
                await write_chunk(
                    store, lease, idx, entries, schema=schema, status=status
                )
            report = await render_item(store, NAME, ("csv",), output, lease)
            assert output.with_suffix(".csv") in report.written

    asyncio.run(main())
    return output.with_suffix(".csv").read_bytes().decode("utf-8")


def test_nested_objects_become_dotted_columns_and_lists_are_joined(
    tmp_path: Path,
) -> None:
    text = _render_csv(tmp_path, [(0, [GRANARY], "done"), (1, [MILL], "done")], NESTED)

    assert text == (
        f"{BOM}name,place.town,place.region,tags,source_page,notes\n"
        "Granary of Basel,Basel,,grain; storage,12,\n"
        "Mill at Liestal,Liestal,Basel-Land,,,Rebuilt after the flood.\n"
    )


def test_lists_of_objects_are_json_and_quoting_is_minimal(tmp_path: Path) -> None:
    inventory = {**INVENTORY, "notes": 'says "Hans", not Johann'}

    text = _render_csv(tmp_path, [(0, [inventory], "done")], PROBATE)

    header, row, end = text.split("\n")
    assert header == (
        f"{BOM}decedent,decedent_occupation,date,place,currency,items,source_page,notes"
    )
    items = json.dumps(INVENTORY["items"], ensure_ascii=False).replace('"', '""')
    assert row == (
        f'Hans Meyer,baker,1712-03-04,Basel,lb,"{items}",f. 3r,'
        '"says ""Hans"", not Johann"'
    )
    assert end == ""


def test_rows_follow_chunk_order_and_skip_failed_and_pending_chunks(
    tmp_path: Path,
) -> None:
    text = _render_csv(
        tmp_path,
        [(2, [MILL], "done"), (1, [], "failed"), (0, [GRANARY, MILL], "done")],
        NESTED,
    )

    names = [line.split(",")[0] for line in text.splitlines()[1:]]
    assert names == ["Granary of Basel", "Mill at Liestal", "Mill at Liestal"]


def test_without_a_schema_the_entry_keys_are_the_columns(tmp_path: Path) -> None:
    entries = [{"a": True, "b": {"c": 1.5}}, {"d": None, "a": False}]

    text = _render_csv(tmp_path, [(0, entries, "done")], None)

    assert text == f"{BOM}a,b.c,d\ntrue,1.5,\nfalse,,\n"


def test_a_document_without_entries_renders_the_header(tmp_path: Path) -> None:
    text = _render_csv(tmp_path, [(0, [], "done")], NESTED)

    assert text == f"{BOM}name,place.town,place.region,tags,source_page,notes\n"


def test_csv_columns_follow_the_schema_order() -> None:
    assert csv_columns(NESTED) == [
        ("name",),
        ("place", "town"),
        ("place", "region"),
        ("tags",),
        ("source_page",),
        ("notes",),
    ]
