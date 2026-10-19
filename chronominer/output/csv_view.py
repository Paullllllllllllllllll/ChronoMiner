"""The ``{stem}_output.csv`` view: one row per entry, columns from the schema.

The columns follow the schema's entry object in schema order. A field whose
schema is an object (with properties) becomes one column per property,
named with dots (``place.town``), recursively; every other field is one
column. Cells hold strings as they are, numbers as written by Python,
booleans as ``true`` or ``false``, null and missing values as empty cells,
lists of scalars joined by ``"; "``, and lists of objects or other nested
values as JSON. Rows are the entries of the ``entries`` view in chunk and
entry order. Without a stored schema the columns are the entries' own keys
in first-seen order, nested objects again dotted.

The file is UTF-8 with a byte-order mark (for spreadsheet programs), comma
separated with minimal quoting, one ``\\n`` per row, and holds the header
alone when the document has no entries.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Final

from chronominer.output.entries import entry_schema, object_schema, schema_types
from chronominer.output.store import ENTRY_TABLE, KIND_CHUNK, document_metadata

__all__ = ["CSV_FORMAT", "LIST_SEPARATOR", "csv_columns", "render_csv"]

CSV_FORMAT: Final = "csv"
LIST_SEPARATOR: Final = "; "

_ENTRIES: Final = f"""
SELECT e.entry FROM {ENTRY_TABLE} AS e
JOIN units AS u ON u.doc_id = e.doc_id AND u.kind = e.kind AND u.idx = e.idx
WHERE e.doc_id = ? AND e.kind = ? AND u.status = 'done' AND u.stale = 0
AND u.run_id = e.run_id
ORDER BY e.idx, e.entry_index
"""

ColumnPath = tuple[str, ...]


def _schema_paths(
    spec: Mapping[str, Any], prefix: ColumnPath = ()
) -> Iterator[ColumnPath]:
    for name, prop in spec["properties"].items():
        path = (*prefix, str(name))
        nested = object_schema(prop) if isinstance(prop, Mapping) else None
        if nested is not None and schema_types(prop) == ("object",):
            yield from _schema_paths(nested, path)
        else:
            yield path


def _value_paths(
    value: Mapping[str, Any], prefix: ColumnPath = ()
) -> Iterator[ColumnPath]:
    for key, item in value.items():
        path = (*prefix, str(key))
        if isinstance(item, Mapping) and item:
            yield from _value_paths(item, path)
        else:
            yield path


def csv_columns(
    schema: Mapping[str, Any] | None, entries: Sequence[Mapping[str, Any]] = ()
) -> list[ColumnPath]:
    """Return the column paths of the schema, or of *entries* without one."""
    spec = entry_schema(schema)
    if spec is not None:
        return list(_schema_paths(spec))
    paths: dict[ColumnPath, None] = {}
    for entry in entries:
        paths.update(dict.fromkeys(_value_paths(entry)))
    return list(paths)


def _scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _cell(value: Any) -> str:
    if isinstance(value, list):
        if all(not isinstance(item, dict | list) for item in value):
            return LIST_SEPARATOR.join(_scalar(item) for item in value)
        return json.dumps(value, ensure_ascii=False)
    return _scalar(value)


def _lookup(entry: Mapping[str, Any], path: ColumnPath) -> Any:
    value: Any = entry
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _schema(connection: sqlite3.Connection, doc_id: int) -> Mapping[str, Any] | None:
    _, sha = document_metadata(connection, doc_id)
    if sha is None:
        return None
    row = connection.execute(
        "SELECT text FROM artifacts WHERE sha256 = ?", (sha,)
    ).fetchone()
    if row is None:
        return None
    try:
        value = json.loads(row[0])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, Mapping) else None


def render_csv(connection: sqlite3.Connection, doc_id: int) -> str:
    """Return the CSV view of document *doc_id*."""
    entries: list[Mapping[str, Any]] = []
    for (text,) in connection.execute(_ENTRIES, (doc_id, KIND_CHUNK)):
        value = json.loads(text)
        if isinstance(value, Mapping):
            entries.append(value)
    columns = csv_columns(_schema(connection, doc_id), entries)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([".".join(path) for path in columns])
    for entry in entries:
        writer.writerow([_cell(_lookup(entry, path)) for path in columns])
    return "﻿" + buffer.getvalue()
