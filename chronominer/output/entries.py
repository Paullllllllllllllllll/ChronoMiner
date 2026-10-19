"""The ``entry_rows`` content table: one row per extracted schema entry.

A schema's entry object is the item schema of its ``entries`` array (the
whole object when the schema has none). Each top-level property of it is a
field: a scalar field gets a typed column (string TEXT, integer INTEGER,
number REAL, boolean INTEGER 0 or 1; no declared type when the schema allows
several), an object or array field a TEXT column holding JSON. Columns are
named ``f_<field>`` and added the first time a schema writes entries; field
names that differ only in case share a column. The whole entry is kept as
JSON in ``entry``.

:func:`entries_writer` returns the write that replaces one unit's entries;
pass it as ``extra`` to :meth:`~chronominer.common.store.Store.write_unit`
so the entries commit with the unit. The ``entries`` view shows only the
entries of done, non-stale units written in the unit's current run, so the
entries of a reset or stale unit never surface.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from chronominer.common.store import Store
from chronominer.common.structured import parse_json_object
from chronominer.output.store import (
    ENTRY_FIELD_PREFIX,
    ENTRY_TABLE,
    KIND_CHUNK,
    create_entries_view,
)

__all__ = [
    "EntryField",
    "entries_writer",
    "entry_fields",
    "entry_schema",
    "object_schema",
    "parse_entries",
    "schema_types",
]

_SQL_TYPES: Final[Mapping[str, str]] = {
    "string": "TEXT",
    "integer": "INTEGER",
    "number": "REAL",
    "boolean": "INTEGER",
    "object": "TEXT",
    "array": "TEXT",
}


@dataclass(frozen=True)
class EntryField:
    """A top-level field of a schema's entry object and its column type."""

    name: str
    sql_type: str

    @property
    def column(self) -> str:
        """Return the field's column name in ``entry_rows``."""
        return f"{ENTRY_FIELD_PREFIX}{self.name}"


def schema_types(spec: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the JSON types a property schema allows, without ``null``."""
    found: list[str] = []
    declared = spec.get("type")
    if isinstance(declared, str):
        found.append(declared)
    elif isinstance(declared, list):
        found.extend(str(item) for item in declared)
    for key in ("anyOf", "oneOf"):
        for option in spec.get(key) or ():
            if isinstance(option, Mapping):
                found.extend(schema_types(option))
    if "properties" in spec and "object" not in found:
        found.append("object")
    return tuple(dict.fromkeys(item for item in found if item != "null"))


def object_schema(spec: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the object schema of *spec* or of one of its alternatives."""
    if isinstance(spec.get("properties"), Mapping):
        return spec
    for key in ("anyOf", "oneOf"):
        for option in spec.get(key) or ():
            if isinstance(option, Mapping):
                found = object_schema(option)
                if found is not None:
                    return found
    return None


def entry_schema(schema: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Return the entry object schema of a ChronoMiner schema, or None.

    *schema* is the schema file's JSON (``{"name", "schema": {...}}``) or a
    bare JSON schema.
    """
    if not schema:
        return None
    body = schema.get("schema") if isinstance(schema.get("schema"), Mapping) else schema
    if not isinstance(body, Mapping):
        return None
    root = object_schema(body)
    if root is None:
        return None
    entries = root["properties"].get("entries")
    if isinstance(entries, Mapping) and isinstance(entries.get("items"), Mapping):
        return object_schema(entries["items"])
    return root


def entry_fields(schema: Mapping[str, Any] | None) -> tuple[EntryField, ...]:
    """Return the fields of the schema's entry object, in schema order."""
    spec = entry_schema(schema)
    if spec is None:
        return ()
    fields = []
    for name, prop in spec["properties"].items():
        types = schema_types(prop) if isinstance(prop, Mapping) else ()
        sql_type = _SQL_TYPES.get(types[0], "") if len(types) == 1 else ""
        fields.append(EntryField(str(name), sql_type))
    return tuple(fields)


def parse_entries(output_text: str | None) -> list[dict[str, Any]]:
    """Return the entries of a model answer; an empty list when there are none.

    The answer may be wrapped in a code fence or prose. An answer that sets
    ``contains_no_content_of_requested_type``, has no ``entries`` list or
    cannot be parsed has no entries; entries that are not objects are
    dropped.
    """
    parsed = parse_json_object(output_text or "")
    if parsed is None or parsed.get("contains_no_content_of_requested_type"):
        return []
    entries = parsed.get("entries")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _cell(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False)
    if value is None or isinstance(value, int | float | str):
        return value
    return str(value)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _ensure_columns(
    connection: sqlite3.Connection, fields: Sequence[EntryField]
) -> dict[str, str]:
    """Add missing field columns; return the column of each field name."""
    existing = {
        str(row[1]).casefold(): str(row[1])
        for row in connection.execute(f"PRAGMA table_info({ENTRY_TABLE})")
    }
    columns: dict[str, str] = {}
    added = False
    for field in fields:
        key = field.column.casefold()
        if key not in existing:
            declared = f" {field.sql_type}" if field.sql_type else ""
            connection.execute(
                f"ALTER TABLE {ENTRY_TABLE} ADD COLUMN {_quote(field.column)}{declared}"
            )
            existing[key] = field.column
            added = True
        columns[field.name] = existing[key]
    if added:
        create_entries_view(connection)
    return columns


def _keys_of(entries: Sequence[Mapping[str, Any]]) -> tuple[EntryField, ...]:
    names = dict.fromkeys(str(key) for entry in entries for key in entry)
    return tuple(EntryField(name, "") for name in names)


def entries_writer(  # noqa: PLR0913 - unit key and row fields of extract and batch
    store: Store,
    doc_id: int,
    idx: int,
    entries: Sequence[Mapping[str, Any]],
    *,
    model: str,
    provider: str,
    schema: Mapping[str, Any] | None = None,
    schema_name: str | None = None,
    kind: str = KIND_CHUNK,
) -> Callable[[sqlite3.Connection], None]:
    """Return the write that replaces the entries of unit *idx* of *doc_id*.

    *schema* (the schema file's JSON) gives the field columns; without it
    the entries' own keys do, untyped. *schema_name* defaults to the
    schema's ``name``. The store's run must have begun. Writing no entries
    clears the unit's rows.
    """
    run_id = store.run_id
    if run_id is None:
        raise RuntimeError("Begin a run before writing entries.")
    fields = entry_fields(schema) or _keys_of(entries)
    name = schema_name
    if name is None and schema is not None:
        name = str(schema.get("name") or "") or None
    rows = [dict(entry) for entry in entries]
    tool = store.tool

    def write(connection: sqlite3.Connection) -> None:
        connection.execute(
            f"DELETE FROM {ENTRY_TABLE} WHERE doc_id = ? AND kind = ? AND idx = ?",
            (doc_id, kind, idx),
        )
        if not rows:
            return
        columns = _ensure_columns(connection, fields)
        pairs: dict[str, str] = {}
        for field in fields:
            pairs.setdefault(columns[field.name].casefold(), field.name)
        names = list(pairs.values())
        fixed = (
            "doc_id, kind, idx, entry_index, schema_name, run_id, tool, model, "
            "provider, entry"
        )
        extra = "".join(f", {_quote(columns[field])}" for field in names)
        marks = ", ".join("?" for _ in range(10 + len(names)))
        sql = f"INSERT INTO {ENTRY_TABLE} ({fixed}{extra}) VALUES ({marks})"
        for position, entry in enumerate(rows):
            values = [
                doc_id,
                kind,
                idx,
                position,
                name,
                run_id,
                tool,
                model,
                provider,
                json.dumps(entry, ensure_ascii=False),
                *(_cell(entry.get(field)) for field in names),
            ]
            connection.execute(sql, values)

    return write
