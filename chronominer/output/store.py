"""ChronoMiner's store: its schema component, read views and resume.

Every output root (the output folder, or the input's folder when writing
beside the input) holds one ``chronominer.sqlite``. Its core tables come
from :mod:`chronominer.common.store`; this module adds the tool's component:
the content tables ``entry_rows`` and ``document_meta`` and the views below.

A document is one input item, named by its ``{stem}_output.json`` path
relative to the output root without that suffix (``sub/ledger``). Its units
are:

- ``chunk``: one extraction request. ``idx`` is the 0-based chunk position
  in the whole document (``chunk_index`` = ``idx + 1``); ``source_index`` is
  the line range of a text chunk (``"start-end"``, 1-based, inclusive), the
  file name of an image or the 0-based page index of a PDF page;
- ``range``: one line range of a text document after ``ranges generate`` or
  ``ranges adjust``, ``idx`` its 0-based position in the range list.

Unit fields the renderers read
==============================

A ``chunk`` unit holds the model's answer as written by extraction, sync or
batch:

- ``content``: the output text verbatim (``response.output_text``), JSON
  text under schema and prompted JSON; NULL for a failed unit;
- ``usage``: the token usage of the selected attempt;
- ``model`` and ``provider``: the model that produced the content;
- ``provenance``: a JSON object with ``custom_id`` (default
  ``{name}-chunk-{chunk_index}``, ``-page-`` for image and PDF documents),
  ``chunk_range`` (``[start, end]``; default parsed from ``source_index``),
  ``image_provenance`` (the page image's record, written into the record
  when present) and ``response`` (the record's ``response`` fields other
  than ``output_text``, kept verbatim, for example
  ``{"response_data": {...}}`` of a batch result; default
  ``{"response_data": {"usage", "model", "provider"}, "usage"}``).

A ``range`` unit holds its range as JSON ``[start, end]`` in ``content``;
a unit with NULL content (a deleted or merged range) is left out of the
``_line_ranges.txt`` rendering.

A document's run metadata lives in ``document_meta`` (written with
:func:`metadata_writer`): ``schema_name``, ``model_name``,
``chunking_method``, ``chunking_text_version``, ``document_total_chunks``,
``chunk_slice`` and ``image_provenance``, as the ``_chronominer_metadata``
block of ``_output.json`` shows them, and the SHA-256 of the schema text
stored in ``artifacts`` (kind ``schema``), from which the CSV takes its
columns. The entries of a chunk go to ``entry_rows`` through
:func:`chronominer.output.entries.entries_writer`, in the transaction that
writes the unit.

Read contract
=============

Outside code reads the store only through these views; the tables behind
them may change between versions, the views stay stable.

``entries``
    One row per extracted entry of a done, non-stale chunk unit, ordered by
    document, chunk and entry: ``document`` (the document name),
    ``chunk_index`` (1-based, as in ``_output.json``), ``entry_index``
    (0-based position in the chunk's ``entries`` array), ``schema`` (the
    schema name), then one column per top-level field of the schema's entry
    object (scalar fields typed as TEXT, INTEGER or REAL, booleans as 0 or
    1; nested objects and lists as JSON text; a field named like one of the
    fixed columns appears as ``field_<name>``), then ``model``,
    ``provider`` and ``run_id``. Field columns are added when a schema
    first writes entries; fields of other schemas are NULL.

``chunks``
    One row per chunk unit, ordered by document and chunk: ``document``,
    ``chunk_index`` (1-based), ``source_index``, ``label``, ``status``
    (pending, done or failed), ``model``, ``provider``, ``usage`` (JSON
    token counts), ``output_text`` (the answer as received), ``error``,
    ``error_type``, ``custom_id``, ``stale``, ``run_id``, ``outcome`` (ok,
    no_content, unparsed, failed or skipped_irrelevant) and
    ``decision_probability`` (the relevance the decision pre-pass gave the
    chunk, from the ``chunk_decisions`` table; NULL without one).

``ranges``
    One row per range unit, ordered by document and range: ``document``,
    ``range_index`` (1-based position in the starting ranges),
    ``original_start`` and ``original_end`` (the starting bounds),
    ``adjusted_start``, ``adjusted_end`` and ``should_delete`` (0 or 1) of
    the stored boundary decision, NULL for a range without one (generated,
    not yet decided, or its call raised), and ``status`` (pending, done or
    failed; a decision made without a usable answer is ``failed`` and is
    decided again by the next run).

Open the database read-only and close it after the query, so that a running
extraction never waits on the reader::

    import sqlite3
    from pathlib import Path

    database = Path("out") / "chronominer.sqlite"
    connection = sqlite3.connect(
        f"{database.resolve().as_uri()}?mode=ro", uri=True
    )
    try:
        rows = connection.execute(
            "SELECT chunk_index, entry_index, item, price FROM entries "
            "WHERE document = ? ORDER BY chunk_index, entry_index",
            ("ledger",),
        ).fetchall()
    finally:
        connection.close()
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Any, Final

from chronominer.common.native import MODEL_DERIVED_IMAGE_KEYS, changed_image_settings
from chronominer.common.sqlite import read, read_only
from chronominer.common.store import (
    Component,
    ResumeState,
    SchemaState,
    Store,
    UnitKey,
    WriterOptions,
    check_schema,
    file_sha256,
)

__all__ = [
    "COMPONENT",
    "DATABASE_NAME",
    "ENTRY_COLUMNS",
    "ENTRY_FIELD_PREFIX",
    "ENTRY_TABLE",
    "KIND_CHUNK",
    "KIND_RANGE",
    "LINE_RANGES_SUFFIX",
    "MODEL_DERIVED_KEYS",
    "OUTPUT_SUFFIX",
    "TOOL",
    "UNIT_KINDS",
    "VIEWS",
    "Classification",
    "ImageSettings",
    "ItemState",
    "chunk_decision_rows",
    "classify",
    "clear_chunk_decisions",
    "create_entries_view",
    "database_path",
    "document_metadata",
    "document_name",
    "image_changes",
    "image_settings_record",
    "lease_reason",
    "metadata_writer",
    "open_store",
    "output_path",
    "read_only_connection",
    "read_store",
    "tool_version",
]

TOOL: Final = "chronominer"
DATABASE_NAME: Final = "chronominer.sqlite"
OUTPUT_SUFFIX: Final = "_output.json"
LINE_RANGES_SUFFIX: Final = "_line_ranges.txt"

KIND_CHUNK: Final = "chunk"
KIND_RANGE: Final = "range"
UNIT_KINDS: Final = (KIND_CHUNK, KIND_RANGE)

VIEWS: Final = ("entries", "chunks", "ranges")

ENTRY_TABLE: Final = "entry_rows"
ENTRY_FIELD_PREFIX: Final = "f_"
"""Prefix of the field columns of ``entry_rows``."""

ENTRY_COLUMNS: Final = (
    "document",
    "chunk_index",
    "entry_index",
    "schema",
    "model",
    "provider",
    "run_id",
)
"""The fixed columns of the ``entries`` view."""

_TABLES_V1 = """
CREATE TABLE entry_rows (
    doc_id INTEGER NOT NULL REFERENCES documents (doc_id),
    kind TEXT NOT NULL DEFAULT 'chunk',
    idx INTEGER NOT NULL,
    entry_index INTEGER NOT NULL CHECK (entry_index >= 0),
    schema_name TEXT,
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    tool TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    entry TEXT NOT NULL,
    PRIMARY KEY (doc_id, kind, idx, entry_index)
);
CREATE TABLE document_meta (
    doc_id INTEGER PRIMARY KEY REFERENCES documents (doc_id),
    metadata TEXT NOT NULL,
    schema_sha256 TEXT,
    run_id TEXT REFERENCES runs (run_id),
    updated_at TEXT
);
CREATE VIEW chunks AS
SELECT
    d.name AS document,
    u.idx + 1 AS chunk_index,
    u.source_index AS source_index,
    u.label AS label,
    u.status AS status,
    u.model AS model,
    u.provider AS provider,
    u.usage AS usage,
    u.content AS output_text,
    u.error AS error,
    u.error_type AS error_type,
    json_extract(u.provenance, '$.custom_id') AS custom_id,
    u.stale AS stale,
    u.run_id AS run_id
FROM units AS u
JOIN documents AS d ON d.doc_id = u.doc_id
WHERE u.kind = 'chunk'
ORDER BY d.name, u.idx;
CREATE VIEW ranges AS
SELECT
    d.name AS document,
    u.idx + 1 AS range_index,
    CAST(substr(u.source_index, 1, instr(u.source_index, '-') - 1) AS INTEGER)
        AS original_start,
    CAST(substr(u.source_index, instr(u.source_index, '-') + 1) AS INTEGER)
        AS original_end,
    json_extract(u.provenance, '$.decision.adjusted_range[0]') AS adjusted_start,
    json_extract(u.provenance, '$.decision.adjusted_range[1]') AS adjusted_end,
    json_extract(u.provenance, '$.decision.should_delete') AS should_delete,
    u.status AS status
FROM units AS u
JOIN documents AS d ON d.doc_id = u.doc_id
WHERE u.kind = 'range'
ORDER BY d.name, u.idx;
"""


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _field_alias(name: str) -> str:
    fixed = {column.casefold() for column in ENTRY_COLUMNS}
    return f"field_{name}" if name.casefold() in fixed else name


def create_entries_view(connection: sqlite3.Connection) -> None:
    """(Re)create the ``entries`` view over the current field columns."""
    columns = [
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({ENTRY_TABLE})")
        if str(row[1]).startswith(ENTRY_FIELD_PREFIX)
    ]
    fields = "".join(
        f"    e.{_quote(column)} AS "
        f"{_quote(_field_alias(column[len(ENTRY_FIELD_PREFIX) :]))},\n"
        for column in columns
    )
    connection.execute("DROP VIEW IF EXISTS entries")
    connection.execute(
        "CREATE VIEW entries AS\nSELECT\n"
        "    d.name AS document,\n"
        "    e.idx + 1 AS chunk_index,\n"
        "    e.entry_index AS entry_index,\n"
        "    e.schema_name AS schema,\n"
        f"{fields}"
        "    e.model AS model,\n"
        "    e.provider AS provider,\n"
        "    e.run_id AS run_id\n"
        f"FROM {ENTRY_TABLE} AS e\n"
        "JOIN documents AS d ON d.doc_id = e.doc_id\n"
        "JOIN units AS u ON u.doc_id = e.doc_id AND u.kind = e.kind "
        "AND u.idx = e.idx\n"
        "WHERE u.status = 'done' AND u.stale = 0 AND u.run_id = e.run_id\n"
        "ORDER BY d.name, e.idx, e.entry_index"
    )


def _version_1(connection: sqlite3.Connection) -> None:
    for statement in _TABLES_V1.split(";"):
        if statement.strip():
            connection.execute(statement)
    create_entries_view(connection)


_DECISIONS_V2 = """
CREATE TABLE chunk_decisions (
    doc_id INTEGER NOT NULL REFERENCES documents (doc_id),
    idx INTEGER NOT NULL,
    probability REAL NOT NULL,
    question_sha256 TEXT NOT NULL,
    model TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    run_id TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (doc_id, idx)
);
DROP VIEW chunks;
CREATE VIEW chunks AS
SELECT
    d.name AS document,
    u.idx + 1 AS chunk_index,
    u.source_index AS source_index,
    u.label AS label,
    u.status AS status,
    u.model AS model,
    u.provider AS provider,
    u.usage AS usage,
    u.content AS output_text,
    u.error AS error,
    u.error_type AS error_type,
    json_extract(u.provenance, '$.custom_id') AS custom_id,
    u.stale AS stale,
    u.run_id AS run_id,
    u.outcome AS outcome,
    cd.probability AS decision_probability
FROM units AS u
JOIN documents AS d ON d.doc_id = u.doc_id
LEFT JOIN chunk_decisions AS cd ON cd.doc_id = u.doc_id AND cd.idx = u.idx
WHERE u.kind = 'chunk'
ORDER BY d.name, u.idx
"""


def _version_2(connection: sqlite3.Connection) -> None:
    for statement in _DECISIONS_V2.split(";"):
        if statement.strip():
            connection.execute(statement)


COMPONENT: Final = Component(TOOL, (_version_1, _version_2))
"""The tool's schema component: version 1 adds the content tables and views,
version 2 the ``chunk_decisions`` table and the decision columns of
``chunks``."""


def clear_chunk_decisions(doc_id: int) -> Callable[[sqlite3.Connection], None]:
    """Return a write that deletes the relevance rows of a document's chunks."""

    def write(connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM chunk_decisions WHERE doc_id = ?", (doc_id,))

    return write


def chunk_decision_rows(
    doc_id: int,
    decisions: Sequence[tuple[int, float]],
    *,
    question_sha256: str,
    model: str,
    fingerprint: str,
    run_id: str | None,
) -> Callable[[sqlite3.Connection], None]:
    """Return a write that replaces the relevance of the document's chunks.

    *decisions* holds ``(idx, probability)``; pass the write to
    ``Store.write`` so it runs under the document's lease. The rows belong to
    the document's id, so dropping the document removes them.
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rows = [
        (doc_id, idx, probability, question_sha256, model, fingerprint, run_id, now)
        for idx, probability in decisions
    ]

    def write(connection: sqlite3.Connection) -> None:
        connection.executemany(
            "INSERT OR REPLACE INTO chunk_decisions (doc_id, idx, probability, "
            "question_sha256, model, fingerprint, run_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    return write


_COMPONENTS: Final = (COMPONENT,)


def tool_version() -> str:
    """Return the installed version of the tool, or ``unknown``."""
    try:
        return metadata.version(TOOL)
    except metadata.PackageNotFoundError:
        return "unknown"


def database_path(root: Path) -> Path:
    """Return the database file of output root *root*."""
    return Path(root) / DATABASE_NAME


def open_store(
    root: Path,
    *,
    clock: Callable[[], float] = time.time,
    writer: WriterOptions | None = None,
) -> Store:
    """Return the store of output root *root*, not yet opened.

    ``async with open_store(root) as store`` opens its writer, creating the
    database when missing.
    """
    return Store(
        database_path(root),
        TOOL,
        tool_version=tool_version(),
        components=_COMPONENTS,
        clock=clock,
        writer=writer,
    )


def _schema(root: Path) -> SchemaState:
    return check_schema(database_path(root), _COMPONENTS, tool=TOOL)


def read_store(root: Path) -> Store | None:
    """Return the store of *root* for reads only; None without a database.

    The store is not opened: its read methods use short read-only
    connections. A versionless, newer or unreadable database raises the
    :class:`~chronominer.common.store.StoreError` that refuses it.
    """
    state = _schema(root)
    if not state.exists or state.empty:
        return None
    return Store(state.path, TOOL, components=_COMPONENTS)


@contextlib.contextmanager
def read_only_connection(root: Path) -> Iterator[sqlite3.Connection]:
    """Yield a ``mode=ro`` connection to *root*'s database; close it after.

    The schema is checked first, as for :func:`read_store`;
    ``FileNotFoundError`` without a database.
    """
    state = _schema(root)
    if not state.exists:
        raise FileNotFoundError(f"No database at {state.path}")
    with read_only(state.path) as connection:
        yield connection


def document_name(output: Path, root: Path) -> str:
    """Return the document name of an item: *output* relative to *root*.

    *output* is the item's ``{stem}_output.json`` path (or its
    ``{stem}_line_ranges.txt``); the name drops that suffix, any other path
    its file suffix. Outside *root*, the file name alone is used.
    """
    name = output.name
    for suffix in (OUTPUT_SUFFIX, LINE_RANGES_SUFFIX):
        if name.endswith(suffix) and len(name) > len(suffix):
            base = output.with_name(name[: -len(suffix)])
            break
    else:
        base = output.with_suffix("")
    try:
        return base.relative_to(root).as_posix()
    except ValueError:
        return base.name


def output_path(root: Path, name: str) -> Path:
    """Return the ``_output.json`` path of document *name* of root *root*."""
    return Path(f"{Path(root) / name}{OUTPUT_SUFFIX}")


def _plain(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return *value* as JSON would store it (tuples become lists)."""
    plain = json.loads(json.dumps(dict(value), ensure_ascii=False, default=str))
    return plain if isinstance(plain, dict) else {}


def image_settings_record(
    settings: Mapping[str, Any], model_type: str, max_pixels: int | None = None
) -> dict[str, Any]:
    """Return the image-settings record a document's pages are produced with.

    *settings* are the resolved image settings of the model's section,
    *model_type* its model family. The record holds the values the image
    settings fingerprint covers, in their stored JSON form.
    """
    record = dict(settings)
    if max_pixels is not None:
        record["max_pixels_per_page"] = max_pixels
    record["model_type"] = model_type
    return _plain(record)


def _now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="milliseconds")


def metadata_writer(
    store: Store,
    doc_id: int,
    values: Mapping[str, Any],
    *,
    schema_sha256: str | None = None,
) -> Callable[[sqlite3.Connection], None]:
    """Return a write that merges *values* into a document's run metadata.

    Use it as ``await store.write(lease, metadata_writer(...))``. Keys of
    *values* replace the stored ones; a None value removes its key.
    *schema_sha256* names the schema text in ``artifacts`` and is kept when
    None.
    """
    run_id = store.run_id
    updates = _plain(values)

    def write(connection: sqlite3.Connection) -> None:
        current = _metadata_row(connection, doc_id)
        merged = {**current[0], **updates}
        merged = {key: value for key, value in merged.items() if value is not None}
        sha = schema_sha256 if schema_sha256 is not None else current[1]
        connection.execute(
            "INSERT INTO document_meta (doc_id, metadata, schema_sha256, run_id, "
            "updated_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT (doc_id) DO UPDATE SET "
            "metadata = excluded.metadata, schema_sha256 = excluded.schema_sha256, "
            "run_id = excluded.run_id, updated_at = excluded.updated_at",
            (doc_id, json.dumps(merged, ensure_ascii=False), sha, run_id, _now()),
        )

    return write


def _metadata_row(
    connection: sqlite3.Connection, doc_id: int
) -> tuple[dict[str, Any], str | None]:
    row = connection.execute(
        "SELECT metadata, schema_sha256 FROM document_meta WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    if row is None:
        return {}, None
    try:
        value = json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        value = {}
    return (value if isinstance(value, dict) else {}), row[1]


def document_metadata(
    connection: sqlite3.Connection, doc_id: int
) -> tuple[dict[str, Any], str | None]:
    """Return a document's run metadata and the SHA-256 of its schema text."""
    return _metadata_row(connection, doc_id)


class ItemState(StrEnum):
    """What the store holds for an item, as resume sees it."""

    NONE = "none"
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNSTORED = "unstored"
    BLOCKED = "blocked"
    REFUSED = "refused"


@dataclass(frozen=True)
class Classification:
    """The resume state of one item.

    *units* are the units to process of a partial item. *render* asks a
    complete item to re-render its files without model calls; *missing*
    lists those files absent on disk. *note* records a model change, which
    resumes. *total* counts the stored units of a complete or partial item.
    *leased_by* names the run whose lease blocks the item; it is None for an
    item blocked by a batch submission.
    """

    state: ItemState
    name: str
    doc_id: int | None = None
    units: tuple[UnitKey, ...] = ()
    render: bool = False
    missing: tuple[Path, ...] = ()
    reason: str = ""
    note: str | None = None
    total: int = 0
    leased_by: str | None = None

    @property
    def indices(self) -> tuple[int, ...]:
        """Return the indices of the units to process."""
        return tuple(unit.idx for unit in self.units)


def _nonempty(path: Path) -> bool:
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _path_key(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class _Details:
    counts: dict[str, int]
    models: list[str]
    outputs: dict[str, int]
    schema_sha256: str | None = None
    input_path: str | None = None
    input_sha256: str | None = None


def _details(connection: sqlite3.Connection, doc_id: int, kind: str) -> _Details:
    counts = {
        str(status): int(count)
        for status, count in connection.execute(
            "SELECT status, count(*) FROM units WHERE doc_id = ? AND kind = ? "
            "GROUP BY status",
            (doc_id, kind),
        )
    }
    models = [
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT model FROM units WHERE doc_id = ? AND kind = ? "
            "AND status != 'pending' ORDER BY model",
            (doc_id, kind),
        )
    ]
    outputs = {
        str(path): int(stale)
        for path, stale in connection.execute(
            "SELECT path, stale FROM outputs WHERE doc_id = ?", (doc_id,)
        )
    }
    source = connection.execute(
        "SELECT input_path, input_sha256 FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    return _Details(
        counts,
        models,
        outputs,
        _metadata_row(connection, doc_id)[1],
        *(source if source is not None else (None, None)),
    )


def _current_sha256(path: Path) -> str | None:
    try:
        return file_sha256(path)
    except OSError:
        return None


def _input_changed(name: str, stored: str | None, current: Path) -> str:
    if stored is None or Path(stored) == current:
        return (
            f"The input {current} changed since '{name}' was extracted. "
            "Skipping it; use --force to extract it again."
        )
    return (
        f"'{name}' holds the extraction of another input, {stored}; {current} "
        "maps to the same output document. Skipping it; use --force to replace "
        "it, or choose another --output."
    )


def _model_note(
    name: str,
    earlier: Sequence[str],
    model: str | None,
    recorded: Mapping[str, Any] | None,
) -> str:
    previous = ", ".join(earlier)
    if not previous and recorded is not None:
        previous = str(recorded.get("model_name") or "")
    return (
        f"'{name}' holds chunks extracted with {previous or 'another model'}; "
        f"this run uses {model or 'another model'}."
    )


def _of_kind(units: Sequence[UnitKey], kind: str) -> tuple[UnitKey, ...]:
    return tuple(unit for unit in units if unit.kind == kind)


def lease_reason(holder: str | None, until: float | None) -> str:
    """Name the run that leases a document and when its lease expires.

    *until* is the expiry in seconds since the epoch, shown in local time.
    """
    reason = f"leased by run {holder or 'unknown'}"
    if until is not None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(until))
        reason += f" until {stamp}"
    return reason


def _blocked(
    resume: ResumeState,
    name: str,
    kind: str,
    holder: str | None,
    now: float | None,
) -> Classification | None:
    moment = time.time() if now is None else now
    if (
        resume.lease_run_id is not None
        and resume.lease_run_id != holder
        and resume.lease_until is not None
        and resume.lease_until > moment
    ):
        return Classification(
            ItemState.BLOCKED,
            name,
            resume.doc_id,
            reason=lease_reason(resume.lease_run_id, resume.lease_until),
            leased_by=resume.lease_run_id,
        )
    blocked = _of_kind(resume.blocked, kind)
    if blocked:
        return Classification(
            ItemState.BLOCKED,
            name,
            resume.doc_id,
            reason=(
                f"{len(blocked)} unit(s) await an open or unresolved batch "
                "submission; run batch check"
            ),
        )
    return None


def _input_refusal(
    name: str, doc_id: int, details: _Details, input_path: Path | None
) -> Classification | None:
    if input_path is None or details.input_sha256 is None:
        return None
    content = _current_sha256(input_path)
    if content is None or content == details.input_sha256:
        return None
    return Classification(
        ItemState.REFUSED,
        name,
        doc_id,
        reason=_input_changed(name, details.input_path, input_path),
    )


type ImageSettings = (
    Mapping[str, Any] | Callable[[Mapping[str, Any] | None], Mapping[str, Any] | None]
)
"""A run's image-settings record, or a function that returns it for an item's
stored record (None when the item has none)."""

MODEL_DERIVED_KEYS: frozenset[str] = MODEL_DERIVED_IMAGE_KEYS | {
    "detail",
    "original_max_side_px",
}
"""The record keys a model change may change without refusing the item: the
shared model-derived keys, ``detail`` (a copy of ``resolved_detail``) and the
long-side limit a model's capabilities set."""


def image_changes(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Return the changed image settings of two records, by key.

    A model change alone changes nothing (:data:`MODEL_DERIVED_KEYS`).
    """
    return changed_image_settings(
        _plain(recorded), _plain(current), derived=MODEL_DERIVED_KEYS
    )


def _image_changes(
    resume: ResumeState, image_settings: ImageSettings | None
) -> tuple[dict[str, Any] | None, list[str], bool]:
    """Compare a resumed document's recorded image settings with the current ones.

    Returns:
        The recorded settings, the changed user settings, and whether the
        records differ in other fields only.
    """
    stored = resume.image_settings
    recorded = _plain(stored) if isinstance(stored, Mapping) else None
    if callable(image_settings):
        image_settings = image_settings(recorded)
    current = _plain(image_settings) if image_settings is not None else None
    if recorded is None or current is None:
        return recorded, [], False
    changed = image_changes(recorded, current)
    return recorded, changed, not changed and recorded != current


def _schema_changed(details: _Details, schema_sha256: str | None) -> bool:
    stored = details.schema_sha256
    return (
        schema_sha256 is not None
        and stored is not None
        and stored != schema_sha256
        and details.counts.get("done", 0) > 0
    )


def classify(  # noqa: PLR0913 - keyword-only options of run.py and tests
    root: Path,
    name: str,
    *,
    outputs: Sequence[Path] = (),
    image_settings: ImageSettings | None = None,
    model: str | None = None,
    retry_errors: bool = True,
    holder: str | None = None,
    now: float | None = None,
    kind: str = KIND_CHUNK,
    schema_sha256: str | None = None,
    input_path: Path | None = None,
) -> Classification:
    """Classify item *name* of output root *root* from the store; never write.

    Only units of *kind* count. *outputs* are the item's rendered files for
    this run's formats. *image_settings* is this run's image-settings record,
    or a function that returns it for the item's stored record (None for
    text input), *model* its model and *schema_sha256* the SHA-256
    of its schema text. Pending units make the item partial, and so do
    failed ones unless *retry_errors* is False. A lease held by another run
    than *holder* that has not expired at *now* blocks the item, as an open
    or unresolved batch submission does. An *input_path* whose content
    differs from the document's recorded input refuses the item, partial or
    complete. A changed user image setting refuses a partial item, and so
    does a changed schema once a chunk is done; a changed model alone
    resumes with a note. Raises the store's errors for a versionless, newer
    or unreadable database.
    """
    path = database_path(root)
    present = tuple(output for output in outputs if _nonempty(output))
    state = _schema(root)
    store = None
    if state.exists and not state.empty:
        store = Store(path, TOOL, components=_COMPONENTS)
    resume = store.resume_state(name) if store is not None else None
    doc_id = resume.doc_id if resume is not None else None
    details = None
    if resume is not None:
        known = resume.doc_id
        details = read(path, lambda connection: _details(connection, known, kind))
    if resume is None or details is None or not details.counts:
        if present:
            return Classification(
                ItemState.UNSTORED,
                name,
                doc_id,
                reason=(
                    f"{present[0].name} exists without units in {DATABASE_NAME}; "
                    "--force extracts it again"
                ),
            )
        return Classification(ItemState.NONE, name, doc_id, reason="no stored units")
    assert doc_id is not None

    hold = _blocked(resume, name, kind, holder, now) or _input_refusal(
        name, doc_id, details, input_path
    )
    if hold is not None:
        return hold

    recorded, changed, model_changed = _image_changes(resume, image_settings)
    earlier = [used for used in details.models if model is not None and used != model]
    note = (
        _model_note(name, earlier, model, recorded)
        if earlier or model_changed
        else None
    )

    pending = _of_kind(resume.pending, kind)
    failed = _of_kind(resume.failed, kind)
    work = pending + (failed if retry_errors else ())
    if not work:
        missing = tuple(output for output in outputs if not output.is_file())
        stale = any(
            details.outputs.get(_path_key(output, root)) == 1 for output in outputs
        )
        render = bool(missing) or stale
        reason = "all units stored"
        if failed:
            reason += f", {len(failed)} failed"
        if render:
            reason += "; its files are rendered again"
        return Classification(
            ItemState.COMPLETE,
            name,
            doc_id,
            render=render,
            missing=missing,
            reason=reason,
            note=note,
            total=sum(details.counts.values()),
        )
    if changed:
        return Classification(
            ItemState.REFUSED,
            name,
            doc_id,
            reason=(
                f"Image settings changed for '{name}': {', '.join(changed)}. "
                "Skipping it; use --force to replace the existing run."
            ),
        )
    if _schema_changed(details, schema_sha256):
        return Classification(
            ItemState.REFUSED,
            name,
            doc_id,
            reason=(
                f"The schema changed for '{name}' since its done chunks were "
                "extracted. Skipping it; use --force to extract it again with "
                "the current schema, or restore the earlier schema."
            ),
        )
    units = tuple(sorted(work))
    return Classification(
        ItemState.PARTIAL,
        name,
        doc_id,
        units=units,
        reason=(f"{len(units)} of {sum(details.counts.values())} unit(s) to process"),
        note=note,
        total=sum(details.counts.values()),
    )
