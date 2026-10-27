"""The run store: one SQLite database per tool and output folder.

The store holds every run of a tool in one output folder: the runs with their
effective specs, the prompt and schema texts they used, the documents and
their expected units, every provider attempt verbatim, the batch submissions
and the registry of rendered files. Tools add content tables through their
own schema component and render their files from the store.

Schema versions live in ``meta``, one row per component: ``core`` for the
tables defined here and one per tool component registered by the caller.
After a drop, the row ``dropped_doc_id`` holds the highest document id ever
dropped; a new document takes an id above it and above every stored id, so
no id names two documents in turn.

Opening a database applies the missing migration steps of every component
additively in one transaction. A database with a newer schema, one with
tables but no ``meta`` (not a store) and one SQLite cannot read are refused
with a :class:`StoreError` (exit code 2), and none of them is recreated or
modified. Only a writer creates a missing file.

Writes run through one :class:`~.sqlite.SqliteWriter` per store, each in one
short transaction that never spans a network call; a busy database delays a
write without failing it. A document is worked under a lease: every write
transaction on its units, outputs and submissions checks, inside the
transaction, that the lease still names this holder, carries its generation
and has not expired by the clock read inside the transaction; it raises
:class:`LeaseLostError` otherwise, and a renewal of an expired lease fails the
same way. A write that waited on a busy database past the lease's expiry
therefore fails, while the response rows of its attempts stay: response rows
are appended without a lease check and are never updated; only
:meth:`Store.drop` deletes them, together with their document. Reads use
short read-only connections; a busy database delays a read the same way,
retried with the writer's backoff and warnings, without failing it.

The error text of a unit or response row is cut to :data:`ERROR_TEXT_LIMIT`
characters (:func:`bounded_error`); a response's ``raw`` column keeps the
body as received.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Final, TypeVar

from .capture import AttemptRecord, redact_payloads
from .selection import select_items
from .sqlite import (
    DEFAULT_BUSY_TIMEOUT,
    DEFAULT_QUEUE_SIZE,
    DEFAULT_WARN_INTERVAL,
    BusyCallback,
    SqliteWriter,
    is_busy,
    read,
)

__all__ = [
    "BLOCKING_SUBMISSIONS",
    "CORE",
    "CORE_TABLES",
    "CORE_VERSION",
    "DOCUMENT_STATUSES",
    "ERROR_TEXT_LIMIT",
    "HEARTBEAT_INTERVAL",
    "LEASE_TTL",
    "NOTE_LEVELS",
    "OPEN_SUBMISSIONS",
    "RAW_KINDS",
    "SUBMISSION_STATUSES",
    "UNIT_COUNTS_SQL",
    "UNIT_STATUSES",
    "Component",
    "DocumentRow",
    "DocumentSelection",
    "DocumentSelectionError",
    "DropRefusal",
    "DropReport",
    "KeptFile",
    "Lease",
    "LeaseKeeper",
    "LeaseLostError",
    "NewerSchemaError",
    "OutputRow",
    "PlannedUnit",
    "RenderReport",
    "RenderTarget",
    "Renderer",
    "ResponseRow",
    "ResumeState",
    "SchemaState",
    "Store",
    "StoreError",
    "StoreRecorder",
    "Submission",
    "SubmissionRejectedError",
    "SubmissionStateError",
    "UnitKey",
    "UnitResult",
    "UnitRow",
    "UnreadableDatabaseError",
    "VersionlessDatabaseError",
    "WriterOptions",
    "bounded_error",
    "check_schema",
    "current_unit",
    "file_sha256",
    "refresh_documents",
    "select_documents",
    "unit_context",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")

CORE: Final = "core"
LEASE_TTL: Final = 600.0
HEARTBEAT_INTERVAL: Final = 60.0
ERROR_TEXT_LIMIT: Final = 2000
"""Characters of error text a unit or response row keeps."""

UNIT_STATUSES: Final = ("pending", "done", "failed")
DOCUMENT_STATUSES: Final = ("pending", "partial", "complete", "failed")
SUBMISSION_STATUSES: Final = (
    "submitting",
    "submitted",
    "unresolved",
    "failed",
    "finalized",
    "abandoned",
    "cancelled",
)
OPEN_SUBMISSIONS: Final = ("submitting", "submitted")
BLOCKING_SUBMISSIONS: Final = ("submitting", "submitted", "unresolved")
RAW_KINDS: Final = ("http", "sdk", "line")
NOTE_LEVELS: Final = ("info", "warning", "error")
CORE_TABLES: Final = (
    "meta",
    "runs",
    "artifacts",
    "documents",
    "units",
    "responses",
    "submissions",
    "submission_requests",
    "outputs",
    "notes",
)

_JSON_COLUMNS = frozenset(
    {"usage", "provenance", "image_settings", "counts", "request", "spec"}
)


def _choices(values: Sequence[str]) -> str:
    return ", ".join(f"'{value}'" for value in values)


_CORE_V1 = f"""
CREATE TABLE meta (
    component TEXT PRIMARY KEY,
    version INTEGER NOT NULL CHECK (version >= 1)
);
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    tool TEXT NOT NULL,
    tool_version TEXT,
    command TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    exit_code INTEGER,
    spec TEXT,
    settings_fingerprint TEXT
);
CREATE TABLE artifacts (
    sha256 TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    run_id TEXT REFERENCES runs (run_id),
    created_at TEXT NOT NULL
);
CREATE TABLE documents (
    doc_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    tool TEXT NOT NULL,
    input_path TEXT,
    input_sha256 TEXT,
    kind TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ({_choices(DOCUMENT_STATUSES)})),
    units_total INTEGER NOT NULL DEFAULT 0,
    units_pending INTEGER NOT NULL DEFAULT 0,
    units_done INTEGER NOT NULL DEFAULT 0,
    units_failed INTEGER NOT NULL DEFAULT 0,
    image_settings TEXT,
    first_run_id TEXT REFERENCES runs (run_id),
    last_run_id TEXT REFERENCES runs (run_id),
    lease_run_id TEXT,
    lease_until REAL,
    lease_generation INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT
);
CREATE TABLE units (
    doc_id INTEGER NOT NULL REFERENCES documents (doc_id),
    kind TEXT NOT NULL,
    idx INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ({_choices(UNIT_STATUSES)})),
    outcome TEXT,
    source_index,
    label TEXT,
    content TEXT,
    tool TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    usage TEXT,
    error TEXT,
    error_type TEXT,
    provenance TEXT,
    response_id INTEGER REFERENCES responses (response_id),
    stale INTEGER NOT NULL DEFAULT 0 CHECK (stale IN (0, 1)),
    updated_at TEXT,
    PRIMARY KEY (doc_id, kind, idx)
);
CREATE INDEX units_by_status ON units (doc_id, status);
CREATE TABLE responses (
    response_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER REFERENCES documents (doc_id),
    kind TEXT,
    idx INTEGER,
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    tool TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    role TEXT,
    attempt INTEGER,
    submission_id TEXT REFERENCES submissions (submission_id),
    custom_id TEXT,
    request TEXT,
    raw BLOB,
    raw_kind TEXT CHECK (raw_kind IS NULL OR raw_kind IN ({_choices(RAW_KINDS)})),
    usage TEXT,
    error TEXT,
    error_type TEXT,
    http_status INTEGER,
    latency REAL,
    created_at TEXT NOT NULL,
    CHECK (raw IS NULL OR raw_kind IS NOT NULL)
);
CREATE INDEX responses_by_unit ON responses (doc_id, kind, idx);
CREATE TRIGGER responses_no_update BEFORE UPDATE ON responses
BEGIN
    SELECT RAISE (ABORT, 'responses are append-only');
END;
CREATE TRIGGER responses_no_delete BEFORE DELETE ON responses
BEGIN
    SELECT RAISE (ABORT, 'responses are append-only');
END;
CREATE TABLE submissions (
    submission_id TEXT PRIMARY KEY,
    tool TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    part INTEGER NOT NULL DEFAULT 0,
    intent_at TEXT NOT NULL,
    upload_ref TEXT,
    batch_id TEXT,
    status TEXT NOT NULL CHECK (status IN ({_choices(SUBMISSION_STATUSES)})),
    provider_status TEXT,
    request_count INTEGER NOT NULL DEFAULT 0,
    counts TEXT,
    last_checked_at TEXT,
    finalized_at TEXT,
    error TEXT,
    updated_at TEXT,
    CHECK (status NOT IN ('submitted', 'finalized') OR batch_id IS NOT NULL)
);
CREATE INDEX submissions_by_status ON submissions (status);
CREATE UNIQUE INDEX submissions_by_batch ON submissions (provider, batch_id)
    WHERE batch_id IS NOT NULL;
CREATE TABLE submission_requests (
    submission_id TEXT NOT NULL REFERENCES submissions (submission_id),
    custom_id TEXT NOT NULL,
    doc_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    idx INTEGER NOT NULL,
    PRIMARY KEY (submission_id, custom_id),
    FOREIGN KEY (doc_id, kind, idx) REFERENCES units (doc_id, kind, idx)
);
CREATE INDEX submission_requests_by_unit ON submission_requests (doc_id, kind, idx);
CREATE TABLE outputs (
    doc_id INTEGER NOT NULL REFERENCES documents (doc_id),
    path TEXT NOT NULL,
    format TEXT NOT NULL,
    sha256 TEXT,
    rendered_at TEXT,
    run_id TEXT REFERENCES runs (run_id),
    stale INTEGER NOT NULL DEFAULT 0 CHECK (stale IN (0, 1)),
    PRIMARY KEY (doc_id, path)
);
CREATE TABLE notes (
    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER REFERENCES documents (doc_id),
    run_id TEXT REFERENCES runs (run_id),
    level TEXT NOT NULL CHECK (level IN ({_choices(NOTE_LEVELS)})),
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

UNIT_COUNTS_SQL: Final = (
    "SELECT status, count(*) FROM units WHERE doc_id = ? GROUP BY status"
)
"""The resume query of unit counts per status; it reads ``units_by_status``."""

_UNITS_WITH_STATUS = (
    "SELECT kind, idx FROM units WHERE doc_id = ? AND status = ? ORDER BY kind, idx"
)
_BLOCKED_UNITS = f"""
SELECT DISTINCT r.kind, r.idx FROM submission_requests AS r
JOIN submissions AS s ON s.submission_id = r.submission_id
WHERE r.doc_id = ? AND s.status IN ({_choices(BLOCKING_SUBMISSIONS)})
ORDER BY r.kind, r.idx
"""
_STALE_CANDIDATES = """
SELECT s.submission_id FROM submissions AS s
WHERE s.status = 'submitting' AND s.batch_id IS NULL AND s.run_id IS NOT :run
AND NOT EXISTS (
    SELECT 1 FROM submission_requests AS r
    JOIN documents AS d ON d.doc_id = r.doc_id
    WHERE r.submission_id = s.submission_id
    AND d.lease_run_id IS NOT NULL AND d.lease_run_id IS NOT :run
    AND d.lease_until > :now
)
ORDER BY s.intent_at, s.submission_id
"""
_ACQUIRE = """
UPDATE documents
SET lease_run_id = :holder, lease_until = :until,
    lease_generation = lease_generation + 1
WHERE doc_id = :doc_id
AND (lease_run_id IS NULL OR lease_until IS NULL OR lease_until <= :now
     OR lease_run_id = :holder)
"""
_REFRESH_COUNTS = """
UPDATE documents SET
    units_total = (SELECT count(*) FROM units AS u
                   WHERE u.doc_id = documents.doc_id),
    units_pending = (SELECT count(*) FROM units AS u
                     WHERE u.doc_id = documents.doc_id AND u.status = 'pending'),
    units_done = (SELECT count(*) FROM units AS u
                  WHERE u.doc_id = documents.doc_id AND u.status = 'done'),
    units_failed = (SELECT count(*) FROM units AS u
                    WHERE u.doc_id = documents.doc_id AND u.status = 'failed')
"""
_REFRESH_STATUS = """
UPDATE documents SET status = CASE
    WHEN units_total = 0 OR units_pending = units_total THEN 'pending'
    WHEN units_done = units_total THEN 'complete'
    WHEN units_pending = 0 THEN 'failed'
    ELSE 'partial'
END
"""
_UPSERT_UNIT = """
INSERT INTO units (
    doc_id, kind, idx, status, outcome, source_index, label, content, tool,
    model, provider, run_id, attempts, usage, error, error_type, provenance,
    response_id, stale, updated_at
) VALUES (
    :doc_id, :kind, :idx, :status, :outcome, :source_index, :label, :content,
    :tool, :model, :provider, :run_id, COALESCE(:attempts, 1), :usage, :error,
    :error_type, :provenance, :response_id, 0, :now
)
ON CONFLICT (doc_id, kind, idx) DO UPDATE SET
    status = excluded.status,
    outcome = excluded.outcome,
    source_index = COALESCE(excluded.source_index, units.source_index),
    label = COALESCE(excluded.label, units.label),
    content = excluded.content,
    tool = excluded.tool,
    model = excluded.model,
    provider = excluded.provider,
    run_id = excluded.run_id,
    attempts = COALESCE(:attempts, units.attempts + 1),
    usage = excluded.usage,
    error = excluded.error,
    error_type = excluded.error_type,
    provenance = excluded.provenance,
    response_id = excluded.response_id,
    stale = 0,
    updated_at = excluded.updated_at
"""
_PLAN_UNIT = """
INSERT INTO units (
    doc_id, kind, idx, status, source_index, label, tool, model, provider,
    run_id, attempts, stale, updated_at
) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, 0, 0, ?)
ON CONFLICT (doc_id, kind, idx) DO NOTHING
"""
_INSERT_RESPONSE = """
INSERT INTO responses (
    doc_id, kind, idx, run_id, tool, provider, model, role, attempt,
    submission_id, custom_id, request, raw, raw_kind, usage, error, error_type,
    http_status, latency, created_at
) VALUES (
    :doc_id, :kind, :idx, :run_id, :tool, :provider, :model, :role, :attempt,
    :submission_id, :custom_id, :request, :raw, :raw_kind, :usage, :error,
    :error_type, :http_status, :latency, :now
)
"""
_OUTPUT_DONE = """
INSERT INTO outputs (doc_id, path, format, sha256, rendered_at, run_id, stale)
VALUES (?, ?, ?, ?, ?, ?, 0)
ON CONFLICT (doc_id, path) DO UPDATE SET
    format = excluded.format, sha256 = excluded.sha256,
    rendered_at = excluded.rendered_at, run_id = excluded.run_id, stale = 0
"""
_OUTPUT_STALE = """
INSERT INTO outputs (doc_id, path, format, sha256, rendered_at, run_id, stale)
VALUES (?, ?, ?, ?, NULL, ?, 1)
ON CONFLICT (doc_id, path) DO UPDATE SET format = excluded.format, stale = 1
"""

_DROP_DOCUMENTS = """
SELECT doc_id, name, lease_run_id, lease_until FROM documents
WHERE doc_id IN (SELECT value FROM json_each(?))
"""
_DROP_SUBMISSIONS = f"""
SELECT DISTINCT s.submission_id, s.status, s.intent_at FROM submission_requests AS r
JOIN submissions AS s ON s.submission_id = r.submission_id
WHERE r.doc_id = ? AND s.status IN ({_choices(BLOCKING_SUBMISSIONS)})
ORDER BY s.intent_at, s.submission_id
"""
_DROP_OUTPUTS = """
SELECT o.path, o.sha256, EXISTS (
    SELECT 1 FROM outputs AS other WHERE other.path = o.path
    AND other.doc_id NOT IN (SELECT value FROM json_each(:ids))
) FROM outputs AS o
WHERE o.doc_id IN (SELECT value FROM json_each(:ids))
ORDER BY o.path
"""
_IN_IDS = "WHERE doc_id IN (SELECT value FROM json_each(?))"
_NO_DELETE_TRIGGER = "responses_no_delete"
_DROPPED_MARK = "dropped_doc_id"
_NEXT_DOC_ID = f"""(SELECT max(
    COALESCE((SELECT max(doc_id) FROM documents), 0),
    COALESCE((SELECT version FROM meta WHERE component = '{_DROPPED_MARK}'), 0)
) + 1)"""
"""The id of a new document: above every stored and every dropped id."""

_TRANSITIONS: Final[Mapping[str, tuple[str, ...]]] = {
    "submitted": ("submitting", "unresolved"),
    "unresolved": ("submitting",),
    "failed": ("submitting", "submitted", "unresolved"),
    "finalized": ("submitted",),
    "abandoned": ("unresolved", "submitted"),
    "cancelled": ("submitted",),
}
_NOW: Final = object()
"""Placeholder for the transaction's timestamp in submission updates."""


# =============================================================================
# Errors
# =============================================================================


class StoreError(Exception):
    """A database the store refuses to open; tools exit with code 2."""

    exit_code: int = 2


class NewerSchemaError(StoreError):
    """The database carries a schema version newer than this tool knows."""


class VersionlessDatabaseError(StoreError):
    """The database has tables but no ``meta`` table: it is not a store."""


class UnreadableDatabaseError(StoreError):
    """SQLite cannot read the file as a database."""


class LeaseLostError(RuntimeError):
    """This holder no longer holds the document's lease; the write was refused."""

    def __init__(self, message: str, lease: Lease | None = None) -> None:
        super().__init__(message)
        self.lease = lease


class SubmissionStateError(RuntimeError):
    """A submission cannot move from its current status to the requested one."""


class SubmissionRejectedError(Exception):
    """Raised by a create callback when the provider certainly made no batch."""


class DocumentSelectionError(ValueError):
    """A document selection that is malformed or matches no stored document."""


# =============================================================================
# Schema components and versions
# =============================================================================

Migration = str | Callable[[sqlite3.Connection], None]
"""One migration step: SQL statements, or a function run on the connection."""


@dataclass(frozen=True)
class Component:
    """A schema component: its name and its ordered migration steps.

    The component's version is the number of steps. Steps run in order inside
    the transaction that opens the database; a released step never changes,
    so a new version only appends steps.
    """

    name: str
    steps: tuple[Migration, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A schema component needs a name.")

    @property
    def version(self) -> int:
        """Return the schema version this component's steps reach."""
        return len(self.steps)


_CORE_V2 = "ALTER TABLE submission_requests ADD COLUMN request TEXT;"
"""Version 2 keeps each batch request's body as sent, payloads redacted."""

CORE_COMPONENT: Final = Component(CORE, (_CORE_V1, _CORE_V2))
CORE_VERSION: Final = CORE_COMPONENT.version


@dataclass(frozen=True)
class SchemaState:
    """What a database file holds: its tables with their columns and versions."""

    path: Path
    exists: bool
    tables: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    versions: Mapping[str, int] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        """Return True when the database holds no table."""
        return not self.tables

    @property
    def versionless(self) -> bool:
        """Return True for tables without a ``meta`` table."""
        return bool(self.tables) and "meta" not in self.tables


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _statements(script: str) -> Iterator[str]:
    pieces = script.split(";")
    buffer = ""
    for number, piece in enumerate(pieces):
        buffer += piece if number == len(pieces) - 1 else piece + ";"
        if sqlite3.complete_statement(buffer):
            yield buffer.strip()
            buffer = ""
    rest = buffer.strip()
    if rest and not sqlite3.complete_statement(rest + ";"):
        raise ValueError(f"Incomplete SQL statement in migration: {rest!r}")
    if rest:
        yield rest


def _apply_step(connection: sqlite3.Connection, step: Migration) -> None:
    if isinstance(step, str):
        for statement in _statements(step):
            connection.execute(statement)
    else:
        step(connection)


def _schema(connection: sqlite3.Connection, path: Path) -> SchemaState:
    names = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    tables = {
        name: tuple(
            row[1] for row in connection.execute(f"PRAGMA table_info({_ident(name)})")
        )
        for name in names
    }
    versions: dict[str, int] = {}
    if "meta" in tables:
        versions = {
            str(component): int(version)
            for component, version in connection.execute(
                "SELECT component, version FROM meta"
            )
        }
    return SchemaState(path, True, tables, versions)


def _inspect(path: Path, busy_timeout: float) -> SchemaState:
    """Read the tables and versions of the database at *path*; never write.

    A missing file yields a state with ``exists`` False; a file SQLite cannot
    read raises :class:`UnreadableDatabaseError`.
    """
    if not path.exists():
        return SchemaState(path, False)
    try:
        return read(
            path,
            lambda connection: _schema(connection, path),
            busy_timeout=busy_timeout,
        )
    except sqlite3.DatabaseError as exc:
        if is_busy(exc):
            raise
        raise UnreadableDatabaseError(
            f"{path} is not a readable SQLite database ({exc}); it is left as it is."
        ) from exc
    except OSError as exc:
        raise UnreadableDatabaseError(
            f"{path} cannot be opened as a database ({exc}); it is left as it is."
        ) from exc


def _check(
    state: SchemaState, components: Sequence[Component], tool: str | None
) -> None:
    if state.versionless:
        store = f"{tool} store" if tool else "store"
        raise VersionlessDatabaseError(
            f"{state.path} holds tables but no schema version: it is not a "
            f"{store} and is left as it is."
        )
    for component in (CORE_COMPONENT, *components):
        have = state.versions.get(component.name, 0)
        if have > component.version:
            raise NewerSchemaError(
                f"{state.path} carries schema {component.name!r} version {have}; "
                f"this tool knows version {component.version}. Use a newer tool."
            )


def check_schema(
    path: Path | str,
    components: Sequence[Component] = (),
    *,
    tool: str | None = None,
    busy_timeout: float = DEFAULT_BUSY_TIMEOUT,
) -> SchemaState:
    """Return the schema state of *path*; raise a :class:`StoreError` to refuse it.

    *tool* names the tool in the refusal of a database without a schema
    version.
    """
    state = _inspect(Path(path), busy_timeout)
    if state.exists:
        _check(state, components, tool)
    return state


def _migrate(
    connection: sqlite3.Connection,
    components: Sequence[Component],
    path: Path,
    tool: str,
) -> dict[str, int]:
    state = _schema(connection, path)
    _check(state, components, tool)
    return _upgrade(connection, components, state.versions)


def _upgrade(
    connection: sqlite3.Connection,
    components: Sequence[Component],
    current: Mapping[str, int],
) -> dict[str, int]:
    versions = dict(current)
    for component in (CORE_COMPONENT, *components):
        have = versions.get(component.name, 0)
        if have == component.version:
            continue
        for step in component.steps[have:]:
            _apply_step(connection, step)
        connection.execute(
            "INSERT INTO meta (component, version) VALUES (?, ?) "
            "ON CONFLICT (component) DO UPDATE SET version = excluded.version",
            (component.name, component.version),
        )
        versions[component.name] = component.version
    return versions


# =============================================================================
# Rows and values
# =============================================================================


@dataclass(frozen=True, order=True)
class UnitKey:
    """Identifies one unit: document id, unit kind and index."""

    doc_id: int
    kind: str
    idx: int


@dataclass(frozen=True)
class PlannedUnit:
    """An expected unit of a document, created as ``pending`` at planning."""

    idx: int
    label: str | None = None
    source_index: int | str | None = None


@dataclass(frozen=True)
class UnitResult:
    """The outcome of one unit as a run writes it.

    *status* is ``done`` or ``failed``; *outcome* is the tool's own
    classification of the content. *attempts* None counts one more attempt.
    """

    kind: str
    idx: int
    status: str
    model: str
    provider: str
    content: str | None = None
    outcome: str | None = None
    label: str | None = None
    source_index: int | str | None = None
    usage: Mapping[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None
    provenance: Mapping[str, Any] | None = None
    response_id: int | None = None
    attempts: int | None = None


@dataclass(frozen=True)
class ResponseRow:
    """One provider attempt to append: the request and the response verbatim.

    *raw* is the response body as it came over HTTP (``raw_kind`` ``http``),
    a dump of the SDK object (``sdk``) or a batch result line as downloaded
    (``line``). Image payloads in *request* are replaced by placeholders
    before the row is written.
    """

    provider: str
    model: str
    unit: UnitKey | None = None
    role: str | None = None
    attempt: int | None = None
    submission_id: str | None = None
    custom_id: str | None = None
    request: Mapping[str, Any] | None = None
    raw: str | bytes | None = None
    raw_kind: str | None = None
    usage: Mapping[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None
    http_status: int | None = None
    latency: float | None = None


@dataclass(frozen=True)
class Lease:
    """A document lease: holder (a run id), generation and expiry (epoch s)."""

    doc_id: int
    holder: str
    generation: int
    until: float


@dataclass(frozen=True)
class DocumentRow:
    """A ``documents`` row."""

    doc_id: int
    name: str
    tool: str
    input_path: str | None
    input_sha256: str | None
    kind: str | None
    status: str
    units_total: int
    units_pending: int
    units_done: int
    units_failed: int
    image_settings: Any
    first_run_id: str | None
    last_run_id: str | None
    lease_run_id: str | None
    lease_until: float | None
    lease_generation: int
    updated_at: str | None


@dataclass(frozen=True)
class UnitRow:
    """A ``units`` row; ``usage`` and ``provenance`` are decoded from JSON."""

    doc_id: int
    kind: str
    idx: int
    status: str
    outcome: str | None
    source_index: int | str | None
    label: str | None
    content: str | None
    tool: str
    model: str
    provider: str
    run_id: str
    attempts: int
    usage: Any
    error: str | None
    error_type: str | None
    provenance: Any
    response_id: int | None
    stale: int
    updated_at: str | None

    @property
    def key(self) -> UnitKey:
        """Return the unit's key."""
        return UnitKey(self.doc_id, self.kind, self.idx)


@dataclass(frozen=True)
class Submission:
    """A ``submissions`` row: one submitted part of a batch."""

    submission_id: str
    tool: str
    provider: str
    model: str
    run_id: str
    part: int
    intent_at: str
    upload_ref: str | None
    batch_id: str | None
    status: str
    provider_status: str | None
    request_count: int
    counts: Any
    last_checked_at: str | None
    finalized_at: str | None
    error: str | None
    updated_at: str | None


@dataclass(frozen=True)
class OutputRow:
    """An ``outputs`` row; *path* is relative to the database folder if inside."""

    doc_id: int
    path: str
    format: str
    sha256: str | None
    rendered_at: str | None
    run_id: str | None
    stale: int


@dataclass(frozen=True)
class ResumeState:
    """What resume needs of one document, read in one short connection."""

    doc_id: int
    name: str
    status: str
    counts: Mapping[str, int]
    pending: tuple[UnitKey, ...]
    failed: tuple[UnitKey, ...]
    blocked: tuple[UnitKey, ...]
    image_settings: Any
    lease_run_id: str | None
    lease_until: float | None

    @property
    def ready(self) -> tuple[UnitKey, ...]:
        """Return the pending units that no open or unresolved batch holds."""
        blocked = set(self.blocked)
        return tuple(unit for unit in self.pending if unit not in blocked)


Renderer = Callable[[sqlite3.Connection, int], str | bytes]
"""Returns a document's rendering from a read-only connection and its id.

Text is written as UTF-8; bytes (a DOCX archive) are written as they are.
"""


@dataclass(frozen=True)
class RenderTarget:
    """One rendered file: its path, its format name and its renderer."""

    path: Path
    format: str
    renderer: Renderer


@dataclass
class RenderReport:
    """The files a render wrote, found unchanged, left alone or marked stale."""

    written: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    kept: list[Path] = field(default_factory=list)
    stale: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentSelection:
    """The stored documents a selection chose, by name, and its warnings."""

    documents: tuple[DocumentRow, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DropRefusal:
    """A document :meth:`Store.drop` left in place, and why.

    *name* is None when no document has the id.
    """

    doc_id: int
    name: str | None
    reason: str


@dataclass(frozen=True)
class KeptFile:
    """A rendered file :meth:`Store.drop` did not delete, and why."""

    path: Path
    reason: str


@dataclass(frozen=True)
class DropReport:
    """The outcome of :meth:`Store.drop`.

    *dropped* names the documents deleted and *deleted_files* the rendered
    files removed; in a dry run they name what a real drop would delete.
    """

    dry_run: bool
    dropped: tuple[str, ...] = ()
    refused: tuple[DropRefusal, ...] = ()
    deleted_files: tuple[Path, ...] = ()
    kept_files: tuple[KeptFile, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return the report as JSON-serializable values."""
        return {
            "dry_run": self.dry_run,
            "dropped": list(self.dropped),
            "refused": [
                {"doc_id": item.doc_id, "name": item.name, "reason": item.reason}
                for item in self.refused
            ],
            "deleted_files": [str(path) for path in self.deleted_files],
            "kept_files": [
                {"path": str(item.path), "reason": item.reason}
                for item in self.kept_files
            ],
        }


@dataclass(frozen=True)
class WriterOptions:
    """Tuning of the store's writer (see :class:`~.sqlite.SqliteWriter`)."""

    busy_timeout: float = DEFAULT_BUSY_TIMEOUT
    warn_interval: float = DEFAULT_WARN_INTERVAL
    queue_size: int = DEFAULT_QUEUE_SIZE
    on_busy: BusyCallback | None = None


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat(timespec="milliseconds")


def _json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


def _decode(column: str, value: Any) -> Any:
    if column in _JSON_COLUMNS and isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _dicts(
    connection: sqlite3.Connection, sql: str, params: Sequence[Any] = ()
) -> list[dict[str, Any]]:
    cursor = connection.execute(sql, tuple(params))
    names = [description[0] for description in cursor.description]
    return [
        {name: _decode(name, value) for name, value in zip(names, row, strict=True)}
        for row in cursor.fetchall()
    ]


def _pick(row: Mapping[str, Any], cls: type[Any]) -> dict[str, Any]:
    return {item.name: row[item.name] for item in fields(cls)}


def _raw_value(raw: str | bytes | None) -> str | bytes | None:
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw
    return raw


_HASH_BLOCK: Final = 1024 * 1024


def file_sha256(path: Path) -> str | None:
    """Return the SHA-256 of the file at *path*, or None when it is missing.

    The file is read in 1 MiB blocks; any other ``OSError`` propagates.
    """
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while block := handle.read(_HASH_BLOCK):
                digest.update(block)
    except FileNotFoundError:
        return None
    return digest.hexdigest()


def refresh_documents(
    connection: sqlite3.Connection, doc_id: int | None = None
) -> None:
    """Recompute the unit counts and status of one document or of all."""
    if doc_id is None:
        connection.execute(_REFRESH_COUNTS)
        connection.execute(_REFRESH_STATUS)
    else:
        connection.execute(_REFRESH_COUNTS + " WHERE doc_id = ?", (doc_id,))
        connection.execute(_REFRESH_STATUS + " WHERE doc_id = ?", (doc_id,))


def _touch(
    connection: sqlite3.Connection, doc_id: int, run_id: str | None, now: str
) -> None:
    refresh_documents(connection, doc_id)
    connection.execute(
        "UPDATE documents SET last_run_id = COALESCE(?, last_run_id), "
        "updated_at = ? WHERE doc_id = ?",
        (run_id, now, doc_id),
    )


def bounded_error(text: str | None, limit: int = ERROR_TEXT_LIMIT) -> str | None:
    """Return *text* cut to its first *limit* characters, marked when cut.

    A provider can answer an error with a whole HTML page; the error columns
    keep its head, and a response's ``raw`` column the body as received.
    """
    if text is None or len(text) <= limit:
        return text
    return f"{text[:limit]} [... {len(text) - limit:,} more characters]"


def _unit_params(
    doc_id: int, result: UnitResult, tool: str, run_id: str, now: str
) -> dict[str, Any]:
    """Return the parameters of :data:`_UPSERT_UNIT` for *result*."""
    return {
        "doc_id": doc_id,
        "kind": result.kind,
        "idx": result.idx,
        "status": result.status,
        "outcome": result.outcome,
        "source_index": result.source_index,
        "label": result.label,
        "content": result.content,
        "tool": tool,
        "model": result.model,
        "provider": result.provider,
        "run_id": run_id,
        "attempts": result.attempts,
        "usage": _json(result.usage),
        "error": bounded_error(result.error),
        "error_type": result.error_type,
        "provenance": _json(result.provenance),
        "response_id": result.response_id,
        "now": now,
    }


def _response_params(
    response: ResponseRow, tool: str, run_id: str, now: str
) -> dict[str, Any]:
    """Return the parameters of :data:`_INSERT_RESPONSE` for *response*."""
    unit = response.unit
    return {
        "doc_id": None if unit is None else unit.doc_id,
        "kind": None if unit is None else unit.kind,
        "idx": None if unit is None else unit.idx,
        "run_id": run_id,
        "tool": tool,
        "provider": response.provider,
        "model": response.model,
        "role": response.role,
        "attempt": response.attempt,
        "submission_id": response.submission_id,
        "custom_id": response.custom_id,
        "request": _json(
            None if response.request is None else redact_payloads(response.request)
        ),
        "raw": _raw_value(response.raw),
        "raw_kind": response.raw_kind,
        "usage": _json(response.usage),
        "error": bounded_error(response.error),
        "error_type": response.error_type,
        "http_status": response.http_status,
        "latency": response.latency,
        "now": now,
    }


def _check_lease(connection: sqlite3.Connection, lease: Lease, now: float) -> None:
    """Raise unless *lease* holds and has not expired at *now*.

    Holder, generation and expiry are compared with the row read inside the
    transaction; *now* is the clock reading taken inside it.
    """
    row = connection.execute(
        "SELECT lease_run_id, lease_generation, lease_until FROM documents "
        "WHERE doc_id = ?",
        (lease.doc_id,),
    ).fetchone()
    if row is None:
        raise LeaseLostError(f"Document {lease.doc_id} does not exist.", lease)
    holder, generation, until = row
    if holder != lease.holder or generation != lease.generation:
        raise LeaseLostError(
            f"The lease on document {lease.doc_id} passed to another holder.", lease
        )
    if until is None or until <= now:
        raise LeaseLostError(f"The lease on document {lease.doc_id} expired.", lease)


# =============================================================================
# Current unit context
# =============================================================================

_UNIT: ContextVar[UnitKey | None] = ContextVar("store_unit", default=None)


def current_unit() -> UnitKey | None:
    """Return the unit the current provider call works on, or None."""
    return _UNIT.get()


@contextlib.contextmanager
def unit_context(unit: UnitKey | None) -> Iterator[None]:
    """Attribute the provider attempts made inside the block to *unit*."""
    token = _UNIT.set(unit)
    try:
        yield
    finally:
        _UNIT.reset(token)


# =============================================================================
# Render helpers
# =============================================================================


@dataclass
class _PlannedFile:
    target: RenderTarget
    key: str
    sha256: str
    current: str | None
    recorded: str | None
    temp: Path | None = None
    error: str | None = None
    kept: bool = False


def _changed_since_plan(item: _PlannedFile) -> bool:
    """Whether the target's contents changed after the render planned it."""
    try:
        return file_sha256(item.target.path) != item.current
    except OSError:
        return True


def _replace(item: _PlannedFile) -> str | None:
    temp = item.temp
    if temp is None:
        return None
    try:
        os.replace(temp, item.target.path)
    except OSError as exc:
        try:
            landed = file_sha256(item.target.path) == item.sha256
        except OSError:
            landed = False
        if landed and not temp.exists():
            return None
        return (
            f"{item.target.path} could not be replaced ({exc}); the data is "
            "stored and the output is marked stale."
        )
    return None


def _commit_render(
    connection: sqlite3.Connection,
    doc_id: int,
    items: Sequence[_PlannedFile],
    run_id: str,
    now: str,
    force: bool = False,
) -> list[str | None]:
    """Replace the planned files and record them; return each file's error.

    Unless *force*, a target whose contents changed after planning is left
    as it is and its item marked kept.
    """
    errors: list[str | None] = []
    for item in items:
        if (
            item.error is None
            and item.temp is not None
            and not force
            and _changed_since_plan(item)
        ):
            item.kept = True
            errors.append(None)
            continue
        error = item.error if item.error is not None else _replace(item)
        if error is None:
            connection.execute(
                _OUTPUT_DONE,
                (doc_id, item.key, item.target.format, item.sha256, now, run_id),
            )
        else:
            recorded = item.recorded if item.recorded is not None else item.current
            connection.execute(
                _OUTPUT_STALE,
                (doc_id, item.key, item.target.format, recorded, run_id),
            )
        errors.append(error)
    return errors


# =============================================================================
# The store
# =============================================================================


class Store:
    """One tool's database in one output folder.

    Open it with ``async with Store(path, tool) as store``, then call
    :meth:`begin_run` before any write that records a run. *components* are
    the tool's schema components; *clock* returns epoch seconds and is
    injectable for tests.
    """

    def __init__(
        self,
        path: Path | str,
        tool: str,
        *,
        tool_version: str = "unknown",
        components: Sequence[Component] = (),
        clock: Callable[[], float] = time.time,
        lease_ttl: float = LEASE_TTL,
        writer: WriterOptions | None = None,
    ) -> None:
        names = [component.name for component in components]
        if not tool:
            raise ValueError("The store needs a tool name.")
        if CORE in names or len(set(names)) != len(names):
            raise ValueError(f"Component names must be unique and not {CORE!r}.")
        self.path = Path(path)
        self.tool = tool
        self.tool_version = tool_version
        self.components = tuple(components)
        self.clock = clock
        self.lease_ttl = lease_ttl
        self._options = writer or WriterOptions()
        self._writer: SqliteWriter | None = None
        self._run_id: str | None = None
        self._held: dict[int, Lease] = {}

    # ----------------------------------------------------------------- open

    @property
    def run_id(self) -> str | None:
        """Return the id of the run begun by this store, or None."""
        return self._run_id

    def check(self) -> SchemaState:
        """Return the database's schema state; raise to refuse it."""
        return check_schema(
            self.path,
            self.components,
            tool=self.tool,
            busy_timeout=self._options.busy_timeout,
        )

    async def open(self, *, create: bool = True) -> None:
        """Open the writer and apply missing migrations in one transaction."""
        if self._writer is not None:
            return
        state = self.check()
        if not state.exists and not create:
            raise FileNotFoundError(f"No database at {self.path}")
        options = self._options
        writer = SqliteWriter(
            self.path,
            (),
            busy_timeout=options.busy_timeout,
            warn_interval=options.warn_interval,
            queue_size=options.queue_size,
            on_busy=options.on_busy,
        )
        await writer.open()
        try:
            await writer.transaction(
                lambda connection: _migrate(
                    connection, self.components, self.path, self.tool
                )
            )
        except BaseException:
            await writer.close()
            raise
        self._writer = writer

    async def close(self) -> None:
        """Release every lease of this run and close the writer.

        Besides the held leases, a lease whose acquisition committed after
        its caller was cancelled (a Ctrl+Break during :meth:`acquire`) is
        released: the release names the run, not the lease, and runs after
        every write already queued.
        """
        writer = self._writer
        if writer is None:
            return
        try:
            for lease in list(self._held.values()):
                with contextlib.suppress(sqlite3.Error):
                    await self.release(lease)
            if self._run_id is not None:
                with contextlib.suppress(sqlite3.Error):
                    await self._release_run_leases(self._run_id)
        finally:
            self._writer = None
            await writer.close()

    async def __aenter__(self) -> Store:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def wait_capacity(self) -> None:
        """Return once the writer's queue has room; pause provider calls on it."""
        await self._require_writer().wait_capacity()

    def _require_writer(self) -> SqliteWriter:
        if self._writer is None:
            raise RuntimeError(f"The store at {self.path} is not open.")
        return self._writer

    def _require_run(self) -> str:
        if self._run_id is None:
            raise RuntimeError("Begin a run before writing run-stamped rows.")
        return self._run_id

    async def _write(self, func: Callable[[sqlite3.Connection], T]) -> T:
        try:
            return await self._require_writer().transaction(func)
        except LeaseLostError as exc:
            if exc.lease is not None:
                self._forget(exc.lease)
            raise

    async def _leased(self, lease: Lease, func: Callable[[sqlite3.Connection], T]) -> T:
        clock = self.clock

        def write(connection: sqlite3.Connection) -> T:
            _check_lease(connection, lease, clock())
            return func(connection)

        return await self._write(write)

    def _read(self, func: Callable[[sqlite3.Connection], T]) -> T:
        return read(self.path, func, busy_timeout=self._options.busy_timeout)

    def _forget(self, lease: Lease) -> None:
        held = self._held.get(lease.doc_id)
        if (
            held is not None
            and held.holder == lease.holder
            and held.generation == lease.generation
        ):
            del self._held[lease.doc_id]

    # ----------------------------------------------------------------- runs

    async def begin_run(
        self,
        command: str,
        *,
        spec: Mapping[str, Any] | None = None,
        settings_fingerprint: str | None = None,
    ) -> str:
        """Record a new run and return its id.

        *spec* maps option paths to their value and the source of the value.
        """
        run_id = str(uuid.uuid4())
        row = (
            run_id,
            self.tool,
            self.tool_version,
            command,
            _iso(self.clock()),
            _json(spec),
            settings_fingerprint,
        )
        await self._write(
            lambda connection: connection.execute(
                "INSERT INTO runs (run_id, tool, tool_version, command, started_at, "
                "spec, settings_fingerprint) VALUES (?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        )
        self._run_id = run_id
        return run_id

    async def end_run(self, exit_code: int) -> None:
        """Record the end time and exit code of the current run."""
        row = (_iso(self.clock()), exit_code, self._require_run())
        await self._write(
            lambda connection: connection.execute(
                "UPDATE runs SET ended_at = ?, exit_code = ? WHERE run_id = ?", row
            )
        )

    async def store_artifact(self, kind: str, text: str) -> str:
        """Store a prompt, schema or context text once; return its SHA-256."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = (digest, kind, text, self._run_id, _iso(self.clock()))
        await self._write(
            lambda connection: connection.execute(
                "INSERT INTO artifacts (sha256, kind, text, run_id, created_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (sha256) DO NOTHING",
                row,
            )
        )
        return digest

    async def add_note(
        self, message: str, *, doc_id: int | None = None, level: str = "warning"
    ) -> int:
        """Record a per-document warning of this run; return the note id."""
        row = (doc_id, self._run_id, level, message, _iso(self.clock()))

        def write(connection: sqlite3.Connection) -> int:
            cursor = connection.execute(
                "INSERT INTO notes (doc_id, run_id, level, message, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                row,
            )
            return int(cursor.lastrowid or 0)

        return await self._write(write)

    # ------------------------------------------------------------ documents

    async def ensure_document(
        self,
        name: str,
        *,
        input_path: Path | str | None = None,
        input_sha256: str | None = None,
        kind: str | None = None,
    ) -> int:
        """Return the id of document *name*, creating its row when missing.

        A new row takes an id above every stored and every dropped id.
        """
        path = None if input_path is None else str(input_path)
        row = (name, self.tool, path, input_sha256, kind, self._run_id, self._run_id)
        now = _iso(self.clock())

        def write(connection: sqlite3.Connection) -> int:
            connection.execute(
                "INSERT INTO documents (doc_id, name, tool, input_path, "
                "input_sha256, kind, first_run_id, last_run_id, updated_at) "
                f"VALUES ({_NEXT_DOC_ID}, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (name) DO UPDATE SET "
                "input_path = COALESCE(excluded.input_path, documents.input_path), "
                "input_sha256 = COALESCE(excluded.input_sha256, "
                "documents.input_sha256), "
                "kind = COALESCE(excluded.kind, documents.kind)",
                (*row, now),
            )
            (doc_id,) = connection.execute(
                "SELECT doc_id FROM documents WHERE name = ?", (name,)
            ).fetchone()
            return int(doc_id)

        return await self._write(write)

    async def rename_document(
        self, old: str, new: str, *, input_path: Path | str | None = None
    ) -> int:
        """Rename document *old* to *new* in one update; return its id."""
        path = None if input_path is None else str(input_path)
        now = _iso(self.clock())

        def write(connection: sqlite3.Connection) -> int:
            cursor = connection.execute(
                "UPDATE documents SET name = ?, "
                "input_path = COALESCE(?, input_path), updated_at = ? "
                "WHERE name = ? RETURNING doc_id",
                (new, path, now, old),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(f"No document named {old!r} in {self.path}")
            return int(row[0])

        return await self._write(write)

    async def set_image_settings(
        self, lease: Lease, record: Mapping[str, Any] | None
    ) -> None:
        """Record the image settings a document's units were produced with."""
        values = (_json(record), lease.doc_id)
        await self._leased(
            lease,
            lambda connection: connection.execute(
                "UPDATE documents SET image_settings = ? WHERE doc_id = ?", values
            ),
        )

    # --------------------------------------------------------------- leases

    async def acquire(self, doc_id: int) -> Lease | None:
        """Lease document *doc_id* for this run; None while another holds it.

        The lease is taken when it is free, expired or this run's own, and
        every acquisition increments the document's lease generation.
        """
        holder = self._require_run()
        clock, ttl = self.clock, self.lease_ttl

        def write(connection: sqlite3.Connection) -> Lease | None:
            now = clock()
            params = {"holder": holder, "until": now + ttl, "doc_id": doc_id}
            cursor = connection.execute(_ACQUIRE, {**params, "now": now})
            row = connection.execute(
                "SELECT lease_generation FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"No document {doc_id} in {self.path}")
            if cursor.rowcount == 0:
                return None
            return Lease(doc_id, holder, int(row[0]), now + ttl)

        lease = await self._write(write)
        if lease is not None:
            self._held[doc_id] = lease
        return lease

    async def renew(self, lease: Lease) -> Lease:
        """Extend *lease*; raise :class:`LeaseLostError` once it has expired.

        Expiry is judged by the clock when the renewal's transaction runs, and
        the new expiry counts from that reading; an expired lease is not
        renewed, so its holder acquires the document again.
        """
        clock, ttl = self.clock, self.lease_ttl

        def write(connection: sqlite3.Connection) -> Lease:
            now = clock()
            cursor = connection.execute(
                "UPDATE documents SET lease_until = ? WHERE doc_id = ? "
                "AND lease_run_id = ? AND lease_generation = ? AND lease_until > ?",
                (now + ttl, lease.doc_id, lease.holder, lease.generation, now),
            )
            if cursor.rowcount == 0:
                raise LeaseLostError(
                    f"The lease on document {lease.doc_id} cannot be renewed: "
                    "it expired or passed to another holder.",
                    lease,
                )
            return Lease(lease.doc_id, lease.holder, lease.generation, now + ttl)

        renewed = await self._write(write)
        held = self._held.get(lease.doc_id)
        if held is not None and held.generation == lease.generation:
            self._held[lease.doc_id] = renewed
        return renewed

    async def release(self, lease: Lease) -> None:
        """Give *lease* up if this holder still has it.

        The lease leaves the held leases before its write is queued, so the
        heartbeat no longer renews it once the release has begun.
        """
        values = (lease.doc_id, lease.holder, lease.generation)
        self._forget(lease)
        await self._write(
            lambda connection: connection.execute(
                "UPDATE documents SET lease_run_id = NULL, lease_until = NULL "
                "WHERE doc_id = ? AND lease_run_id = ? AND lease_generation = ?",
                values,
            )
        )

    async def _release_run_leases(self, holder: str) -> None:
        """Give up every lease *holder* still has, recorded or not."""
        self._held.clear()
        await self._write(
            lambda connection: connection.execute(
                "UPDATE documents SET lease_run_id = NULL, lease_until = NULL "
                "WHERE lease_run_id = ?",
                (holder,),
            )
        )

    def held_leases(self) -> dict[int, Lease]:
        """Return the leases this store holds, by document id."""
        return dict(self._held)

    def heartbeat(
        self,
        *,
        interval: float = HEARTBEAT_INTERVAL,
        on_lost: Callable[[Lease], None] | None = None,
    ) -> LeaseKeeper:
        """Return a keeper that renews the held leases every *interval* seconds."""
        return LeaseKeeper(self, interval=interval, on_lost=on_lost)

    # ---------------------------------------------------------------- units

    async def plan_units(
        self,
        lease: Lease,
        kind: str,
        units: Sequence[PlannedUnit],
        *,
        model: str,
        provider: str,
    ) -> int:
        """Create the expected units as ``pending``; keep existing rows.

        Returns the number of units created.
        """
        run_id = self._require_run()
        now = _iso(self.clock())
        rows = [
            (
                lease.doc_id,
                kind,
                unit.idx,
                unit.source_index,
                unit.label,
                self.tool,
                model,
                provider,
                run_id,
                now,
            )
            for unit in units
        ]

        def write(connection: sqlite3.Connection) -> int:
            before = connection.total_changes
            connection.executemany(_PLAN_UNIT, rows)
            created = connection.total_changes - before
            _touch(connection, lease.doc_id, run_id, now)
            return created

        return await self._leased(lease, write)

    async def write_unit(
        self,
        lease: Lease,
        result: UnitResult,
        *,
        extra: Callable[[sqlite3.Connection], object] | None = None,
    ) -> None:
        """Commit one unit's state and the document's counts in one transaction.

        *extra* runs first in the same transaction, for a tool's content row
        of the unit, so that the row and the unit commit or roll back together.
        """
        run_id = self._require_run()
        now = _iso(self.clock())
        params = _unit_params(lease.doc_id, result, self.tool, run_id, now)

        def write(connection: sqlite3.Connection) -> None:
            if extra is not None:
                extra(connection)
            connection.execute(_UPSERT_UNIT, params)
            _touch(connection, lease.doc_id, run_id, now)

        await self._leased(lease, write)

    async def reset_units(
        self,
        lease: Lease,
        *,
        kind: str | None = None,
        idxs: Sequence[int] | None = None,
        statuses: Sequence[str] | None = None,
    ) -> int:
        """Return matching units to ``pending``; responses stay. Returns the count."""
        run_id = self._require_run()
        now = _iso(self.clock())
        where = ["doc_id = ?"]
        params: list[Any] = [run_id, now, lease.doc_id]
        if kind is not None:
            where.append("kind = ?")
            params.append(kind)
        if idxs is not None:
            where.append("idx IN (SELECT value FROM json_each(?))")
            params.append(json.dumps(list(idxs)))
        if statuses is not None:
            where.append("status IN (SELECT value FROM json_each(?))")
            params.append(json.dumps(list(statuses)))
        sql = (
            "UPDATE units SET status = 'pending', outcome = NULL, content = NULL, "
            "usage = NULL, error = NULL, error_type = NULL, provenance = NULL, "
            "response_id = NULL, attempts = 0, stale = 0, run_id = ?, "
            f"updated_at = ? WHERE {' AND '.join(where)}"
        )

        def write(connection: sqlite3.Connection) -> int:
            count = connection.execute(sql, params).rowcount
            _touch(connection, lease.doc_id, run_id, now)
            return int(count)

        return await self._leased(lease, write)

    async def delete_units(
        self, lease: Lease, kind: str, *, count: int | None = None
    ) -> int:
        """Delete the units of every kind but *kind*; responses stay.

        With *count*, the units of *kind* from index *count* on go too. The
        requests of settled submissions go with the units. A tool's own
        rows that reference a unit are the caller's to delete first.
        Returns the number of units deleted.

        Raises:
            SubmissionStateError: A ``submitting``, ``submitted`` or
                ``unresolved`` submission holds one of the units; nothing
                is deleted.
        """
        run_id = self._require_run()
        now = _iso(self.clock())
        values = {"doc": lease.doc_id, "kind": kind, "count": count}
        chosen = "doc_id = :doc AND (kind != :kind OR idx >= :count)"
        holders = (
            "SELECT DISTINCT r.submission_id FROM submission_requests AS r "
            "JOIN submissions AS s ON s.submission_id = r.submission_id "
            f"WHERE s.status IN ({_choices(BLOCKING_SUBMISSIONS)}) "
            "AND r.doc_id = :doc AND (r.kind != :kind OR r.idx >= :count) "
            "ORDER BY r.submission_id"
        )

        def write(connection: sqlite3.Connection) -> int:
            held = [str(row[0]) for row in connection.execute(holders, values)]
            if held:
                raise SubmissionStateError(
                    f"Submission {', '.join(held)} still holds units of document "
                    f"{lease.doc_id}; run batch check, or abandon it, first."
                )
            connection.execute(
                f"DELETE FROM submission_requests WHERE {chosen}", values
            )
            deleted = connection.execute(f"DELETE FROM units WHERE {chosen}", values)
            _touch(connection, lease.doc_id, run_id, now)
            return int(deleted.rowcount)

        return await self._leased(lease, write)

    async def set_units_stale(
        self, lease: Lease, kind: str, idxs: Sequence[int], *, stale: bool = True
    ) -> int:
        """Flag units whose inputs changed; returns the number of units flagged."""
        values = (int(stale), lease.doc_id, kind, json.dumps(list(idxs)))
        return await self._leased(
            lease,
            lambda connection: int(
                connection.execute(
                    "UPDATE units SET stale = ? WHERE doc_id = ? AND kind = ? "
                    "AND idx IN (SELECT value FROM json_each(?))",
                    values,
                ).rowcount
            ),
        )

    async def write(self, lease: Lease, func: Callable[[sqlite3.Connection], T]) -> T:
        """Run *func* in one write transaction that first checks *lease*.

        This is the leased write the store's own methods use, for a tool's
        content tables: *func* runs on the writer's connection after the lease
        check and commits with it, or rolls back when it raises.
        """
        return await self._leased(lease, func)

    # ------------------------------------------------------------ responses

    async def append_response(self, response: ResponseRow) -> int:
        """Append one provider attempt in its own transaction; return its id.

        No lease is checked: response rows always stay.
        """
        params = _response_params(
            response, self.tool, self._require_run(), _iso(self.clock())
        )

        def write(connection: sqlite3.Connection) -> int:
            cursor = connection.execute(_INSERT_RESPONSE, params)
            return int(cursor.lastrowid or 0)

        return await self._write(write)

    def recorder(self) -> StoreRecorder:
        """Return an attempt recorder that appends to this store."""
        return StoreRecorder(self)

    # -------------------------------------------------------------- outputs

    def _path_key(self, path: Path) -> str:
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(self.path.resolve().parent).as_posix()
        except ValueError:
            return str(resolved)

    async def register_output(
        self,
        lease: Lease,
        path: Path,
        *,
        format: str,
        sha256: str | None = None,
    ) -> None:
        """Record an existing rendered file with its hash."""
        digest = sha256 if sha256 is not None else file_sha256(path)
        if digest is None:
            raise FileNotFoundError(f"No output file at {path}")
        values = (
            lease.doc_id,
            self._path_key(path),
            format,
            digest,
            _iso(self.clock()),
            self._run_id,
        )
        await self._leased(
            lease, lambda connection: connection.execute(_OUTPUT_DONE, values)
        )

    def _plan_file(
        self,
        target: RenderTarget,
        text: str | bytes,
        recorded: Mapping[str, str | None],
        force: bool,
        report: RenderReport,
    ) -> _PlannedFile | None:
        data = text if isinstance(text, bytes) else text.encode("utf-8")
        key = self._path_key(target.path)
        try:
            current = file_sha256(target.path)
        except OSError as exc:
            return _PlannedFile(
                target=target,
                key=key,
                sha256=hashlib.sha256(data).hexdigest(),
                current=None,
                recorded=recorded.get(key),
                error=(
                    f"{target.path} could not be read ({exc}); the data is stored "
                    "and the output is marked stale."
                ),
            )
        item = _PlannedFile(
            target=target,
            key=key,
            sha256=hashlib.sha256(data).hexdigest(),
            current=current,
            recorded=recorded.get(key),
        )
        edited = current is not None and current not in (item.recorded, item.sha256)
        if edited and not force:
            report.kept.append(target.path)
            report.warnings.append(
                f"{target.path} was changed outside the tool and is left as it "
                "is; render with force to replace it."
            )
            return None
        if current == item.sha256:
            return item
        temp = target.path.with_name(f".{target.path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            target.path.parent.mkdir(parents=True, exist_ok=True)
            temp.write_bytes(data)
        except OSError as exc:
            item.error = (
                f"{target.path} could not be written ({exc}); the data is stored "
                "and the output is marked stale."
            )
            with contextlib.suppress(OSError):
                temp.unlink(missing_ok=True)
            return item
        item.temp = temp
        return item

    async def render(
        self,
        lease: Lease,
        targets: Sequence[RenderTarget],
        *,
        force: bool = False,
        hook: Callable[[], None] | None = None,
    ) -> RenderReport:
        """Render a document's files from the store and record them.

        Each rendering goes to a temporary file first; the targets are
        replaced while the transaction that checks the lease and updates
        ``outputs`` holds the write lock. A file that matches neither its
        recorded hash nor the new rendering is left alone unless *force*; so is
        a file whose contents changed between that check and the replacement,
        which the transaction checks again. A
        target that cannot be read is not replaced, since the overwrite guard
        cannot judge it. That case and a failed replacement mark the output
        stale and add a warning. *hook*
        runs inside that transaction, before the replacements (for tests).
        """
        run_id = self._require_run()
        doc_id = lease.doc_id

        def texts(connection: sqlite3.Connection) -> list[str | bytes]:
            return [target.renderer(connection, doc_id) for target in targets]

        def recorded_hashes(connection: sqlite3.Connection) -> dict[str, str | None]:
            rows = connection.execute(
                "SELECT path, sha256 FROM outputs WHERE doc_id = ?", (doc_id,)
            ).fetchall()
            return {str(path): digest for path, digest in rows}

        rendered = self._read(texts)
        recorded = self._read(recorded_hashes)
        report = RenderReport()
        items: list[_PlannedFile] = []
        try:
            for target, text in zip(targets, rendered, strict=True):
                item = self._plan_file(target, text, recorded, force, report)
                if item is not None:
                    items.append(item)
            if not items:
                return report
            now = _iso(self.clock())

            def write(connection: sqlite3.Connection) -> list[str | None]:
                if hook is not None:
                    hook()
                return _commit_render(connection, doc_id, items, run_id, now, force)

            errors = await self._leased(lease, write)
        finally:
            for item in items:
                if item.temp is not None:
                    with contextlib.suppress(OSError):
                        item.temp.unlink(missing_ok=True)
        for item, error in zip(items, errors, strict=True):
            if item.kept:
                report.kept.append(item.target.path)
                report.warnings.append(
                    f"{item.target.path} changed while it was rendered and is left "
                    "as it is; render with force to replace it."
                )
            elif error is not None:
                report.stale.append(item.target.path)
                report.warnings.append(error)
            elif item.temp is None:
                report.unchanged.append(item.target.path)
            else:
                report.written.append(item.target.path)
        return report

    # ---------------------------------------------------------- submissions

    def _leases_for(self, doc_ids: Sequence[int]) -> list[Lease]:
        leases = []
        for doc_id in sorted(set(doc_ids)):
            lease = self._held.get(doc_id)
            if lease is None:
                raise LeaseLostError(f"Document {doc_id} is not leased by this run.")
            leases.append(lease)
        return leases

    async def open_submission(
        self,
        *,
        provider: str,
        model: str,
        units: Sequence[UnitKey],
        part: int = 0,
        custom_ids: Sequence[str] | None = None,
    ) -> Submission:
        """Commit a ``submitting`` submission and its requests before any upload.

        Custom ids default to ids scoped to the submission (its id prefix and
        the request's position); the store must lease every document.
        """
        run_id = self._require_run()
        if not units:
            raise ValueError("A submission needs at least one unit.")
        submission_id = str(uuid.uuid4())
        ids = (
            list(custom_ids)
            if custom_ids is not None
            else [f"{submission_id[:8]}-{number:05d}" for number in range(len(units))]
        )
        if len(ids) != len(units) or len(set(ids)) != len(ids):
            raise ValueError("Custom ids must be unique, one per unit.")
        leases = self._leases_for([unit.doc_id for unit in units])
        clock = self.clock
        submission = (
            submission_id,
            self.tool,
            provider,
            model,
            run_id,
            part,
            len(units),
        )
        requests = [
            (submission_id, custom_id, unit.doc_id, unit.kind, unit.idx)
            for custom_id, unit in zip(ids, units, strict=True)
        ]

        def write(connection: sqlite3.Connection) -> None:
            now = clock()
            for lease in leases:
                _check_lease(connection, lease, now)
            connection.execute(
                "INSERT INTO submissions (submission_id, tool, provider, model, "
                "run_id, part, request_count, intent_at, updated_at, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'submitting')",
                (*submission, _iso(now), _iso(now)),
            )
            connection.executemany(
                "INSERT INTO submission_requests (submission_id, custom_id, doc_id, "
                "kind, idx) VALUES (?, ?, ?, ?, ?)",
                requests,
            )

        await self._write(write)
        return self.submission(submission_id)

    async def _update_submission(
        self,
        submission_id: str,
        allowed: tuple[str, ...],
        values: Mapping[str, Any],
        after: Callable[[sqlite3.Connection, str], None] | None = None,
    ) -> Submission:
        held = dict(self._held)
        clock = self.clock
        columns = {**values, "updated_at": _NOW}
        assignments = ", ".join(f"{column} = ?" for column in columns)
        sql = f"UPDATE submissions SET {assignments} WHERE submission_id = ?"

        def write(connection: sqlite3.Connection) -> None:
            now = clock()
            stamp = _iso(now)
            row = connection.execute(
                "SELECT status FROM submissions WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"No submission {submission_id} in {self.path}")
            if row[0] not in allowed:
                raise SubmissionStateError(
                    f"Submission {submission_id} is {row[0]}; this change needs "
                    f"status {' or '.join(allowed)}."
                )
            _check_submission_leases(connection, submission_id, held, now)
            params = [stamp if value is _NOW else value for value in columns.values()]
            connection.execute(sql, (*params, submission_id))
            if after is not None:
                after(connection, stamp)

        await self._write(write)
        return self.submission(submission_id)

    async def _set_status(
        self,
        submission_id: str,
        status: str,
        values: Mapping[str, Any],
        after: Callable[[sqlite3.Connection, str], None] | None = None,
    ) -> Submission:
        changes = {key: value for key, value in values.items() if value is not None}
        return await self._update_submission(
            submission_id, _TRANSITIONS[status], {**changes, "status": status}, after
        )

    async def record_upload(self, submission_id: str, upload_ref: str) -> Submission:
        """Record the provider's reference of the uploaded request file."""
        return await self._update_submission(
            submission_id, ("submitting",), {"upload_ref": upload_ref}
        )

    async def record_batch(
        self, submission_id: str, batch_id: str, *, provider_status: str | None = None
    ) -> Submission:
        """Record the provider batch id: the submission becomes ``submitted``.

        An ``unresolved`` submission is adopted the same way.
        """
        return await self._set_status(
            submission_id,
            "submitted",
            {"batch_id": batch_id, "provider_status": provider_status},
        )

    async def record_check(
        self,
        submission_id: str,
        *,
        provider_status: str | None,
        counts: Mapping[str, Any] | None = None,
    ) -> Submission:
        """Record the provider status and counts seen by a batch check."""
        values = {
            "provider_status": provider_status,
            "counts": _json(counts),
            "last_checked_at": _NOW,
        }
        return await self._update_submission(submission_id, ("submitted",), values)

    async def mark_unresolved(self, submission_id: str, error: str) -> Submission:
        """Mark a submission whose batch create has no confirmed outcome."""
        return await self._set_status(submission_id, "unresolved", {"error": error})

    async def mark_failed(self, submission_id: str, error: str) -> Submission:
        """Mark a submission that certainly produced no usable batch."""
        return await self._set_status(submission_id, "failed", {"error": error})

    async def mark_finalized(
        self,
        submission_id: str,
        *,
        provider_status: str | None = None,
        counts: Mapping[str, Any] | None = None,
    ) -> Submission:
        """Mark a submission whose results are stored."""
        values = {
            "provider_status": provider_status,
            "counts": _json(counts),
            "finalized_at": _NOW,
        }
        return await self._set_status(submission_id, "finalized", values)

    async def mark_cancelled(
        self, submission_id: str, *, provider_status: str | None = None
    ) -> Submission:
        """Mark a submission whose provider batch was cancelled."""
        values = {"provider_status": provider_status}
        return await self._set_status(submission_id, "cancelled", values)

    async def abandon_submission(self, submission_id: str) -> Submission:
        """Abandon an unresolved or submitted submission.

        Its units that are not done become pending. The provider batch of a
        submitted one is not touched.
        """
        run_id = self._run_id

        def after(connection: sqlite3.Connection, stamp: str) -> None:
            connection.execute(
                "UPDATE units SET status = 'pending', outcome = NULL, "
                "content = NULL, error = NULL, error_type = NULL, "
                "response_id = NULL, run_id = COALESCE(?, run_id), updated_at = ? "
                "WHERE status != 'done' AND (doc_id, kind, idx) IN ("
                "SELECT doc_id, kind, idx FROM submission_requests "
                "WHERE submission_id = ?)",
                (run_id, stamp, submission_id),
            )
            for (doc_id,) in connection.execute(
                "SELECT DISTINCT doc_id FROM submission_requests "
                "WHERE submission_id = ?",
                (submission_id,),
            ).fetchall():
                _touch(connection, int(doc_id), run_id, stamp)

        return await self._set_status(submission_id, "abandoned", {}, after)

    async def record_requests(
        self, submission_id: str, bodies: Mapping[str, Mapping[str, Any]]
    ) -> None:
        """Keep the request body sent for each custom id of a submission.

        Image and audio payloads are replaced by placeholders
        (:func:`~.capture.redact_payloads`). Finalization stores each body
        as the ``request`` of the unit's response row.
        """
        rows = [
            (_json(redact_payloads(body)), submission_id, custom_id)
            for custom_id, body in bodies.items()
        ]

        def write(connection: sqlite3.Connection) -> None:
            connection.executemany(
                "UPDATE submission_requests SET request = ? "
                "WHERE submission_id = ? AND custom_id = ?",
                rows,
            )

        await self._write(write)

    async def submit(
        self,
        *,
        provider: str,
        model: str,
        units: Sequence[UnitKey],
        upload: Callable[[Submission], Awaitable[str | None]],
        create: Callable[[Submission], Awaitable[str]],
        part: int = 0,
    ) -> Submission:
        """Run one batch submission in the committed order.

        The submission and its requests are committed first; *upload* then
        sends the request file and returns its reference (None when the
        provider takes requests inline); *create* creates the batch and
        returns the provider batch id. A failed upload, or a create that
        raises :class:`SubmissionRejectedError`, marks the submission failed;
        any other error from *create* leaves the outcome unknown and marks it
        unresolved, so its units are never resubmitted automatically.
        """
        submission = await self.open_submission(
            provider=provider, model=model, units=units, part=part
        )
        submission_id = submission.submission_id
        try:
            upload_ref = await upload(submission)
        except Exception as exc:
            await self.mark_failed(submission_id, f"upload failed: {exc}")
            raise
        if upload_ref is not None:
            submission = await self.record_upload(submission_id, upload_ref)
        try:
            batch_id = await create(submission)
        except SubmissionRejectedError as exc:
            await self.mark_failed(submission_id, f"batch create rejected: {exc}")
            raise
        except Exception as exc:
            await self.mark_unresolved(
                submission_id, f"batch create without a confirmed response: {exc}"
            )
            raise
        return await self.record_batch(submission_id, batch_id)

    async def mark_stale_submissions(self) -> list[str]:
        """Mark submissions left ``submitting`` without a batch id as unresolved.

        A submission counts as left behind when another run opened it and no
        live holder other than this run leases any of its documents. Returns
        the ids marked.
        """
        run_id = self._run_id
        clock = self.clock

        def write(connection: sqlite3.Connection) -> list[str]:
            now = clock()
            ids = [
                str(row[0])
                for row in connection.execute(
                    _STALE_CANDIDATES, {"run": run_id, "now": now}
                )
            ]
            connection.executemany(
                "UPDATE submissions SET status = 'unresolved', "
                "error = COALESCE(error, ?), updated_at = ? WHERE submission_id = ?",
                [
                    (
                        "left submitting without a provider batch id",
                        _iso(now),
                        submission_id,
                    )
                    for submission_id in ids
                ],
            )
            return ids

        return await self._write(write)

    # ----------------------------------------------------------------- drop

    async def drop(
        self, doc_ids: Sequence[int], *, dry_run: bool = False
    ) -> DropReport:
        """Delete documents with every row keyed to them and their rendered files.

        A document is refused while a run other than this store's own holds
        a live lease on it, or while a ``submitting``, ``submitted`` or
        ``unresolved`` submission holds any of its units; an id that names no
        document is refused as well. The rest are deleted with their units,
        responses, outputs, notes, the requests of resolved submissions and
        the rows of every tool table with a ``doc_id`` column, so the next run
        processes them from scratch. Submission and run rows stay.

        A rendered file is deleted when its contents still match a hash
        recorded for it. A file that differs, is missing or unreadable, cannot
        be deleted, or is still recorded for a document that stays, is kept
        and named in the report.

        One write transaction classifies the documents, deletes their rows
        and then deletes the files before it commits, so no other run can
        lease a document or render a file in between. The files go before the
        commit on purpose: should the commit fail, the rows still describe
        the document and its missing files, which a render writes again and a
        second drop finishes; a file left behind by a committed drop would be
        recorded nowhere, and a later render would keep it as changed outside
        the tool. A dry run classifies and checks the files in a read
        transaction and changes nothing; it needs no open writer.
        """
        holder = self._run_id
        clock = self.clock
        base = self.path.resolve().parent
        if dry_run:
            plan = self._read(
                lambda connection: _plan_drop(connection, doc_ids, holder, clock())
            )
            deleted, kept = _settle_files(plan.outputs, base, delete=False, gone=set())
            return plan.report(True, deleted, kept)
        # The writer reruns a transaction whose commit stayed busy; the files
        # an earlier attempt deleted then count as deleted, not missing.
        gone: set[Path] = set()

        def write(
            connection: sqlite3.Connection,
        ) -> tuple[_DropPlan, list[Path], list[KeptFile]]:
            plan = _plan_drop(connection, doc_ids, holder, clock())
            if plan.dropped:
                _delete_documents(connection, [doc_id for doc_id, _ in plan.dropped])
            deleted, kept = _settle_files(plan.outputs, base, delete=True, gone=gone)
            return plan, deleted, kept

        plan, deleted, kept = await self._write(write)
        for doc_id, _ in plan.dropped:
            self._held.pop(doc_id, None)
        return plan.report(False, deleted, kept)

    # ---------------------------------------------------------------- reads

    def submission(self, submission_id: str) -> Submission:
        """Return the submission *submission_id*; ``KeyError`` when missing."""
        rows = self._read(
            lambda connection: _dicts(
                connection,
                "SELECT * FROM submissions WHERE submission_id = ?",
                (submission_id,),
            )
        )
        if not rows:
            raise KeyError(f"No submission {submission_id} in {self.path}")
        return Submission(**_pick(rows[0], Submission))

    def submissions(self, statuses: Sequence[str] | None = None) -> list[Submission]:
        """Return the submissions, optionally only those with *statuses*."""
        sql = "SELECT * FROM submissions"
        params: list[Any] = []
        if statuses is not None:
            sql += " WHERE status IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(list(statuses)))
        sql += " ORDER BY intent_at, part, submission_id"
        rows = self._read(lambda connection: _dicts(connection, sql, params))
        return [Submission(**_pick(row, Submission)) for row in rows]

    def open_submissions(self) -> list[Submission]:
        """Return the submissions still submitting or submitted."""
        return self.submissions(OPEN_SUBMISSIONS)

    def unresolved_submissions(self) -> list[Submission]:
        """Return the submissions whose batch create has no confirmed outcome."""
        return self.submissions(("unresolved",))

    def stale_submissions(self) -> list[Submission]:
        """Return the submissions still submitting without a provider batch id."""
        return [
            submission
            for submission in self.submissions(("submitting",))
            if submission.batch_id is None
        ]

    def submission_requests(self, submission_id: str) -> dict[str, UnitKey]:
        """Return the units of a submission by custom id."""
        rows = self._read(
            lambda connection: connection.execute(
                "SELECT custom_id, doc_id, kind, idx FROM submission_requests "
                "WHERE submission_id = ? ORDER BY rowid",
                (submission_id,),
            ).fetchall()
        )
        return {
            str(custom_id): UnitKey(int(doc_id), str(kind), int(idx))
            for custom_id, doc_id, kind, idx in rows
        }

    def submission_bodies(self, submission_id: str) -> dict[str, dict[str, Any]]:
        """Return the request bodies kept for a submission, by custom id."""
        rows = self._read(
            lambda connection: connection.execute(
                "SELECT custom_id, request FROM submission_requests "
                "WHERE submission_id = ? AND request IS NOT NULL",
                (submission_id,),
            ).fetchall()
        )
        bodies: dict[str, dict[str, Any]] = {}
        for custom_id, text in rows:
            body = json.loads(text)
            if isinstance(body, dict):
                bodies[str(custom_id)] = body
        return bodies

    def blocked_units(self, doc_id: int) -> list[UnitKey]:
        """Return a document's units held by open or unresolved submissions."""
        rows = self._read(
            lambda connection: connection.execute(_BLOCKED_UNITS, (doc_id,)).fetchall()
        )
        return [UnitKey(doc_id, str(kind), int(idx)) for kind, idx in rows]

    def document(self, document: int | str) -> DocumentRow | None:
        """Return the document with this id or name, or None."""
        column = "doc_id" if isinstance(document, int) else "name"
        rows = self._read(
            lambda connection: _dicts(
                connection, f"SELECT * FROM documents WHERE {column} = ?", (document,)
            )
        )
        return DocumentRow(**_pick(rows[0], DocumentRow)) if rows else None

    def documents(self) -> list[DocumentRow]:
        """Return every document, ordered by name."""
        rows = self._read(
            lambda connection: _dicts(
                connection, "SELECT * FROM documents ORDER BY name"
            )
        )
        return [DocumentRow(**_pick(row, DocumentRow)) for row in rows]

    def units(self, doc_id: int, *, kind: str | None = None) -> list[UnitRow]:
        """Return a document's units, ordered by kind and index."""
        sql = "SELECT * FROM units WHERE doc_id = ?"
        params: list[Any] = [doc_id]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        rows = self._read(
            lambda connection: _dicts(connection, sql + " ORDER BY kind, idx", params)
        )
        return [UnitRow(**_pick(row, UnitRow)) for row in rows]

    def failed_units(
        self, *, doc_id: int | None = None, kind: str | None = None
    ) -> list[UnitRow]:
        """Return the failed units, optionally of one document or kind."""
        sql = "SELECT * FROM units WHERE status = 'failed'"
        params: list[Any] = []
        if doc_id is not None:
            sql += " AND doc_id = ?"
            params.append(doc_id)
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        rows = self._read(
            lambda connection: _dicts(
                connection, sql + " ORDER BY doc_id, kind, idx", params
            )
        )
        return [UnitRow(**_pick(row, UnitRow)) for row in rows]

    def responses(
        self, *, unit: UnitKey | None = None, submission_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return response rows in append order, for a unit or a submission."""
        sql = "SELECT * FROM responses WHERE 1 = 1"
        params: list[Any] = []
        if unit is not None:
            sql += " AND doc_id = ? AND kind = ? AND idx = ?"
            params.extend((unit.doc_id, unit.kind, unit.idx))
        if submission_id is not None:
            sql += " AND submission_id = ?"
            params.append(submission_id)
        return self._read(
            lambda connection: _dicts(connection, sql + " ORDER BY response_id", params)
        )

    def outputs(self, doc_id: int) -> list[OutputRow]:
        """Return a document's registered outputs."""
        rows = self._read(
            lambda connection: _dicts(
                connection,
                "SELECT * FROM outputs WHERE doc_id = ? ORDER BY path",
                (doc_id,),
            )
        )
        return [OutputRow(**_pick(row, OutputRow)) for row in rows]

    def notes(self, *, doc_id: int | None = None) -> list[dict[str, Any]]:
        """Return the notes in order, optionally of one document."""
        sql = "SELECT * FROM notes"
        params: list[Any] = []
        if doc_id is not None:
            sql += " WHERE doc_id = ?"
            params.append(doc_id)
        return self._read(
            lambda connection: _dicts(connection, sql + " ORDER BY note_id", params)
        )

    def run(self, run_id: str | None = None) -> dict[str, Any] | None:
        """Return the ``runs`` row of *run_id* (default: this store's run)."""
        key = run_id if run_id is not None else self._run_id
        rows = self._read(
            lambda connection: _dicts(
                connection, "SELECT * FROM runs WHERE run_id = ?", (key,)
            )
        )
        return rows[0] if rows else None

    def resume_state(self, document: int | str) -> ResumeState | None:
        """Return what resume needs of a document, or None when it is unknown."""
        return self._read(lambda connection: _resume_state(connection, document))


# =============================================================================
# Store helpers outside the class
# =============================================================================


def _check_submission_leases(
    connection: sqlite3.Connection,
    submission_id: str,
    held: Mapping[int, Lease],
    now: float,
) -> None:
    rows = connection.execute(
        "SELECT DISTINCT doc_id FROM submission_requests WHERE submission_id = ?",
        (submission_id,),
    ).fetchall()
    for (doc_id,) in rows:
        lease = held.get(int(doc_id))
        if lease is None:
            raise LeaseLostError(f"Document {doc_id} is not leased by this run.")
        _check_lease(connection, lease, now)


def _resume_state(
    connection: sqlite3.Connection, document: int | str
) -> ResumeState | None:
    column = "doc_id" if isinstance(document, int) else "name"
    rows = _dicts(
        connection, f"SELECT * FROM documents WHERE {column} = ?", (document,)
    )
    if not rows:
        return None
    row = rows[0]
    doc_id = int(row["doc_id"])

    def keys(sql: str, params: tuple[Any, ...]) -> tuple[UnitKey, ...]:
        return tuple(
            UnitKey(doc_id, str(kind), int(idx))
            for kind, idx in connection.execute(sql, params).fetchall()
        )

    counts = {
        str(status): int(count)
        for status, count in connection.execute(UNIT_COUNTS_SQL, (doc_id,))
    }
    return ResumeState(
        doc_id=doc_id,
        name=str(row["name"]),
        status=str(row["status"]),
        counts=counts,
        pending=keys(_UNITS_WITH_STATUS, (doc_id, "pending")),
        failed=keys(_UNITS_WITH_STATUS, (doc_id, "failed")),
        blocked=keys(_BLOCKED_UNITS, (doc_id,)),
        image_settings=row["image_settings"],
        lease_run_id=row["lease_run_id"],
        lease_until=row["lease_until"],
    )


@dataclass(frozen=True)
class _DropPlan:
    """The documents a drop deletes or refuses, and their recorded outputs.

    *outputs* maps each recorded path to its recorded hashes and whether a
    document that stays records it too.
    """

    dropped: tuple[tuple[int, str], ...]
    refused: tuple[DropRefusal, ...]
    outputs: Mapping[str, tuple[frozenset[str], bool]]

    def report(
        self, dry_run: bool, deleted: Sequence[Path], kept: Sequence[KeptFile]
    ) -> DropReport:
        return DropReport(
            dry_run=dry_run,
            dropped=tuple(sorted(name for _, name in self.dropped)),
            refused=self.refused,
            deleted_files=tuple(deleted),
            kept_files=tuple(kept),
        )


def _plan_drop(
    connection: sqlite3.Connection,
    doc_ids: Sequence[int],
    holder: str | None,
    now: float,
) -> _DropPlan:
    """Classify *doc_ids* for a drop at *now*; *holder* is the dropping run."""
    wanted = sorted(set(doc_ids))
    rows = {
        int(row[0]): row
        for row in connection.execute(_DROP_DOCUMENTS, (json.dumps(wanted),))
    }
    dropped: list[tuple[int, str]] = []
    refused: list[DropRefusal] = []
    for doc_id in wanted:
        if doc_id not in rows:
            refused.append(DropRefusal(doc_id, None, "no such document in the store"))
            continue
        _, name, lease_holder, until = rows[doc_id]
        reasons = []
        if (
            lease_holder is not None
            and lease_holder != holder
            and until is not None
            and until > now
        ):
            reasons.append(f"run {lease_holder} leases it until {_iso(until)}")
        reasons.extend(
            f"batch submission {submission_id} ({status}) holds its units"
            for submission_id, status, _ in connection.execute(
                _DROP_SUBMISSIONS, (doc_id,)
            )
        )
        if reasons:
            refused.append(DropRefusal(doc_id, str(name), "; ".join(reasons)))
        else:
            dropped.append((doc_id, str(name)))
    ids = json.dumps([doc_id for doc_id, _ in dropped])
    hashes: dict[str, set[str]] = {}
    shared: dict[str, bool] = {}
    for path, digest, other in connection.execute(_DROP_OUTPUTS, {"ids": ids}):
        key = str(path)
        hashes.setdefault(key, set())
        if digest is not None:
            hashes[key].add(str(digest))
        shared[key] = shared.get(key, False) or bool(other)
    refused.sort(key=lambda item: (item.name is None, item.name or "", item.doc_id))
    return _DropPlan(
        dropped=tuple(dropped),
        refused=tuple(refused),
        outputs={key: (frozenset(hashes[key]), shared[key]) for key in hashes},
    )


def _doc_tables(connection: sqlite3.Connection) -> list[str]:
    """Return the tables with a ``doc_id`` column, except the three core ones.

    :func:`_delete_documents` empties the core tables itself, in order.
    """
    names = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    return [
        name
        for name in names
        if name not in ("documents", "units", "responses")
        and any(
            row[1] == "doc_id"
            for row in connection.execute(f"PRAGMA table_info({_ident(name)})")
        )
    ]


def _delete_documents(connection: sqlite3.Connection, doc_ids: Sequence[int]) -> None:
    """Delete documents and every row keyed to them, inside the transaction.

    The other tables keyed by ``doc_id`` go first, then units, responses and
    documents; foreign keys are checked at each statement, so a violation
    raises before the drop deletes any file. The trigger that keeps
    responses append-only is dropped for the delete and created again from
    its stored text, in the same transaction, so the committed schema is
    unchanged. The highest id dropped is recorded in ``meta``.
    """
    ids = (json.dumps(list(doc_ids)),)
    connection.execute(
        "INSERT INTO meta (component, version) VALUES (?, ?) "
        "ON CONFLICT (component) DO UPDATE SET "
        "version = max(version, excluded.version)",
        (_DROPPED_MARK, max(doc_ids)),
    )
    for table in _doc_tables(connection):
        connection.execute(f"DELETE FROM {_ident(table)} {_IN_IDS}", ids)
    connection.execute(f"DELETE FROM units {_IN_IDS}", ids)
    trigger = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
        (_NO_DELETE_TRIGGER,),
    ).fetchone()
    if trigger is not None:
        connection.execute(f"DROP TRIGGER {_ident(_NO_DELETE_TRIGGER)}")
    connection.execute(f"DELETE FROM responses {_IN_IDS}", ids)
    if trigger is not None:
        connection.execute(str(trigger[0]))
    connection.execute(f"DELETE FROM documents {_IN_IDS}", ids)


def _settle_files(
    outputs: Mapping[str, tuple[frozenset[str], bool]],
    base: Path,
    *,
    delete: bool,
    gone: set[Path],
) -> tuple[list[Path], list[KeptFile]]:
    """Delete (or, unless *delete*, judge) the recorded output files.

    A relative key is resolved against *base*, the database folder. *gone*
    collects the files deleted, across reruns of the same transaction.
    """
    deleted: list[Path] = []
    kept: list[KeptFile] = []
    for key, (recorded, shared) in outputs.items():
        path = Path(key) if Path(key).is_absolute() else base / key
        if shared:
            kept.append(KeptFile(path, "a document that stays records it too"))
            continue
        try:
            current = file_sha256(path)
        except OSError as exc:
            kept.append(KeptFile(path, f"it could not be read ({exc})"))
            continue
        if current is None:
            if path in gone:
                deleted.append(path)
            else:
                kept.append(KeptFile(path, "it is missing"))
            continue
        if current not in recorded:
            kept.append(KeptFile(path, "it was changed outside the tool"))
            continue
        if delete:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                kept.append(KeptFile(path, f"it could not be deleted ({exc})"))
                continue
            gone.add(path)
        deleted.append(path)
    return deleted, kept


def select_documents(
    documents: Sequence[DocumentRow],
    *,
    select: str | None = None,
    files: Sequence[str] = (),
    select_all: bool = False,
) -> DocumentSelection:
    """Choose stored documents by ``--select``, ``--files`` or ``--all``.

    Exactly one of the three must be given. ``--select`` numbers the
    documents from 1 in name order and otherwise searches their names (see
    :func:`~.selection.select_items`); its out-of-range parts become
    warnings. A ``--files`` entry names a document by its name or by the
    file name of its input path and must name exactly one. The documents
    come back in name order.

    Raises:
        DocumentSelectionError: Not exactly one selection was given,
            ``--select`` is blank or matched no document, or a ``--files``
            entry named none or several.
    """
    if sum((select is not None, bool(files), select_all)) != 1:
        raise DocumentSelectionError("Give exactly one of --select, --files and --all.")
    ordered = sorted(documents, key=lambda document: document.name)
    if select_all:
        return DocumentSelection(tuple(ordered))
    if select is not None:
        if not select.strip():
            raise DocumentSelectionError("--select is blank; name the documents.")
        result = select_items([document.name for document in ordered], select)
        warnings: tuple[str, ...] = ()
        if result.unmatched:
            warnings = (
                "Ignored out-of-range or invalid --select part(s): "
                f"{', '.join(result.unmatched)} (valid: 1-{len(ordered)})",
            )
        picked = result.pick(ordered)
        if not picked:
            raise DocumentSelectionError(
                f"No stored document matches --select {select!r}."
            )
        return DocumentSelection(tuple(picked), warnings)

    def named(document: DocumentRow, entry: str) -> bool:
        if document.name == entry:
            return True
        return document.input_path is not None and (
            Path(document.input_path).name == entry
        )

    matches = {
        entry: [document.name for document in ordered if named(document, entry)]
        for entry in files
    }
    missing = [entry for entry, names in matches.items() if not names]
    if missing:
        listed = ", ".join(repr(entry) for entry in missing)
        raise DocumentSelectionError(f"--files names no stored document: {listed}")
    for entry, names in matches.items():
        if len(names) > 1:
            raise DocumentSelectionError(
                f"--files entry {entry!r} names {len(names)} documents "
                f"({', '.join(names)}); choose them with --select"
            )
    return DocumentSelection(
        tuple(
            document
            for document in ordered
            if any(named(document, entry) for entry in files)
        )
    )


class StoreRecorder:
    """Appends each provider attempt to the store, attributed to the current unit.

    Install it with :func:`~.capture.use_recorder`; the unit comes from
    :func:`unit_context`.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    async def record(self, record: AttemptRecord) -> int | None:
        """Append *record* as one response row and return its id."""
        return await self._store.append_response(
            ResponseRow(
                provider=record.provider,
                model=record.model,
                unit=current_unit(),
                role=record.role,
                attempt=record.attempt,
                request=record.request,
                raw=record.raw,
                raw_kind=record.raw_kind,
                usage=record.usage,
                error=record.error,
                error_type=record.error_type,
                http_status=record.http_status,
                latency=record.latency,
            )
        )


class LeaseKeeper:
    """Renews a store's held leases every *interval* seconds while it runs.

    Use it as ``async with store.heartbeat():``. A lease that can no longer
    be renewed is dropped from the store and passed to *on_lost*.
    """

    def __init__(
        self,
        store: Store,
        *,
        interval: float = HEARTBEAT_INTERVAL,
        on_lost: Callable[[Lease], None] | None = None,
    ) -> None:
        self._store = store
        self._interval = interval
        self._on_lost = on_lost
        self._task: asyncio.Task[None] | None = None
        self.renewals = 0
        self.lost: list[Lease] = []

    async def __aenter__(self) -> LeaseKeeper:
        self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()

    @property
    def running(self) -> bool:
        """Return True while the renewal task runs."""
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start the renewal task on the running loop."""
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="lease-heartbeat")

    async def stop(self) -> None:
        """Stop the renewal task and wait for it to end."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def renew_all(self) -> None:
        """Renew every held lease once.

        Each lease is looked up right before its renewal is queued, with no
        await in between, so a lease released meanwhile is skipped rather
        than reported lost; the writer commits a renewal queued before a
        release first.
        """
        store = self._store
        for doc_id in list(store.held_leases()):
            lease = store.held_leases().get(doc_id)
            if lease is None:
                continue
            try:
                await store.renew(lease)
            except LeaseLostError:
                self.lost.append(lease)
                if self._on_lost is not None:
                    self._on_lost(lease)
            except sqlite3.Error as exc:
                logger.warning(
                    "Lease renewal for document %s failed: %s", lease.doc_id, exc
                )
            else:
                self.renewals += 1

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            await self.renew_all()
