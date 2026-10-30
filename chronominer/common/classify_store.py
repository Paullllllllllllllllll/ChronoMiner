"""The ``classifications`` store component: classifier answers, shadow only.

A tool registers :data:`COMPONENT` beside its own component. Every answer of
a classifier question about one unit is one row of ``classification_rows``;
rows are only appended, and a row whose inputs changed is flagged ``stale``
instead of deleted.

Only entry rows carry a ``doc_id``: an entry belongs to a stored document,
and its rows are written under that document's lease. Page, chunk and
document rows never do; they are keyed by document name, locator and the
input's SHA-256 as last observed, and are written without a lease, so
classifying creates no document row; such a write is refused
(:class:`ChangedInputError`) while a stored document of that name records
another input SHA-256. The store enforces foreign keys; a trigger deletes a
document's rows (by name and ``doc_id``) before any writer deletes the
document, so a drop never fails on them and leaves none behind.

Read contract
=============

Outside code reads the view ``classifications``. Per ``document``,
``unit``, ``locator`` and ``classifier`` it shows only rows of the
classifier revision (``classifier_sha256``) of the latest row, and of those
the latest row per ``question``; stale rows are left out, and so is every
row whose document has a stored ``documents`` row with another
``input_sha256`` (the input changed since). Its columns:

- ``document`` (the tool's document name), ``doc_id`` (the stored document
  of an entry, else NULL), ``unit`` (page, chunk, entry or document),
  ``locator`` (the tool's name of the unit) and ``ordinal`` (its position in
  the document);
- ``input_sha256`` (the input file), ``content_sha256`` (exactly the parts
  sent about the unit) and ``request_sha256`` (the whole request: document,
  units with their locators in order, context and classifier);
- ``classifier``, ``classifier_version``, ``classifier_sha256`` (the
  normalized classifier file) and ``question``;
- ``kind`` (predicate, choice or score), ``value`` (the chosen value of a
  choice as text, the expected 0-based level of a score as a number, NULL
  for a predicate), ``probability`` (a predicate's probability, the chosen
  value's probability, a score's confidence), ``probabilities`` (JSON object
  by value or level label) and ``refusal`` (NULL, or the reason the model
  declined, ``''`` without one; the answer columns are then NULL);
- ``model``, ``dialect``, ``fingerprint`` (the tool's request fingerprint),
  ``run_id``, ``created_at`` and ``reused`` (1 for a row a run copied from an
  earlier answer instead of asking again).

Writes: :func:`record_answers` appends one request's rows in one
transaction. An unleased write first flags stale the document's unleased
rows of every unit kind whose ``input_sha256`` differs, so answers about
two versions of an input never mix, and refuses
(:class:`DocumentChangedError`) when the stored document the run saw at
planning (``seen_doc_id``) is gone or replaced, as after a drop. Inside a
tool's own write transactions,
:func:`mark_stale_for_document` and :func:`mark_stale_for_requests` flag
rows of one document whose inputs changed (every row of a request goes
stale with any of its rows, since its answers were formed together).
:func:`reusable_answers` finds the answers a run may keep; a caller that
keeps them must pass them to :func:`republish_answers`, which appends them
again under its run with the unit's current locator and ordinal, under the
same checks, so the view shows the answers the latest run used.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from typing import Any, Final

from .classifiers import Classifier
from .decision_runner import RequestResult
from .store import Component, Lease, Store

__all__ = [
    "COMPONENT",
    "NAME",
    "TABLE",
    "VIEW",
    "AnswerRow",
    "ChangedInputError",
    "DocumentChangedError",
    "RowContext",
    "answer_rows",
    "mark_stale_for_document",
    "mark_stale_for_requests",
    "record_answers",
    "republish_answers",
    "reusable_answers",
    "write_answers",
]

NAME: Final = "classifications"
TABLE: Final = "classification_rows"
VIEW: Final = "classifications"

_V1 = """
CREATE TABLE classification_rows (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document TEXT NOT NULL,
    doc_id INTEGER REFERENCES documents (doc_id),
    unit TEXT NOT NULL CHECK (unit IN ('page', 'chunk', 'entry', 'document')),
    locator TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    input_sha256 TEXT,
    content_sha256 TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    classifier TEXT NOT NULL,
    classifier_version INTEGER NOT NULL,
    classifier_sha256 TEXT NOT NULL,
    question TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('predicate', 'choice', 'score')),
    value,
    probability REAL,
    probabilities TEXT,
    refusal TEXT,
    model TEXT NOT NULL,
    dialect TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0 CHECK (stale IN (0, 1)),
    run_id TEXT REFERENCES runs (run_id),
    created_at TEXT NOT NULL,
    reused INTEGER NOT NULL DEFAULT 0 CHECK (reused IN (0, 1)),
    CHECK ((unit = 'entry') = (doc_id IS NOT NULL))
);
CREATE TRIGGER classification_rows_follow_documents BEFORE DELETE ON documents
BEGIN
    DELETE FROM classification_rows
    WHERE document = OLD.name OR doc_id = OLD.doc_id;
END;
CREATE INDEX classification_rows_by_answer ON classification_rows
    (document, unit, locator, classifier, question, stale);
CREATE INDEX classification_rows_by_ordinal ON classification_rows
    (document, unit, ordinal);
CREATE INDEX classification_rows_by_doc ON classification_rows (doc_id);
CREATE INDEX classification_rows_by_request ON classification_rows
    (document, request_sha256);
CREATE VIEW classifications AS
WITH live AS (
    SELECT r.* FROM classification_rows AS r
    WHERE r.stale = 0 AND NOT EXISTS (
        SELECT 1 FROM documents AS d
        WHERE d.name = r.document AND d.input_sha256 IS NOT NULL
        AND d.input_sha256 IS NOT r.input_sha256
    )
),
ranked AS (
    SELECT live.*,
        first_value(classifier_sha256) OVER (
            PARTITION BY document, unit, locator, classifier ORDER BY row_id DESC
        ) AS current_sha256,
        row_number() OVER (
            PARTITION BY document, unit, locator, classifier, question
            ORDER BY row_id DESC
        ) AS newest
    FROM live
)
SELECT
    document, doc_id, unit, locator, ordinal, input_sha256, content_sha256,
    request_sha256, classifier, classifier_version, classifier_sha256, question,
    kind, value, probability, probabilities, refusal, model, dialect,
    fingerprint, run_id, created_at, reused
FROM ranked
WHERE newest = 1 AND classifier_sha256 = current_sha256
ORDER BY document, unit, ordinal, locator, classifier, question;
"""

COMPONENT: Final = Component(NAME, (_V1,))
"""Version 1: the table ``classification_rows``, its delete trigger on
``documents`` and the view ``classifications``."""


class ChangedInputError(Exception):
    """A stored document of this name records another input SHA-256.

    The tool refuses the item as it refuses a changed input elsewhere: the
    answers would describe a file the store does not hold.
    """


class DocumentChangedError(ChangedInputError):
    """The stored document the run saw at planning is gone or replaced.

    A drop (or a drop and a new transcription) happened while the request
    was in flight; its answers are not written.
    """


@dataclass(frozen=True)
class RowContext:
    """Where the answers of one request belong, and under which run.

    One request asks about units of one document. ``doc_id`` is set for
    entries only; ``input_sha256`` is the input as observed by this run.
    ``seen_doc_id`` is the ``doc_id`` of the stored document of this name
    when the run planned, None when there was none; tools pass it to
    :func:`record_answers` and :func:`republish_answers`.
    """

    document: str
    fingerprint: str
    doc_id: int | None = None
    input_sha256: str | None = None
    run_id: str | None = None
    seen_doc_id: int | None = None


@dataclass(frozen=True)
class AnswerRow:
    """One row of ``classification_rows`` (``row_id`` and ``stale`` aside)."""

    document: str
    doc_id: int | None
    unit: str
    locator: str
    ordinal: int
    input_sha256: str | None
    content_sha256: str
    request_sha256: str
    classifier: str
    classifier_version: int
    classifier_sha256: str
    question: str
    kind: str
    value: str | float | None
    probability: float | None
    probabilities: Mapping[str, float] | None
    refusal: str | None
    model: str
    dialect: str
    fingerprint: str
    run_id: str | None
    created_at: str


_COLUMNS: Final = tuple(f.name for f in fields(AnswerRow))
_INSERT: Final = (
    f"INSERT INTO {TABLE} ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join(':' + name for name in _COLUMNS)})"
)
_STAMPED: Final = frozenset({"run_id", "created_at"})
_PLACED: Final = frozenset({"locator", "ordinal"})


def _republished(name: str) -> str:
    if name in _STAMPED:
        return ":" + name
    if name in _PLACED:
        return f"COALESCE(:{name}, {name})"
    return name


_REPUBLISH: Final = (
    f"INSERT INTO {TABLE} ({', '.join(_COLUMNS)}, reused) SELECT "
    + ", ".join(_republished(name) for name in _COLUMNS)
    + f", 1 FROM {TABLE} WHERE row_id IN (SELECT value FROM json_each(:ids)) "
    "ORDER BY row_id"
)


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat(timespec="milliseconds")


def answer_rows(
    result: RequestResult, classifier: Classifier, context: RowContext, now: float
) -> list[AnswerRow]:
    """Return the rows of every answer in *result*, stamped at epoch *now*.

    Raises:
        ValueError: Entry answers without a ``doc_id``, other answers with
            one, or a context naming another document than the request.
    """
    entry = classifier.unit == "entry"
    if entry and context.doc_id is None:
        raise ValueError("answers about entries need the document's doc_id")
    if not entry and context.doc_id is not None:
        raise ValueError(f"answers about a {classifier.unit} carry no doc_id")
    if context.document != result.request.document:
        raise ValueError(
            f"the context names {context.document!r}, the request "
            f"{result.request.document!r}"
        )
    created_at = _iso(now)
    return [
        AnswerRow(
            document=context.document,
            doc_id=context.doc_id,
            unit=classifier.unit,
            locator=unit.unit.locator,
            ordinal=unit.unit.ordinal,
            input_sha256=context.input_sha256,
            content_sha256=unit.unit.content_sha256,
            request_sha256=result.request.sha256,
            classifier=classifier.name,
            classifier_version=classifier.version,
            classifier_sha256=classifier.sha256,
            question=answer.question,
            kind=answer.kind,
            value=answer.value,
            probability=answer.probability,
            probabilities=answer.probabilities,
            refusal=answer.refusal,
            model=result.model,
            dialect=result.dialect,
            fingerprint=context.fingerprint,
            run_id=context.run_id,
            created_at=created_at,
        )
        for unit in result.units
        for answer in unit.answers
    ]


def _params(row: AnswerRow) -> dict[str, Any]:
    params = {name: getattr(row, name) for name in _COLUMNS}
    if row.probabilities is not None:
        params["probabilities"] = json.dumps(
            dict(row.probabilities), sort_keys=True, ensure_ascii=False
        )
    return params


def write_answers(connection: sqlite3.Connection, rows: Sequence[AnswerRow]) -> int:
    """Append *rows* on *connection*; return the number written.

    :func:`record_answers` runs it as one transaction per request; a tool
    may also run it inside a transaction of its own writer.
    """
    connection.executemany(_INSERT, [_params(row) for row in rows])
    return len(rows)


async def record_answers(
    store: Store,
    rows: Sequence[AnswerRow],
    *,
    lease: Lease | None = None,
    seen_doc_id: int | None = None,
) -> int:
    """Append one request's *rows* in one transaction; return the number written.

    Entry rows (a ``doc_id``) are written under *lease*, so a lost lease
    refuses them; page, chunk and document rows are written through
    :meth:`~.store.Store.write_unleased` and need no lease. Before such an
    unleased write, the document's unleased rows of every unit kind with
    another ``input_sha256`` are flagged stale. *seen_doc_id* is the
    ``doc_id`` of the stored document the run saw at planning
    (``RowContext.seen_doc_id``); None checks nothing.

    Raises:
        ValueError: The rows belong to more than one document, or entry rows
            come without the lease on their document.
        DocumentChangedError: Unleased rows of a document that was stored as
            *seen_doc_id* and is now missing or another; nothing is written.
        ChangedInputError: Unleased rows whose stored document records
            another input SHA-256; nothing is written.
        LeaseLostError: The lease expired or passed to another holder.
    """
    if not rows:
        return 0
    if len({(row.document, row.doc_id, row.input_sha256) for row in rows}) > 1:
        raise ValueError("the rows of one request belong to one document")
    first = rows[0]

    def write(connection: sqlite3.Connection) -> int:
        if first.doc_id is None:
            _check_unleased(connection, first.document, first.input_sha256, seen_doc_id)
        return write_answers(connection, rows)

    return await _write_for(store, first.doc_id, lease, write)


def _check_unleased(
    connection: sqlite3.Connection,
    document: str,
    input_sha256: str | None,
    seen_doc_id: int | None,
) -> None:
    """Refuse a dropped or changed document, then stale rows of other inputs."""
    if seen_doc_id is not None:
        row = connection.execute(
            "SELECT doc_id FROM documents WHERE name = ?", (document,)
        ).fetchone()
        if row is None or int(row[0]) != seen_doc_id:
            raise DocumentChangedError(
                f"{document}: the store dropped or replaced this document "
                f"(doc_id {seen_doc_id}) since the run planned it; its answers "
                "are not written"
            )
    _refuse_changed_input(connection, document, input_sha256)
    connection.execute(
        f"UPDATE {TABLE} SET stale = 1 WHERE stale = 0 AND doc_id IS NULL "
        "AND document = ? AND input_sha256 IS NOT ?",
        (document, input_sha256),
    )


async def _write_for[T](
    store: Store,
    doc_id: int | None,
    lease: Lease | None,
    func: Callable[[sqlite3.Connection], T],
) -> T:
    """Run *func* unleased for unstored units, else under the document's lease."""
    if doc_id is None:
        return await store.write_unleased(func)
    if lease is None or lease.doc_id != doc_id:
        raise ValueError(f"rows of document {doc_id} need its lease")
    return await store.write(lease, func)


def _refuse_changed_input(
    connection: sqlite3.Connection, document: str, input_sha256: str | None
) -> None:
    row = connection.execute(
        "SELECT input_sha256 FROM documents WHERE name = ?", (document,)
    ).fetchone()
    if row is not None and row[0] is not None and row[0] != input_sha256:
        raise ChangedInputError(
            f"{document}: the store holds this document with input SHA-256 "
            f"{row[0][:12]}..., the classified file has {str(input_sha256)[:12]}...; "
            "process the changed file first"
        )


async def republish_answers(
    store: Store,
    answers: Mapping[str, Mapping[str, Any]],
    *,
    run_id: str | None,
    lease: Lease | None = None,
    locator: str | None = None,
    ordinal: int | None = None,
    seen_doc_id: int | None = None,
) -> int:
    """Append the reused *answers* again under *run_id*; return the count.

    *answers* is what :func:`reusable_answers` returned for one unit. The
    copies keep every answer column, take *run_id*, the store clock's time,
    ``reused`` 1 and the unit's current *locator* and *ordinal* (None keeps
    the stored one), so the view shows what this run used and where; when
    either changes, the source rows are flagged stale in the same
    transaction. Entry answers need *lease*; the others pass the checks of
    :func:`record_answers`, *seen_doc_id* included.

    Raises:
        ValueError: Entry answers without the lease on their document.
        DocumentChangedError: The document seen at planning is gone or
            replaced.
        ChangedInputError: The stored document records another input.
        LeaseLostError: The lease expired or passed to another holder.
    """
    if not answers:
        return 0
    ids = sorted(int(answer["row_id"]) for answer in answers.values())
    doc_ids = {answer["doc_id"] for answer in answers.values()}
    if len(doc_ids) > 1:
        raise ValueError("reused answers belong to one unit")
    params = {
        "run_id": run_id,
        "created_at": _iso(store.clock()),
        "ids": json.dumps(ids),
        "locator": locator,
        "ordinal": ordinal,
    }

    def write(connection: sqlite3.Connection) -> int:
        found = connection.execute(
            f"SELECT document, input_sha256, locator, ordinal FROM {TABLE} "
            "WHERE row_id = ?",
            (ids[0],),
        ).fetchone()
        if found is None:  # only a drop deletes rows
            raise DocumentChangedError(
                "the reused answers are gone: the store dropped their document"
            )
        document, input_sha256, old_locator, old_ordinal = found
        if not doc_ids - {None}:
            _check_unleased(connection, document, input_sha256, seen_doc_id)
        count = int(connection.execute(_REPUBLISH, params).rowcount)
        moved = (locator is not None and locator != old_locator) or (
            ordinal is not None and ordinal != old_ordinal
        )
        if moved:
            connection.execute(
                f"UPDATE {TABLE} SET stale = 1 "
                "WHERE row_id IN (SELECT value FROM json_each(?))",
                (params["ids"],),
            )
        return count

    return await _write_for(store, next(iter(doc_ids)), lease, write)


def mark_stale_for_requests(
    connection: sqlite3.Connection, document: str, request_sha256s: Sequence[str]
) -> int:
    """Flag every row of *document*'s requests *request_sha256s*; return the count."""
    cursor = connection.execute(
        f"UPDATE {TABLE} SET stale = 1 WHERE stale = 0 AND document = ? "
        "AND request_sha256 IN (SELECT value FROM json_each(?))",
        (document, json.dumps(list(request_sha256s))),
    )
    return int(cursor.rowcount)


def mark_stale_for_document(
    connection: sqlite3.Connection,
    document: str,
    *,
    unit: str | None = None,
    ordinal: int | None = None,
) -> int:
    """Flag the rows of *document* and every row of the requests they share.

    *unit* narrows the rows to one unit kind and *ordinal* (which needs
    *unit*) to one unit, such as a re-extracted chunk. Rows of other
    documents are never flagged. Returns the number of rows flagged.

    Raises:
        ValueError: *ordinal* is given without *unit*.
    """
    if ordinal is not None and unit is None:
        raise ValueError("an ordinal needs its unit kind")
    where = ["document = ?", "stale = 0"]
    params: list[Any] = [document]
    if unit is not None:
        where.append("unit = ?")
        params.append(unit)
    if ordinal is not None:
        where.append("ordinal = ?")
        params.append(ordinal)
    requests = [
        str(row[0])
        for row in connection.execute(
            f"SELECT DISTINCT request_sha256 FROM {TABLE} WHERE {' AND '.join(where)}",
            params,
        )
    ]
    return mark_stale_for_requests(connection, document, requests) if requests else 0


_REUSABLE: Final = f"""
SELECT row_id, doc_id, question, kind, value, probability, probabilities, refusal,
    model, dialect
FROM {TABLE} AS r
WHERE document = :document AND unit = :unit AND locator = :locator
AND classifier = :classifier AND stale = 0 AND fingerprint = :fingerprint
AND input_sha256 IS :input AND content_sha256 = :content
AND request_sha256 = :request
AND row_id = (
    SELECT max(o.row_id) FROM {TABLE} AS o
    WHERE o.document = r.document AND o.unit = r.unit AND o.locator = r.locator
    AND o.classifier = r.classifier AND o.question = r.question AND o.stale = 0
    AND o.fingerprint = r.fingerprint AND o.input_sha256 IS r.input_sha256
    AND o.content_sha256 = r.content_sha256
    AND o.request_sha256 = r.request_sha256
)
ORDER BY question
"""


def reusable_answers(
    connection: sqlite3.Connection,
    context: RowContext,
    *,
    unit: str,
    locator: str,
    classifier: str,
    content_sha256: str,
    request_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Return the stored answers a run may keep for one unit, by question.

    A row counts only while it is not stale and its fingerprint, content
    SHA-256 and request SHA-256 all equal the given ones, its document and
    input SHA-256 those of *context*; of several such rows per question the
    latest counts. ``probabilities`` is decoded; ``row_id`` and ``doc_id``
    are what :func:`republish_answers` needs, which a caller that keeps the
    answers must call.
    """
    params = {
        "document": context.document,
        "unit": unit,
        "locator": locator,
        "classifier": classifier,
        "fingerprint": context.fingerprint,
        "input": context.input_sha256,
        "content": content_sha256,
        "request": request_sha256,
    }
    previous = connection.row_factory
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(_REUSABLE, params).fetchall()
    finally:
        connection.row_factory = previous
    answers: dict[str, dict[str, Any]] = {}
    for row in rows:
        answer = dict(row)
        if answer["probabilities"] is not None:
            answer["probabilities"] = json.loads(answer["probabilities"])
        answers[str(answer.pop("question"))] = answer
    return answers
