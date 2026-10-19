"""Finalize submitted batches; shared by batch check and repair.

:func:`finalize_submission` finalizes one submission of the store: it asks
the provider for the batch status and, once the batch has finished,
downloads the results. Every result record becomes one ``responses`` row
holding the record as downloaded; then every unit of the submission is
written in a transaction that checks its document's lease, together with
its entries: a usable answer as a done chunk whose entries go to the
``entries`` table, a failed request, an answer without usable text and a
unit without a result as a failed chunk. Last, the submission is marked
finalized. A batch that ended without results marks the submission failed,
so its units are pending again. A provider error that outlasts the retries
of :class:`BatchClients` ends in an outcome and changes nothing. Each
response row holds as its request the body kept for the record's custom id
when it was submitted.

:func:`render_documents` then renders the finalized documents
(``_output.json`` and the CSV) under the leases the caller holds, and
:func:`delete_provider_files` deletes the batch's input and result files
once the results are stored and rendered.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronominer.batch.clients import (
    BatchClients,
    MissingKeyError,
    results_of,
    status_of,
)
from chronominer.common.batch import (
    MISSING_RESULT_ERROR,
    BatchHandle,
    BatchResultItem,
    BatchStatus,
    BatchStatusInfo,
    outcome_counts,
    supports_batch,
)
from chronominer.common.retry import ErrorKind, classify_error
from chronominer.common.sqlite import read
from chronominer.common.store import (
    LeaseLostError,
    ResponseRow,
    Store,
    Submission,
    UnitKey,
    UnitResult,
    UnitRow,
)
from chronominer.common.usage import Usage
from chronominer.extract.response_parsing import (
    INVALID_JSON,
    NO_OBJECT,
    answer_outcome,
    parse_answer,
    parse_extraction,
)
from chronominer.extract.units import EMPTY_ANSWER, EmptyAnswerError
from chronominer.output.entries import entries_writer
from chronominer.output.render import JSON_FORMAT, output_formats, render_item
from chronominer.output.store import document_metadata, output_path

__all__ = [
    "FAILED",
    "FINALIZED_OUTCOME",
    "MISSING_TEXT",
    "PENDING",
    "SKIPPED",
    "DocumentSchema",
    "SubmissionFinalization",
    "delete_provider_files",
    "document_schema",
    "finalize_submission",
    "placeholder",
    "render_documents",
    "submission_handle",
]

logger = logging.getLogger(__name__)

FINALIZED_OUTCOME = "finalized"
PENDING = "pending"
FAILED = "failed"
SKIPPED = "skipped"

MISSING_TEXT = "[extraction error: no result returned]"
_RETRYABLE = frozenset(
    {
        ErrorKind.RATE_LIMIT,
        ErrorKind.SERVER_ERROR,
        ErrorKind.TIMEOUT,
        ErrorKind.CONNECTION,
    }
)


def placeholder(provider: str, item: BatchResultItem) -> str:
    """Return the error text of a failed result.

    OpenAI results name the HTTP status of an error in the output file and
    the error code; the other providers give their error message.
    """
    if item.origin == "missing":
        return MISSING_TEXT
    if provider != "openai":
        return f"[extraction error: {item.error or 'unknown error'}]"
    parts = ["[extraction error"]
    status = item.response.get("status_code") if item.origin == "output" else None
    if isinstance(status, int) and status != 200:
        parts.append(f"status {status}")
    if item.error_code:
        parts.append(f"code {item.error_code}")
    if item.error and item.error != f"HTTP {status}":
        parts.append(item.error)
    return "; ".join(parts) + "]"


def _usage_of(items: Sequence[BatchResultItem]) -> Usage:
    total = Usage()
    for item in items:
        total = total + item.usage
    return total


def _error_suffix(info: BatchStatusInfo) -> str:
    return f": {info.error_message}" if info.error_message else ""


@dataclass(frozen=True)
class SubmissionFinalization:
    """The outcome of finalizing one submission of the store.

    *outcome* is ``finalized``, ``pending`` (still running, or a retryable
    provider error), ``failed`` (the batch ended without results, or a
    provider error that is not retryable) or ``skipped`` (the submission is
    not submitted); *detail* says why. *documents* are the ids of the
    documents whose units were written and *errors* the number of units
    written as failed chunks.
    """

    submission_id: str
    outcome: str
    detail: str | None = None
    provider_status: str | None = None
    documents: tuple[int, ...] = ()
    units: int = 0
    errors: int = 0
    usage: Usage = Usage()


@dataclass(frozen=True)
class DocumentSchema:
    """The schema a document was extracted with: its name and JSON, if stored."""

    name: str | None = None
    schema: Mapping[str, Any] | None = None


def _schema_of(connection: sqlite3.Connection, doc_id: int) -> DocumentSchema:
    metadata, sha = document_metadata(connection, doc_id)
    name = metadata.get("schema_name")
    schema: Mapping[str, Any] | None = None
    if sha:
        row = connection.execute(
            "SELECT text FROM artifacts WHERE sha256 = ?", (sha,)
        ).fetchone()
        if row is not None:
            try:
                loaded = json.loads(row[0])
            except (TypeError, json.JSONDecodeError):
                loaded = None
            schema = loaded if isinstance(loaded, dict) else None
    if not name and schema is not None:
        name = schema.get("name")
    return DocumentSchema(str(name) if name else None, schema)


def document_schema(store: Store, doc_id: int) -> DocumentSchema:
    """Return the schema document *doc_id* was extracted with, from the store."""
    return read(store.path, lambda connection: _schema_of(connection, doc_id))


def submission_handle(
    submission: Submission, custom_ids: Iterable[str] = ()
) -> BatchHandle:
    """Return the backend handle of *submission*'s provider batch.

    The upload reference becomes the input file the backend deletes in its
    cleanup.
    """
    metadata: dict[str, Any] = {}
    if submission.upload_ref:
        key = "input_file_name" if submission.provider == "google" else "input_file_id"
        metadata[key] = submission.upload_ref
    return BatchHandle(
        provider=submission.provider,
        batch_id=submission.batch_id or "",
        custom_ids=list(custom_ids),
        metadata=metadata,
    )


def _status_label(info: BatchStatusInfo) -> str:
    return info.provider_status or info.status.value


def _usage_record(usage: Usage) -> dict[str, int]:
    """Return *usage* in the shape every stored usage record has."""
    return usage.as_dict()


def _http_status(item: BatchResultItem) -> int | None:
    status = item.response.get("status_code")
    if isinstance(status, int) and not isinstance(status, bool):
        return status
    return None


def _missing(custom_id: str) -> BatchResultItem:
    return BatchResultItem(
        custom_id=custom_id,
        error=MISSING_RESULT_ERROR,
        error_code="missing",
        origin="missing",
    )


async def _append_records(
    store: Store,
    submission: Submission,
    requests: Mapping[str, UnitKey],
    items: Sequence[BatchResultItem],
    role: str,
) -> tuple[dict[str, BatchResultItem], dict[str, int]]:
    """Store every result record as one response row, in provider order.

    A record without a custom id (a malformed result line) is stored with
    neither a custom id nor a unit and is never attributed to a unit. A
    record's request is the body kept for its custom id at submission.
    Returns the first result of each requested custom id and the response
    id of each stored record.
    """
    results: dict[str, BatchResultItem] = {}
    response_ids: dict[str, int] = {}
    bodies = store.submission_bodies(submission.submission_id)
    for item in items:
        if not item.custom_id:
            if item.raw is not None:
                logger.warning(
                    "Batch %s of submission %s returned a record without a "
                    "custom id; stored without a unit: %s",
                    submission.batch_id,
                    submission.submission_id,
                    item.error or "no custom id",
                )
                await store.append_response(
                    ResponseRow(
                        provider=submission.provider,
                        model=submission.model,
                        role=role,
                        attempt=1,
                        submission_id=submission.submission_id,
                        raw=item.raw,
                        raw_kind=item.raw_kind,
                        error=item.error,
                        error_type=item.error_code,
                    )
                )
            continue
        unit = requests.get(item.custom_id)
        if unit is None:
            logger.warning(
                "Batch %s returned %r, which submission %s did not request",
                submission.batch_id,
                item.custom_id,
                submission.submission_id,
            )
        if item.raw is not None:
            response_ids.setdefault(
                item.custom_id,
                await store.append_response(
                    ResponseRow(
                        provider=submission.provider,
                        model=submission.model,
                        unit=unit,
                        role=role,
                        attempt=1,
                        submission_id=submission.submission_id,
                        custom_id=item.custom_id,
                        request=bodies.get(item.custom_id),
                        raw=item.raw,
                        raw_kind=item.raw_kind,
                        usage=_usage_record(item.usage),
                        error=item.error,
                        error_type=item.error_code,
                        http_status=_http_status(item),
                    )
                ),
            )
        if unit is not None:
            results.setdefault(item.custom_id, item)
    return results, response_ids


@dataclass(frozen=True)
class _Answer:
    """What one result yields: its text, entries and error, if any."""

    content: str | None
    entries: list[dict[str, Any]]
    error: str | None = None
    error_type: str | None = None


def _answer(provider: str, item: BatchResultItem) -> _Answer:
    """Return the chunk content and entries of a result, or why it failed.

    A content filter, a refusal, a truncated answer and an answer without a
    JSON object fail with the outcome's message and kind, as in a
    synchronous run.
    """
    if item.outcome.stops_page:
        return _Answer(None, [], item.outcome.message, item.outcome.kind)
    if not item.success:
        return _Answer(None, [], placeholder(provider, item), None)
    text = item.text or ""
    if not text.strip():
        return _Answer(None, [], EMPTY_ANSWER, EmptyAnswerError.__name__)
    if parse_answer(text) is None:
        return _Answer(None, [], NO_OBJECT, INVALID_JSON)
    return _Answer(text, parse_extraction(text))


def _provenance(row: UnitRow | None, usage: Usage) -> dict[str, Any]:
    """Return the unit's provenance with the response fields of its record.

    The ``_output.json`` record of a batch answer carries the input and
    output tokens under ``response_data.usage``, or no usage when the
    provider reported none.
    """
    stored = row.provenance if row is not None else None
    provenance = dict(stored) if isinstance(stored, Mapping) else {}
    response_data: dict[str, Any] = {}
    if usage.input_tokens or usage.output_tokens:
        response_data["usage"] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
    provenance["response"] = {"response_data": response_data}
    return provenance


async def _write_units(
    store: Store,
    submission: Submission,
    requests: Mapping[str, UnitKey],
    results: Mapping[str, BatchResultItem],
    response_ids: Mapping[str, int],
) -> tuple[int, Usage]:
    """Write every requested unit; return the error count and the usage."""
    doc_ids = sorted({key.doc_id for key in requests.values()})
    held = store.held_leases()
    for doc_id in doc_ids:
        if doc_id not in held:
            raise LeaseLostError(f"Document {doc_id} is not leased by this run.")
    rows: dict[UnitKey, UnitRow] = {
        row.key: row for doc_id in doc_ids for row in store.units(doc_id)
    }
    schemas = {doc_id: document_schema(store, doc_id) for doc_id in doc_ids}
    errors = 0
    usage = Usage()
    for custom_id, key in requests.items():
        item = results.get(custom_id) or _missing(custom_id)
        row = rows.get(key)
        answer = _answer(submission.provider, item)
        failed = answer.content is None
        errors += failed
        usage = usage + item.usage
        schema = schemas[key.doc_id]
        await store.write_unit(
            held[key.doc_id],
            UnitResult(
                kind=key.kind,
                idx=key.idx,
                status="failed" if failed else "done",
                model=submission.model,
                provider=submission.provider,
                content=answer.content,
                outcome="failed" if failed else answer_outcome(answer.content),
                label=row.label if row is not None else None,
                source_index=row.source_index if row is not None else None,
                usage=_usage_record(item.usage),
                error=answer.error,
                error_type=(answer.error_type or item.error_code) if failed else None,
                provenance=_provenance(row, item.usage),
                response_id=response_ids.get(custom_id),
                attempts=1,
            ),
            extra=entries_writer(
                store,
                key.doc_id,
                key.idx,
                answer.entries,
                model=submission.model,
                provider=submission.provider,
                schema=schema.schema,
                schema_name=schema.name,
                kind=key.kind,
            ),
        )
    return errors, usage


def _submission_failure(
    submission_id: str, exc: Exception, what: str
) -> SubmissionFinalization:
    """Return the outcome of a provider call that failed after its retries.

    A missing key and a non-retryable error fail the submission; a
    retryable error leaves it pending for the next check.
    """
    if isinstance(exc, MissingKeyError):
        return SubmissionFinalization(submission_id, FAILED, str(exc))
    outcome = PENDING if classify_error(exc) in _RETRYABLE else FAILED
    logger.warning("%s: %s: %s", what, type(exc).__name__, exc)
    return SubmissionFinalization(submission_id, outcome, f"{what}: {exc}")


async def finalize_submission(
    store: Store, submission: Submission, clients: BatchClients
) -> SubmissionFinalization:
    """Finalize *submission* from its provider batch into the store.

    The store must hold the lease of every document the submission covers.
    Nothing is written until the batch has finished and its results are
    downloaded, apart from the provider status of each check.

    Raises:
        LeaseLostError: A document's lease was lost before its units were
            written; the submission stays submitted for the next check.
    """
    submission_id = submission.submission_id
    if submission.status != "submitted" or not submission.batch_id:
        return SubmissionFinalization(
            submission_id, SKIPPED, f"submission is {submission.status}"
        )
    provider = submission.provider
    if not supports_batch(provider):
        return SubmissionFinalization(
            submission_id, FAILED, f"provider {provider!r} has no batch API"
        )
    requests = store.submission_requests(submission_id)
    handle = submission_handle(submission, requests)
    try:
        info = await clients.call(
            provider,
            status_of(handle),
            model=submission.model,
            label=f"status of batch {handle.batch_id}",
        )
    except Exception as exc:  # noqa: BLE001 - classified by _submission_failure
        return _submission_failure(
            submission_id, exc, f"status of batch {handle.batch_id} not retrieved"
        )
    status = _status_label(info)
    await store.record_check(
        submission_id, provider_status=status, counts=info.counts()
    )
    if not info.is_terminal:
        detail = f"batch {handle.batch_id} {status}{_error_suffix(info)}"
        return SubmissionFinalization(submission_id, PENDING, detail, status)
    if info.status is not BatchStatus.COMPLETED and not info.results_available:
        reason = f"batch {handle.batch_id} {info.status.value}{_error_suffix(info)}"
        await store.mark_failed(submission_id, reason)
        return SubmissionFinalization(submission_id, FAILED, reason, status)
    try:
        items = await clients.call(
            provider,
            results_of(handle),
            usage=_usage_of,
            model=submission.model,
            label=f"results of batch {handle.batch_id}",
        )
    except Exception as exc:  # noqa: BLE001 - classified by _submission_failure
        return _submission_failure(
            submission_id, exc, f"results of batch {handle.batch_id} not downloaded"
        )
    results, response_ids = await _append_records(
        store, submission, requests, items, clients.role
    )
    errors, usage = await _write_units(
        store, submission, requests, results, response_ids
    )
    await store.mark_finalized(
        submission_id,
        provider_status=status,
        counts=outcome_counts(len(requests), errors),
    )
    documents = tuple(sorted({key.doc_id for key in requests.values()}))
    logger.info(
        "Finalized submission %s from batch %s: %d unit(s), %d error(s)",
        submission_id,
        handle.batch_id,
        len(requests),
        errors,
    )
    return SubmissionFinalization(
        submission_id,
        FINALIZED_OUTCOME,
        None,
        status,
        documents,
        len(requests),
        errors,
        usage,
    )


def _output_file(store: Store, root: Path, doc_id: int, name: str) -> Path:
    """Return a document's ``_output.json``: its registered file, else by name."""
    for row in store.outputs(doc_id):
        if row.format != JSON_FORMAT:
            continue
        path = Path(row.path)
        return path if path.is_absolute() else root / path
    return output_path(root, name)


def _formats(store: Store, doc_id: int, formats: Sequence[str]) -> tuple[str, ...]:
    """Return *formats* with the formats the document was rendered in."""
    registered = [row.format for row in store.outputs(doc_id)]
    return output_formats((*formats, *registered))


async def render_documents(
    store: Store, doc_ids: Sequence[int], formats: Sequence[str] = ()
) -> tuple[list[Path], list[str]]:
    """Render the finalized documents under the leases this store holds.

    Each document gets its ``_output.json`` and the CSV when *formats* asks
    for it or the document was rendered as CSV before. Returns the files
    holding the new rendering and the warnings; a document that cannot be
    rendered is reported, its units stay stored.
    """
    root = store.path.parent
    outputs: list[Path] = []
    warnings: list[str] = []
    held = store.held_leases()
    for doc_id in doc_ids:
        document = store.document(doc_id)
        lease = held.get(doc_id)
        if document is None or lease is None:
            warnings.append(f"Document {doc_id} was not rendered: it is not leased")
            continue
        output = _output_file(store, root, doc_id, document.name)
        try:
            report = await render_item(
                store, doc_id, _formats(store, doc_id, formats), output, lease
            )
        except (LeaseLostError, OSError, sqlite3.Error) as exc:
            warnings.append(f"'{document.name}' was not rendered: {exc}")
            continue
        outputs.extend([*report.written, *report.unchanged])
        warnings.extend(report.warnings)
    return outputs, warnings


async def delete_provider_files(clients: BatchClients, submission: Submission) -> None:
    """Delete the input and result files of *submission*'s batch.

    Call it after the results are stored and the documents rendered; the
    store then holds every response. A failure is logged.
    """
    if not submission.batch_id:
        return
    await clients.cleanup(submission_handle(submission), model=submission.model)
