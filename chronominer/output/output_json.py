"""The ``{stem}_output.json`` view of a document's chunk units.

The shape is the one its readers know::

    {"_chronominer_metadata": {...}, "records": [...]}

Each done, non-stale chunk unit is one record, in chunk order:
``custom_id``, ``chunk_index`` (1-based), ``chunk_range`` (``[start, end]``
of a text chunk, else null), ``response`` (``output_text`` with the unit's
content as received, then the other response fields; see
:mod:`chronominer.output.store` for the unit fields) and, for an image or
PDF page, ``image_provenance``.

``_chronominer_metadata`` holds ``schema_name``, ``model_name``,
``chunking_method``, ``total_chunks`` (the document's chunk units),
``timestamp`` (the last unit update), ``version`` 1, then when known
``chunking_text_version``, ``document_total_chunks``, ``chunk_slice``,
``partial`` (true while a unit is not done), ``failed_chunks`` (the chunk
indices of failed units), ``image_provenance`` and ``batch_tracking`` (the
document's batch submissions). The run metadata of ``document_meta`` fills
the named keys; a stored ``timestamp`` or ``batch_tracking`` wins over the
computed one, and other stored keys follow at the end. The file is indented
by two spaces and ends with one newline.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Final

from chronominer.output.store import KIND_CHUNK, document_metadata

__all__ = ["METADATA_KEY", "PAGE_KINDS", "default_custom_id", "render_output_json"]

METADATA_KEY: Final = "_chronominer_metadata"
PAGE_KINDS: Final = ("pdf", "image", "images")
"""Document kinds whose chunks are pages (``-page-`` custom ids)."""

_COMPUTED: Final = ("total_chunks", "partial", "failed_chunks", "version")
_LEADING: Final = ("schema_name", "model_name", "chunking_method")
_OPTIONAL: Final = ("chunking_text_version", "document_total_chunks", "chunk_slice")
_RANGE_RE: Final = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_UNITS: Final = (
    "SELECT idx, status, source_index, content, model, provider, usage, "
    "provenance, stale, updated_at FROM units WHERE doc_id = ? AND kind = ? "
    "ORDER BY idx"
)
_SUBMISSIONS: Final = """
SELECT s.submission_id, s.provider, s.model, s.part, s.intent_at, s.batch_id,
       s.status, s.request_count, s.finalized_at
FROM submissions AS s
WHERE s.submission_id IN (
    SELECT submission_id FROM submission_requests WHERE doc_id = ?
)
ORDER BY s.intent_at, s.part, s.submission_id
"""


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def default_custom_id(name: str, kind: str | None, chunk_index: int) -> str:
    """Return the custom id of a chunk that stored none."""
    stem = name.rsplit("/", 1)[-1]
    unit = "page" if kind in PAGE_KINDS else "chunk"
    return f"{stem}-{unit}-{chunk_index}"


def _chunk_range(provenance: dict[str, Any], source_index: Any) -> Any:
    if "chunk_range" in provenance:
        return provenance["chunk_range"]
    if isinstance(source_index, str):
        match = _RANGE_RE.match(source_index)
        if match:
            return [int(match.group(1)), int(match.group(2))]
    return None


def _response(
    content: str, provenance: dict[str, Any], usage: Any, model: str, provider: str
) -> dict[str, Any]:
    stored = provenance.get("response")
    if isinstance(stored, dict):
        return {"output_text": content, **stored}
    response_data: dict[str, Any] = {}
    if usage:
        response_data["usage"] = usage
    response_data["model"] = model
    response_data["provider"] = provider
    return {
        "output_text": content,
        "response_data": response_data,
        "usage": usage or {},
    }


def _batch_tracking(
    connection: sqlite3.Connection, doc_id: int, schema_name: str
) -> dict[str, Any] | None:
    rows = connection.execute(_SUBMISSIONS, (doc_id,)).fetchall()
    if not rows:
        return None
    providers = sorted({str(row[1]) for row in rows})
    finalized = [row[8] for row in rows if row[6] == "finalized" and row[8]]
    tracking = [
        {
            "batch_id": batch_id,
            "metadata": {
                "model": model,
                "request_count": count,
                "submission_id": submission_id,
            },
            "provider": provider,
            "request_count": count,
            "schema_name": schema_name,
            "status": status,
            "timestamp": intent_at,
        }
        for (
            submission_id,
            provider,
            model,
            _part,
            intent_at,
            batch_id,
            status,
            count,
            _finalized_at,
        ) in rows
    ]
    return {
        "provider": ",".join(providers),
        "batch_ids": sorted({str(row[5]) for row in rows if row[5]}),
        "parts": len(rows),
        "fully_completed": all(row[6] == "finalized" for row in rows),
        "processed_at": max(finalized) if finalized else None,
        "completed_batches": sum(1 for row in rows if row[6] == "finalized"),
        "failed_batches": sum(1 for row in rows if row[6] == "failed"),
        "ordered_by_custom_id": True,
        "missing_batches": sorted(
            str(row[5] or row[0]) for row in rows if row[6] == "unresolved"
        ),
        "recovered_batch_ids": [],
        "tracking": tracking,
    }


def _document(connection: sqlite3.Connection, doc_id: int) -> tuple[str, str | None]:
    row = connection.execute(
        "SELECT name, kind FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"No document {doc_id}")
    return str(row[0]), row[1]


def _schema_name(connection: sqlite3.Connection, doc_id: int) -> str | None:
    row = connection.execute(
        "SELECT schema_name FROM entry_rows WHERE doc_id = ? "
        "AND schema_name IS NOT NULL LIMIT 1",
        (doc_id,),
    ).fetchone()
    return None if row is None else str(row[0])


def _add_known_keys(
    meta: dict[str, Any],
    stored: dict[str, Any],
    *,
    complete: bool,
    failed: list[int],
) -> None:
    """Add the optional keys, the partial flag and failed chunks in file order."""
    for key in _OPTIONAL:
        if stored.get(key) is not None:
            meta[key] = stored[key]
    if not complete:
        meta["partial"] = True
    if failed:
        meta["failed_chunks"] = failed
    if stored.get("image_provenance"):
        meta["image_provenance"] = stored["image_provenance"]


def render_output_json(connection: sqlite3.Connection, doc_id: int) -> str:
    """Return the ``_output.json`` view of document *doc_id*."""
    name, kind = _document(connection, doc_id)
    stored, _ = document_metadata(connection, doc_id)
    records: list[dict[str, Any]] = []
    failed: list[int] = []
    total = 0
    complete = True
    latest_model: str | None = None
    timestamp: str | None = None
    for row in connection.execute(_UNITS, (doc_id, KIND_CHUNK)):
        idx, status, source_index, content, model, provider = row[:6]
        usage, provenance = _json(row[6]), _json(row[7]) or {}
        stale, updated_at = row[8], row[9]
        total += 1
        if updated_at and (timestamp is None or updated_at > timestamp):
            timestamp = str(updated_at)
        if status == "failed":
            failed.append(int(idx) + 1)
        if status != "done" or stale or content is None:
            complete = False
            continue
        latest_model = str(model)
        chunk_index = int(idx) + 1
        record: dict[str, Any] = {
            "custom_id": provenance.get("custom_id")
            or default_custom_id(name, kind, chunk_index),
            "chunk_index": chunk_index,
            "chunk_range": _chunk_range(provenance, source_index),
            "response": _response(
                str(content), provenance, usage, str(model), str(provider)
            ),
        }
        if provenance.get("image_provenance"):
            record["image_provenance"] = provenance["image_provenance"]
        records.append(record)

    schema_name = (
        stored.get("schema_name") or _schema_name(connection, doc_id) or "unknown"
    )
    meta: dict[str, Any] = {
        "schema_name": schema_name,
        "model_name": stored.get("model_name") or latest_model or "unknown",
        "chunking_method": stored.get("chunking_method") or "unknown",
        "total_chunks": total,
        "timestamp": stored.get("timestamp") or timestamp,
        "version": 1,
    }
    _add_known_keys(meta, stored, complete=complete, failed=failed)
    tracking = stored.get("batch_tracking") or _batch_tracking(
        connection, doc_id, str(schema_name)
    )
    if tracking:
        meta["batch_tracking"] = tracking
    known = {*_COMPUTED, *_LEADING, *_OPTIONAL, "timestamp", "image_provenance"}
    for key, value in stored.items():
        if key not in known and key != "batch_tracking":
            meta[key] = value
    document = {METADATA_KEY: meta, "records": records}
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"
