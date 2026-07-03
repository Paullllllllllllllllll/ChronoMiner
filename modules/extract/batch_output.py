# modules/extract/batch_output.py

"""Assemble batch finalization results in the unified (sync) output shape.

Since v1.20.0 batch finalization writes the same ``{stem}_output.json``
shape as synchronous extraction: ``{"_chronominer_metadata": {...},
"records": [...]}`` with per-record ``custom_id``, ``chunk_index``,
``chunk_range``, and a nested ``response`` body (``output_text`` plus
``response_data``). Batch provenance (batch ids, parts, provider,
completion counts) is folded into ``_chronominer_metadata`` under
``batch_tracking``. The legacy ``{stem}_final_output.json`` shape
(``responses`` + ``tracking``) is no longer written; files already on
disk remain readable through the legacy fallbacks in
``modules.conversion.json_utils`` and ``modules.extract.resume``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from modules.batch.ops import _extract_chunk_index, _order_responses, _response_to_text
from modules.conversion.json_utils import lean_response
from modules.extract.resume import build_extraction_metadata
from modules.infra.logger import setup_logger

logger = setup_logger(__name__)

_UNKNOWN_INDEX = 10**9  # sentinel returned by _extract_chunk_index


def _resolve_chunk_index(custom_id: Any, meta: dict[str, Any]) -> int | None:
    """Resolve the absolute 1-based chunk index for a response entry."""
    candidate = meta.get("chunk_index")
    if isinstance(candidate, int) and not isinstance(candidate, bool):
        return candidate
    extracted = _extract_chunk_index(custom_id)
    return extracted if extracted < _UNKNOWN_INDEX else None


def _to_unified_record(
    entry: dict[str, Any],
    custom_id_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Convert a batch response entry into a sync-shape output record.

    The record's ``response`` mirrors the normalized body written by the
    synchronous path: ``output_text`` carries the model's text (the JSON
    payload under structured outputs) and ``response_data`` carries
    ancillary data such as token usage when recoverable from the raw
    provider body.
    """
    custom_id = entry.get("custom_id")
    meta = custom_id_map.get(str(custom_id)) or {}
    raw = entry.get("raw_response")

    text = entry.get("response")
    if not isinstance(text, str):
        text = _response_to_text(raw)
    if not text and entry.get("parsed_output") is not None:
        text = json.dumps(entry["parsed_output"], ensure_ascii=False)

    response_data: dict[str, Any] = {}
    if isinstance(raw, dict) and isinstance(raw.get("usage"), dict):
        response_data["usage"] = raw["usage"]

    chunk_range = entry.get("chunk_range")
    if chunk_range is None:
        chunk_range = meta.get("chunk_range")

    return {
        "custom_id": custom_id,
        "chunk_index": _resolve_chunk_index(custom_id, meta),
        "chunk_range": chunk_range,
        "response": lean_response(
            {"output_text": text, "response_data": response_data}
        ),
    }


def build_unified_batch_output(
    responses: list[Any],
    tracking: list[dict[str, Any]],
    *,
    schema_name: str,
    order_map: dict[str, int] | None = None,
    custom_id_map: dict[str, dict[str, Any]] | None = None,
    fully_completed: bool = True,
    completed_batches: int = 0,
    failed_batches: int = 0,
    missing_batches: list[str] | None = None,
    recovered_batch_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build the unified batch output dict (sync shape).

    ``responses`` are the aggregated batch response entries (temp-file
    records plus downloaded results); entries that carry an ``error`` and
    no payload are folded into ``failed_chunks`` metadata instead of
    being written as records, mirroring the synchronous path where failed
    chunks write no record.
    """
    custom_id_map = custom_id_map or {}
    ordered = _order_responses(list(responses), order_map)

    records: list[dict[str, Any]] = []
    failed_chunks: list[int] = []
    for entry in ordered:
        if not isinstance(entry, dict):
            continue
        if entry.get("error") is not None and not entry.get("response"):
            idx = _resolve_chunk_index(
                entry.get("custom_id"),
                custom_id_map.get(str(entry.get("custom_id"))) or {},
            )
            if idx is not None:
                failed_chunks.append(idx)
            logger.warning(
                "Batch request %s failed and is excluded from records: %s",
                entry.get("custom_id"),
                entry.get("error"),
            )
            continue
        records.append(_to_unified_record(entry, custom_id_map))

    total_chunks = _infer_total_chunks(custom_id_map, records, failed_chunks)
    partial = (
        (not fully_completed) or bool(failed_chunks) or (len(records) < total_chunks)
    )

    metadata = build_extraction_metadata(
        schema_name=schema_name,
        model_name=_infer_model_name(tracking),
        chunking_method="unknown",
        total_chunks=total_chunks,
        partial=partial,
        failed_chunks=sorted(set(failed_chunks)) or None,
    )
    metadata["batch_tracking"] = {
        "provider": _infer_provider(tracking),
        "batch_ids": sorted(
            {str(t.get("batch_id")) for t in tracking if t.get("batch_id")}
        ),
        "parts": len(tracking),
        "fully_completed": fully_completed,
        "processed_at": datetime.now(UTC).isoformat(),
        "completed_batches": completed_batches,
        "failed_batches": failed_batches,
        "ordered_by_custom_id": True,
        "missing_batches": list(missing_batches or []),
        "recovered_batch_ids": sorted(recovered_batch_ids or []),
        "tracking": tracking,
    }

    return {"_chronominer_metadata": metadata, "records": records}


def _infer_total_chunks(
    custom_id_map: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    failed_chunks: list[int],
) -> int:
    """Best-effort expected chunk count for the metadata stamp."""
    for meta in custom_id_map.values():
        candidate = meta.get("total_chunks")
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    if custom_id_map:
        return len(custom_id_map)
    return len(records) + len(set(failed_chunks))


def _infer_provider(tracking: list[dict[str, Any]]) -> str:
    providers = {t.get("provider") for t in tracking if t.get("provider")}
    if len(providers) == 1:
        return str(providers.pop())
    if providers:
        return ",".join(sorted(str(p) for p in providers))
    return "openai"  # historic default for tracking records without provider


def _infer_model_name(tracking: list[dict[str, Any]]) -> str:
    for track in tracking:
        metadata = track.get("metadata")
        if isinstance(metadata, dict):
            model = metadata.get("model") or metadata.get("model_name")
            if isinstance(model, str) and model:
                return model
    return "unknown"
