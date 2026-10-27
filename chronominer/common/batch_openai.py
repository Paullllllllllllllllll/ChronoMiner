"""OpenAI Batch API backend for the Responses endpoint.

Request bodies are Responses API bodies; the backend uploads them as one
JSONL file per part. Results come from the output file and the error file:
OpenAI writes successes to the first and failed requests only to the second.
A part created through the part API carries the submission id and the part
number in the batch metadata, which the batch listing returns.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from .batch import (
    TERMINAL_STATUSES,
    BatchBackend,
    BatchHandle,
    BatchListing,
    BatchNotReadyError,
    BatchPart,
    BatchRequest,
    BatchResultItem,
    BatchStatus,
    BatchStatusInfo,
    BatchSummary,
    PartUpload,
    ResultOrigin,
    as_datetime,
    count,
    get_field,
    jsonl_bytes,
    jsonl_lines,
    malformed_line,
    without_retries,
)
from .responses import ResponseOutcome, outcome_of
from .usage import Usage

__all__ = [
    "ENDPOINT",
    "MAX_BATCH_BYTES",
    "MAX_BATCH_REQUESTS",
    "OpenAIBatchBackend",
    "coerce_file_id",
    "parse_result_lines",
    "usage_from_body",
]

logger = logging.getLogger(__name__)

ENDPOINT = "/v1/responses"
MAX_BATCH_REQUESTS = 50_000
MAX_BATCH_BYTES = 150 * 1024 * 1024
MAX_METADATA_VALUE = 512

_ALLOWED_SERVICE_TIERS = frozenset({"auto", "default", "priority"})

_STATUS_MAP = {
    "validating": BatchStatus.PENDING,
    "in_progress": BatchStatus.IN_PROGRESS,
    "finalizing": BatchStatus.IN_PROGRESS,
    "cancelling": BatchStatus.IN_PROGRESS,
    "completed": BatchStatus.COMPLETED,
    "failed": BatchStatus.FAILED,
    "cancelled": BatchStatus.CANCELLED,
    "expired": BatchStatus.EXPIRED,
}

_OUTPUT_FILE_KEYS = (
    "output_file_id",
    "output_file",
    "output_file_ids",
    "response_file_id",
    "result_file_id",
    "results_file_id",
    "result_file_ids",
)
_ERROR_FILE_KEYS = ("error_file_id", "error_file", "errors_file_id", "error_file_ids")


def coerce_file_id(candidate: Any) -> str | None:
    """Return a file id from a string, an ``id`` mapping or a one-item list."""
    if isinstance(candidate, str):
        return candidate or None
    if isinstance(candidate, dict):
        file_id = candidate.get("id") or candidate.get("file_id")
        return file_id if isinstance(file_id, str) and file_id else None
    if isinstance(candidate, list) and candidate:
        return coerce_file_id(candidate[0])
    return None


def _resolve_file_id(batch: Any, keys: Sequence[str]) -> str | None:
    for key in keys:
        file_id = coerce_file_id(get_field(batch, key))
        if file_id:
            return file_id
    return None


def _batch_errors(batch: Any) -> str | None:
    """Join the batch-level ``errors.data`` entries into one message."""
    data = get_field(get_field(batch, "errors"), "data")
    if not isinstance(data, list):
        return None
    parts = []
    for err in data:
        piece = str(get_field(err, "message") or "unknown error")
        code = get_field(err, "code")
        line = get_field(err, "line")
        if code:
            piece = f"{piece} (code={code})"
        if line is not None:
            piece = f"{piece} [line={line}]"
        parts.append(piece)
    return "; ".join(parts) or None


def usage_from_body(body: Any) -> Usage:
    """Return the usage of a Responses API body (zero when absent)."""
    raw = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(raw, dict):
        return Usage()
    input_tokens = count(raw.get("input_tokens") or raw.get("prompt_tokens"))
    output_tokens = count(raw.get("output_tokens") or raw.get("completion_tokens"))
    input_details = raw.get("input_tokens_details") or raw.get("prompt_tokens_details")
    output_details = raw.get("output_tokens_details") or raw.get(
        "completion_tokens_details"
    )
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=count(raw.get("total_tokens")) or input_tokens + output_tokens,
        cached_tokens=count(get_field(input_details, "cached_tokens")),
        reasoning_tokens=count(get_field(output_details, "reasoning_tokens")),
    )


def _error_parts(error: Any) -> tuple[str | None, str | None]:
    """Return ``(message, code)`` of an error object or string."""
    if isinstance(error, dict):
        message = error.get("message") or error.get("error")
        code = error.get("code") or error.get("type")
        return (
            str(message) if message else None,
            str(code) if code else None,
        )
    return (str(error) if error else None), None


def _message_parts(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the content parts of every message item, in order."""
    parts: list[dict[str, Any]] = []
    output = body.get("output")
    for item in output if isinstance(output, list) else []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        for part in content if isinstance(content, list) else []:
            if isinstance(part, dict):
                parts.append(part)
    return parts


def _output_text(body: dict[str, Any]) -> str:
    """Concatenate every ``output_text`` part of every message item."""
    return "".join(
        str(part.get("text") or "")
        for part in _message_parts(body)
        if part.get("type") == "output_text"
    )


def _outcome(body: dict[str, Any]) -> ResponseOutcome:
    """Return how a Responses API body ended: status and refusal parts."""
    refusal = None
    for part in _message_parts(body):
        if part.get("type") == "refusal":
            text = part.get("refusal")
            refusal = text.strip() if isinstance(text, str) else ""
            break
    meta = {key: body.get(key) for key in ("status", "incomplete_details")}
    return outcome_of(meta, refusal)


def _parse_record(record: dict[str, Any], origin: ResultOrigin) -> BatchResultItem:
    item = BatchResultItem(custom_id=str(record.get("custom_id") or ""), origin=origin)
    resp = record.get("response")
    body = resp.get("body") if isinstance(resp, dict) else None
    if isinstance(resp, dict):
        item.response = resp
    if isinstance(body, dict):
        item.usage = usage_from_body(body)

    top_error = record.get("error")
    if top_error and not isinstance(body, dict):
        message, code = _error_parts(top_error)
        item.error = message or "Unknown error"
        item.error_code = code
        if not item.response:
            item.response = {"error": top_error}
        return item

    status_code = resp.get("status_code") if isinstance(resp, dict) else None
    body_error = body.get("error") if isinstance(body, dict) else None
    if (isinstance(status_code, int) and status_code != 200) or body_error:
        message, code = _error_parts(body_error)
        item.error = message or f"HTTP {status_code}"
        item.error_code = code
        return item

    if not isinstance(body, dict):
        body = {}
    item.text = _output_text(body)
    item.settle(
        _outcome(body),
        "No text content in OpenAI batch response "
        f"(status={body.get('status') or 'unknown'})",
    )
    return item


def parse_result_lines(text: str, origin: ResultOrigin) -> list[BatchResultItem]:
    """Parse an output or error file.

    Each item keeps its line as downloaded in ``raw``; a line that is not a
    JSON object becomes an item without a custom id (see
    :func:`~.batch.malformed_line`).
    """
    items: list[BatchResultItem] = []
    source = "the batch error file" if origin == "error_file" else "the batch output"
    for number, line in jsonl_lines(text):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            record = None
        if not isinstance(record, dict):
            items.append(malformed_line(number, line, source))
            continue
        item = _parse_record(record, origin)
        item.raw, item.raw_kind = line, "line"
        items.append(item)
    return items


def _metadata(batch: Any) -> dict[str, str]:
    raw = get_field(batch, "metadata")
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if value is not None}


def _read_text(content: Any) -> str:
    data = content.read() if hasattr(content, "read") else content
    if isinstance(data, bytes | bytearray):
        return bytes(data).decode("utf-8")
    return str(data)


class OpenAIBatchBackend(BatchBackend):
    """OpenAI Batch API over ``/v1/responses``."""

    provider_name = "openai"
    default_max_requests = MAX_BATCH_REQUESTS
    default_max_bytes = MAX_BATCH_BYTES
    lists_newest_first = True

    def __init__(
        self,
        client: Any,
        *,
        max_requests: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        super().__init__(client, max_requests=max_requests, max_bytes=max_bytes)
        self._flex_logged = False

    def _wrap(self, request: BatchRequest, body: dict[str, Any]) -> dict[str, Any]:
        body = dict(body)
        tier = body.get("service_tier")
        if tier == "flex":
            if not self._flex_logged:
                logger.info("The Batch API rejects service_tier 'flex'; using 'auto'")
                self._flex_logged = True
            body["service_tier"] = "auto"
        elif tier is not None and tier not in _ALLOWED_SERVICE_TIERS:
            body.pop("service_tier")
        return {
            "custom_id": request.custom_id,
            "method": "POST",
            "url": ENDPOINT,
            "body": body,
        }

    def _submit_part(self, entries: list[dict[str, Any]], *, label: str) -> BatchHandle:
        uploaded = self.client.files.create(
            file=(f"{label}.jsonl", jsonl_bytes(entries)), purpose="batch"
        )
        try:
            batch = self.client.batches.create(
                input_file_id=uploaded.id,
                endpoint=ENDPOINT,
                completion_window="24h",
                metadata={"description": label},
            )
        except Exception:
            self._delete_files(uploaded.id)
            raise
        return BatchHandle(
            provider=self.provider_name,
            batch_id=str(batch.id),
            custom_ids=[str(entry["custom_id"]) for entry in entries],
            metadata={"input_file_id": uploaded.id, "request_count": len(entries)},
        )

    def _upload_part(
        self, part: BatchPart, *, submission_id: str, label: str
    ) -> PartUpload:
        uploaded = self.client.files.create(
            file=(
                f"{label}-{submission_id}-p{part.index}.jsonl",
                jsonl_bytes(part.entries),
            ),
            purpose="batch",
        )
        return PartUpload(self.provider_name, str(uploaded.id))

    def _create_part(
        self,
        part: BatchPart,
        upload: PartUpload,
        *,
        submission_id: str,
        label: str,
    ) -> BatchHandle:
        if not upload.file_id:
            raise ValueError("an OpenAI batch part needs an uploaded input file")
        input_file_id = upload.file_id
        metadata = {
            "description": label[:MAX_METADATA_VALUE],
            "submission_id": submission_id,
            "part": str(part.index),
        }
        batch = self._guarded_create(
            part,
            upload,
            submission_id,
            lambda: without_retries(self.client).batches.create(
                input_file_id=input_file_id,
                endpoint=ENDPOINT,
                completion_window="24h",
                metadata=metadata,
            ),
        )
        return BatchHandle(
            provider=self.provider_name,
            batch_id=str(batch.id),
            custom_ids=part.custom_ids,
            metadata={
                "input_file_id": input_file_id,
                "request_count": len(part.entries),
            },
        )

    def _delete_upload(self, file_id: str) -> None:
        self.client.files.delete(file_id)

    def _list_page(
        self, page_size: int, cursor: str | None
    ) -> tuple[list[BatchListing], str | None]:
        kwargs: dict[str, Any] = {"limit": page_size}
        if cursor:
            kwargs["after"] = cursor
        page = self.client.batches.list(**kwargs)
        data = get_field(page, "data")
        batches = list(page if data is None else data)
        listings = [self._listing(batch) for batch in batches]
        has_more = get_field(page, "has_more")
        more = has_more if isinstance(has_more, bool) else bool(batches)
        if data is None or not more or not batches:
            return listings, None
        return listings, str(get_field(batches[-1], "id") or "") or None

    def _listing(self, batch: Any) -> BatchListing:
        provider_status = str(get_field(batch, "status") or "")
        counts = get_field(batch, "request_counts")
        total = get_field(counts, "total")
        return BatchListing(
            provider=self.provider_name,
            batch_id=str(get_field(batch, "id") or ""),
            status=_STATUS_MAP.get(provider_status, BatchStatus.UNKNOWN),
            provider_status=provider_status,
            created_at=as_datetime(get_field(batch, "created_at")),
            request_count=None if total is None else count(total),
            metadata=_metadata(batch),
        )

    def status(self, handle: BatchHandle) -> BatchStatusInfo:
        """Return the batch status from its status field and request counts."""
        batch = self.client.batches.retrieve(handle.batch_id)
        provider_status = str(get_field(batch, "status") or "")
        status = _STATUS_MAP.get(provider_status, BatchStatus.UNKNOWN)
        counts = get_field(batch, "request_counts")
        total = count(get_field(counts, "total"))
        completed = count(get_field(counts, "completed"))
        failed = count(get_field(counts, "failed"))
        output_file_id = _resolve_file_id(batch, _OUTPUT_FILE_KEYS)
        error_file_id = _resolve_file_id(batch, _ERROR_FILE_KEYS)
        error_message = None
        if status in (BatchStatus.FAILED, BatchStatus.EXPIRED):
            error_message = _batch_errors(batch)
        elif status is BatchStatus.UNKNOWN:
            error_message = f"Unmapped OpenAI batch status: {provider_status!r}"
        has_files = bool(output_file_id or error_file_id)
        return BatchStatusInfo(
            status=status,
            provider_status=provider_status,
            total_requests=total,
            completed_requests=completed,
            failed_requests=failed,
            pending_requests=max(0, total - completed - failed),
            error_message=error_message,
            results_available=has_files and status in TERMINAL_STATUSES,
            output_file_id=output_file_id,
        )

    def _fetch_results(
        self, handle: BatchHandle
    ) -> tuple[list[BatchResultItem], str | None]:
        batch = self.client.batches.retrieve(handle.batch_id)
        provider_status = str(get_field(batch, "status") or "")
        status = _STATUS_MAP.get(provider_status, BatchStatus.UNKNOWN)
        if status not in TERMINAL_STATUSES:
            raise BatchNotReadyError(
                f"OpenAI batch {handle.batch_id} is {provider_status or 'unknown'}"
            )
        items: list[BatchResultItem] = []
        output_file_id = _resolve_file_id(batch, _OUTPUT_FILE_KEYS)
        if output_file_id:
            text = _read_text(self.client.files.content(output_file_id))
            items.extend(parse_result_lines(text, "output"))
        error_file_id = _resolve_file_id(batch, _ERROR_FILE_KEYS)
        if error_file_id:
            text = _read_text(self.client.files.content(error_file_id))
            items.extend(parse_result_lines(text, "error_file"))
        batch_error = _batch_errors(batch)
        if not output_file_id and not error_file_id:
            batch_error = batch_error or f"batch {provider_status} without result files"
        return items, batch_error

    def _cancel(self, handle: BatchHandle) -> None:
        self.client.batches.cancel(handle.batch_id)

    def cleanup(self, handle: BatchHandle) -> None:
        """Delete the input, output and error files of the batch; never raises."""
        output_file_id = error_file_id = None
        try:
            batch = self.client.batches.retrieve(handle.batch_id)
        except Exception as exc:  # noqa: BLE001 - cleanup is best effort
            logger.warning("Could not look up batch %s: %s", handle.batch_id, exc)
        else:
            output_file_id = _resolve_file_id(batch, _OUTPUT_FILE_KEYS)
            error_file_id = _resolve_file_id(batch, _ERROR_FILE_KEYS)
        self._delete_files(
            coerce_file_id(handle.metadata.get("input_file_id")),
            output_file_id,
            error_file_id,
        )

    def _delete_files(self, *file_ids: str | None) -> None:
        for file_id in file_ids:
            if not file_id:
                continue
            try:
                self.client.files.delete(file_id)
            except Exception as exc:  # noqa: BLE001 - cleanup is best effort
                logger.warning("Failed to delete OpenAI file %s: %s", file_id, exc)

    def list_active(self, *, page_size: int = 100) -> list[BatchSummary]:
        """Return the batches that have not reached a terminal state."""
        active: list[BatchSummary] = []
        for batch in self.client.batches.list(limit=page_size):
            provider_status = str(get_field(batch, "status") or "")
            status = _STATUS_MAP.get(provider_status, BatchStatus.UNKNOWN)
            if status in TERMINAL_STATUSES:
                continue
            active.append(
                BatchSummary(
                    provider=self.provider_name,
                    batch_id=str(get_field(batch, "id") or ""),
                    status=status,
                    provider_status=provider_status,
                )
            )
        return active
