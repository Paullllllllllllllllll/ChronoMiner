"""Anthropic Message Batches backend.

Request bodies are Messages API parameters; the backend sends them inline in
one create call per part and streams the results once the batch has ended.
Batches carry no metadata, so the batch listing reports only their creation
time, request count and status.

When a capture scope (:mod:`.capture`) saw the results download, the backend
parses that body itself: each item keeps its line as downloaded, results pair
with requests by the custom id in each line, and a line that is not a JSON
object becomes an item without a custom id. Without a captured body (a client
without the capture hooks) the SDK decoder parses the results and each item
keeps the JSON dump of the SDK record.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .batch import (
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
    as_datetime,
    as_plain_dict,
    count,
    get_field,
    jsonl_lines,
    malformed_line,
    without_retries,
)
from .capture import CapturedResponse, current_scope, sdk_dump
from .responses import outcome_of
from .usage import Usage

__all__ = [
    "AnthropicBatchBackend",
    "MAX_BATCH_BYTES",
    "MAX_BATCH_REQUESTS",
    "parse_lines",
    "parse_result",
    "usage_from_message",
]

logger = logging.getLogger(__name__)

MAX_BATCH_REQUESTS = 100_000
MAX_BATCH_BYTES = 224 * 1024 * 1024

_IN_PROGRESS_STATES = frozenset({"in_progress", "canceling"})


def usage_from_message(message: Any) -> Usage:
    """Return a message's usage with cache reads and writes added to the input."""
    raw = get_field(message, "usage")
    if raw is None:
        return Usage()
    input_tokens = count(get_field(raw, "input_tokens"))
    output_tokens = count(get_field(raw, "output_tokens"))
    cache_read = count(get_field(raw, "cache_read_input_tokens"))
    cache_write = count(get_field(raw, "cache_creation_input_tokens"))
    total_input = input_tokens + cache_read + cache_write
    return Usage(
        input_tokens=total_input,
        output_tokens=output_tokens,
        total_tokens=total_input + output_tokens,
        cached_tokens=cache_read,
    )


def _map_status(batch: Any) -> tuple[BatchStatus, str | None, dict[str, int]]:
    """Return the status, an error message and the request counts of a batch."""
    processing_status = str(get_field(batch, "processing_status") or "")
    raw_counts = get_field(batch, "request_counts")
    counts = {
        key: count(get_field(raw_counts, key))
        for key in ("processing", "succeeded", "errored", "canceled", "expired")
    }
    total = sum(counts.values())
    if processing_status in _IN_PROGRESS_STATES:
        return BatchStatus.IN_PROGRESS, None, counts
    if processing_status != "ended":
        return BatchStatus.PENDING, None, counts
    if total == 0:
        return (
            BatchStatus.UNKNOWN,
            "Anthropic batch ended but reported no request counts",
            counts,
        )
    if counts["errored"] == total:
        return BatchStatus.FAILED, None, counts
    if counts["canceled"] == total:
        return BatchStatus.CANCELLED, None, counts
    if counts["expired"] == total:
        return BatchStatus.EXPIRED, None, counts
    return BatchStatus.COMPLETED, None, counts


def _succeeded(item: BatchResultItem, message: Any) -> None:
    item.response = {"message": as_plain_dict(message)}
    item.usage = usage_from_message(message)
    blocks = get_field(message, "content") or []
    item.text = "".join(
        str(get_field(block, "text") or "")
        for block in blocks
        if get_field(block, "type") == "text"
    )
    stop_reason = get_field(message, "stop_reason")
    item.settle(
        outcome_of({"stop_reason": stop_reason}),
        "No text content in Anthropic batch response "
        f"(stop_reason={stop_reason or 'unknown'})",
    )


def parse_result(result: Any, line: str | None = None) -> BatchResultItem:
    """Return the item of one ``MessageBatchIndividualResponse``.

    *result* is the SDK record or the parsed result line; *line*, the line
    as downloaded, becomes ``raw``, else the JSON dump of *result*.
    """
    item = BatchResultItem(
        custom_id=str(get_field(result, "custom_id") or ""),
        raw=sdk_dump(result) if line is None else line,
        raw_kind="sdk" if line is None else "line",
    )
    data = get_field(result, "result")
    result_type = get_field(data, "type")
    if result_type == "succeeded":
        message = get_field(data, "message")
        if message is None:
            item.error = "Anthropic batch result succeeded but carried no message"
        else:
            _succeeded(item, message)
    elif result_type == "errored":
        error = get_field(data, "error")
        inner = get_field(error, "error") or error
        item.error = str(get_field(inner, "message") or "Unknown error")
        code = get_field(inner, "type")
        item.error_code = str(code) if code else None
        if error is not None:
            item.response = {"error": as_plain_dict(error)}
    elif result_type in ("canceled", "expired"):
        item.error = f"Request {result_type}"
        item.error_code = str(result_type)
    else:
        item.error = f"Unknown result type: {result_type}"
    return item


def parse_lines(text: str) -> list[BatchResultItem]:
    """Parse a results JSONL body; each item keeps its line as ``raw``.

    A line that is not a JSON object becomes an item without a custom id.
    """
    items: list[BatchResultItem] = []
    for number, line in jsonl_lines(text):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            record = None
        if not isinstance(record, dict):
            items.append(malformed_line(number, line, "the Anthropic batch results"))
            continue
        items.append(parse_result(record, line))
    return items


def _captured_results(decoder: Any, captured: list[CapturedResponse]) -> str | None:
    """Return the captured body of the results download, if a scope saw it.

    The SDK's ``results()`` retrieves the batch first, so the body is the
    one of the results URL, not the last one captured.
    """
    response = getattr(decoder, "http_response", None)
    request = getattr(response, "request", None)
    url = str(request.url) if isinstance(request, httpx.Request) else None
    for found in reversed(captured):
        if found.method != "GET" or not 200 <= found.status < 300:
            continue
        if found.url == url or (
            url is None and httpx.URL(found.url).path.endswith("/results")
        ):
            return found.text
    return None


class AnthropicBatchBackend(BatchBackend):
    """Anthropic Message Batches API."""

    provider_name = "anthropic"
    default_max_requests = MAX_BATCH_REQUESTS
    default_max_bytes = MAX_BATCH_BYTES
    lists_newest_first = True

    def _wrap(self, request: BatchRequest, body: dict[str, Any]) -> dict[str, Any]:
        return {"custom_id": request.custom_id, "params": dict(body)}

    def _submit_part(self, entries: list[dict[str, Any]], *, label: str) -> BatchHandle:
        batch = self.client.messages.batches.create(requests=entries)
        return BatchHandle(
            provider=self.provider_name,
            batch_id=str(batch.id),
            custom_ids=[str(entry["custom_id"]) for entry in entries],
            metadata={"request_count": len(entries)},
        )

    def _create_part(
        self,
        part: BatchPart,
        upload: PartUpload,
        *,
        submission_id: str,
        label: str,
    ) -> BatchHandle:
        batch = self._guarded_create(
            part,
            upload,
            submission_id,
            lambda: without_retries(self.client).messages.batches.create(
                requests=part.entries
            ),
        )
        return BatchHandle(
            provider=self.provider_name,
            batch_id=str(batch.id),
            custom_ids=part.custom_ids,
            metadata={"request_count": len(part.entries)},
        )

    def _list_page(
        self, page_size: int, cursor: str | None
    ) -> tuple[list[BatchListing], str | None]:
        kwargs: dict[str, Any] = {"limit": page_size}
        if cursor:
            kwargs["after_id"] = cursor
        page = self.client.messages.batches.list(**kwargs)
        data = get_field(page, "data")
        batches = list(page if data is None else data)
        listings = [self._listing(batch) for batch in batches]
        has_more = get_field(page, "has_more")
        more = has_more if isinstance(has_more, bool) else bool(batches)
        if data is None or not more or not batches:
            return listings, None
        last_id = get_field(page, "last_id") or get_field(batches[-1], "id")
        return listings, str(last_id or "") or None

    def _listing(self, batch: Any) -> BatchListing:
        status, _, counts = _map_status(batch)
        reported = get_field(batch, "request_counts") is not None
        return BatchListing(
            provider=self.provider_name,
            batch_id=str(get_field(batch, "id") or ""),
            status=status,
            provider_status=str(get_field(batch, "processing_status") or ""),
            created_at=as_datetime(get_field(batch, "created_at")),
            request_count=sum(counts.values()) if reported else None,
        )

    def status(self, handle: BatchHandle) -> BatchStatusInfo:
        """Return the batch status from its ``processing_status`` and counts."""
        batch = self.client.messages.batches.retrieve(handle.batch_id)
        status, error_message, counts = _map_status(batch)
        processing_status = str(get_field(batch, "processing_status") or "")
        results_url = get_field(batch, "results_url")
        return BatchStatusInfo(
            status=status,
            provider_status=processing_status,
            total_requests=sum(counts.values()),
            completed_requests=counts["succeeded"],
            failed_requests=counts["errored"],
            pending_requests=counts["processing"],
            error_message=error_message,
            results_available=processing_status == "ended" and bool(results_url),
            output_file_id=str(results_url) if results_url else None,
        )

    def _fetch_results(
        self, handle: BatchHandle
    ) -> tuple[list[BatchResultItem], str | None]:
        batch = self.client.messages.batches.retrieve(handle.batch_id)
        processing_status = get_field(batch, "processing_status")
        if processing_status != "ended":
            raise BatchNotReadyError(
                f"Anthropic batch {handle.batch_id} is {processing_status or 'unknown'}"
            )
        scope = current_scope()
        mark = len(scope.responses) if scope is not None else 0
        decoder = self.client.messages.batches.results(handle.batch_id)
        captured = scope.responses[mark:] if scope is not None else []
        body = _captured_results(decoder, captured)
        if body is None:
            return [parse_result(result) for result in decoder], None
        close = getattr(decoder, "close", None)
        if callable(close):
            close()
        return parse_lines(body), None

    def _cancel(self, handle: BatchHandle) -> None:
        self.client.messages.batches.cancel(handle.batch_id)

    def list_active(self, *, page_size: int = 100) -> list[BatchSummary]:
        """Return the message batches whose processing has not ended."""
        active: list[BatchSummary] = []
        for batch in self.client.messages.batches.list(limit=page_size):
            processing_status = str(get_field(batch, "processing_status") or "")
            if processing_status == "ended":
                continue
            status, _, _ = _map_status(batch)
            active.append(
                BatchSummary(
                    provider=self.provider_name,
                    batch_id=str(get_field(batch, "id") or ""),
                    status=status,
                    provider_status=processing_status,
                )
            )
        return active
