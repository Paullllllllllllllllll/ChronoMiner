"""Google Gemini Batch API backend.

Request bodies follow the SDK's ``InlinedRequest`` shape: ``model``,
``contents`` and a ``config`` holding ``system_instruction`` and the
generation parameters. Every body names the same model, which becomes the
batch model. A part under the inline limit whose requests carry no response
schema goes inline with its custom id in the request metadata; a part with a
schema, or a larger one, is uploaded as a JSONL file in the REST request
shape, since the Batch API enforces the schema of a file request only
(:func:`goes_inline`).

A part created through the part API carries the submission id in its batch
display name, which google-genai reports on list (``_BatchJob_from_mldev``
maps ``metadata.displayName``); the Gemini API reports no request count
there. Results from a file keep their downloaded lines. Inline results keep
their element of the ``inlinedResponses`` array of the batch GET body when a
capture scope (:mod:`.capture`) saw that body, else the JSON dump of the SDK
record. A file line that is not a JSON object is kept as an item without a
custom id.
"""

from __future__ import annotations

import io
import json
import logging
import time
from collections.abc import Mapping
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
    as_plain_dict,
    count,
    get_field,
    jsonl_bytes,
    jsonl_lines,
    malformed_line,
)
from .capture import CapturedResponse, current_scope, sdk_dump
from .responses import outcome_of
from .usage import Usage

__all__ = [
    "GoogleBatchBackend",
    "MAX_BATCH_BYTES",
    "MAX_BATCH_REQUESTS",
    "MAX_DISPLAY_NAME",
    "MAX_INLINE_BYTES",
    "display_name",
    "file_request",
    "goes_inline",
    "parse_response",
    "usage_from_response",
]

logger = logging.getLogger(__name__)

MAX_BATCH_REQUESTS = 50_000
MAX_BATCH_BYTES = 2 * 1024 * 1024 * 1024
MAX_INLINE_BYTES = 20 * 1024 * 1024
MAX_DISPLAY_NAME = 128
_MAX_FILE_ID = 40
"""The longest file ID (the name without ``files/``) the Files API deletes."""

_STATUS_MAP = {
    "JOB_STATE_QUEUED": BatchStatus.PENDING,
    "JOB_STATE_PENDING": BatchStatus.PENDING,
    "JOB_STATE_RUNNING": BatchStatus.IN_PROGRESS,
    "JOB_STATE_CANCELLING": BatchStatus.IN_PROGRESS,
    "JOB_STATE_PAUSED": BatchStatus.IN_PROGRESS,
    "JOB_STATE_UPDATING": BatchStatus.IN_PROGRESS,
    "JOB_STATE_SUCCEEDED": BatchStatus.COMPLETED,
    "JOB_STATE_PARTIALLY_SUCCEEDED": BatchStatus.COMPLETED,
    "JOB_STATE_FAILED": BatchStatus.FAILED,
    "JOB_STATE_CANCELLED": BatchStatus.CANCELLED,
    "JOB_STATE_EXPIRED": BatchStatus.EXPIRED,
}

_REQUEST_LEVEL_CONFIG = frozenset(
    {"safety_settings", "tools", "tool_config", "cached_content", "labels"}
)


def _state_name(job: Any) -> str:
    state = get_field(job, "state")
    if state is None:
        return ""
    name = getattr(state, "name", None)
    return str(name if isinstance(name, str) else state)


def _pick(data: Mapping[str, Any], snake: str, camel: str) -> Any:
    value = data.get(snake)
    return data.get(camel) if value is None else value


def usage_from_response(response: Mapping[str, Any]) -> Usage:
    """Return the usage of a response dict in REST (camel) or SDK (snake) keys.

    Thought tokens count as output and as reasoning tokens.
    """
    raw = _pick(response, "usage_metadata", "usageMetadata")
    if not isinstance(raw, Mapping):
        return Usage()
    prompt = count(_pick(raw, "prompt_token_count", "promptTokenCount"))
    candidates = count(_pick(raw, "candidates_token_count", "candidatesTokenCount"))
    thoughts = count(_pick(raw, "thoughts_token_count", "thoughtsTokenCount"))
    output = candidates + thoughts
    return Usage(
        input_tokens=prompt,
        output_tokens=output,
        total_tokens=count(_pick(raw, "total_token_count", "totalTokenCount"))
        or prompt + output,
        cached_tokens=count(
            _pick(raw, "cached_content_token_count", "cachedContentTokenCount")
        ),
        reasoning_tokens=thoughts,
    )


def _error_text(error: Any) -> str:
    message = get_field(error, "message")
    if message:
        return str(message)
    return json.dumps(error) if isinstance(error, dict) else str(error)


def parse_response(
    custom_id: str, response: Mapping[str, Any], origin: ResultOrigin = "output"
) -> BatchResultItem:
    """Return the item of one ``GenerateContentResponse`` dict.

    Text is the concatenation of the first candidate's non-thought parts.
    """
    item = BatchResultItem(
        custom_id=custom_id,
        response=dict(response),
        usage=usage_from_response(response),
        origin=origin,
    )
    candidates = response.get("candidates")
    first = candidates[0] if isinstance(candidates, list) and candidates else {}
    content = first.get("content") if isinstance(first, Mapping) else None
    parts = content.get("parts") if isinstance(content, Mapping) else None
    item.text = "".join(
        str(part.get("text") or "")
        for part in (parts if isinstance(parts, list) else [])
        if isinstance(part, Mapping) and not part.get("thought")
    )
    finish = _pick(first, "finish_reason", "finishReason") if first else None
    feedback = _pick(response, "prompt_feedback", "promptFeedback")
    reason = finish
    if not reason and isinstance(feedback, Mapping):
        reason = _pick(feedback, "block_reason", "blockReason")
    item.settle(
        outcome_of({"finish_reason": finish, "prompt_feedback": feedback}),
        "No text content in Gemini batch response "
        f"(finishReason={reason or 'unknown'})",
    )
    return item


def file_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an ``InlinedRequest``-shaped body to the REST request of a file line."""
    config = dict(request.get("config") or {})
    rest: dict[str, Any] = {"contents": request.get("contents") or []}
    instruction = config.pop("system_instruction", None)
    if isinstance(instruction, str):
        rest["system_instruction"] = {"parts": [{"text": instruction}]}
    elif instruction is not None:
        rest["system_instruction"] = instruction
    config.pop("http_options", None)
    for key in _REQUEST_LEVEL_CONFIG & config.keys():
        rest[key] = config.pop(key)
    if config:
        rest["generation_config"] = config
    return rest


def _model_of(entries: list[dict[str, Any]]) -> str:
    models = {str(entry["request"].get("model") or "") for entry in entries}
    if len(models) != 1 or "" in models:
        raise ValueError("every Gemini batch body must name the same model")
    model = models.pop()
    return model if model.startswith("models/") else f"models/{model}"


def _without_model(request: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in request.items() if key != "model"}


def _inline_source(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **_without_model(entry["request"]),
            "metadata": {
                **(entry["request"].get("metadata") or {}),
                "key": str(entry["key"]),
            },
        }
        for entry in entries
    ]


_SCHEMA_KEYS = ("response_json_schema", "response_schema")


def _carries_schema(entry: Mapping[str, Any]) -> bool:
    config = entry["request"].get("config")
    return isinstance(config, Mapping) and any(key in config for key in _SCHEMA_KEYS)


def goes_inline(entries: list[dict[str, Any]]) -> bool:
    """Whether a part's requests travel inline in the create call.

    A part goes inline only below :data:`MAX_INLINE_BYTES` and when no
    request carries a response schema: the Batch API ignores the schema of
    an inline request (a live probe got ``{"message": "Hello!"}`` back
    against a schema that allowed only ``{"word": ...}``), while the
    uploaded JSONL file enforces it.
    """
    if any(_carries_schema(entry) for entry in entries):
        return False
    return len(jsonl_bytes(entries)) < MAX_INLINE_BYTES


def _file_lines(entries: list[dict[str, Any]]) -> bytes:
    return jsonl_bytes(
        [
            {"key": entry["key"], "request": file_request(entry["request"])}
            for entry in entries
        ]
    )


def display_name(label: str, submission_id: str, part: int) -> str:
    """Return the display name of a part's batch; it always keeps the id."""
    suffix = f"-{submission_id}-p{part}"
    return f"{label[: max(0, MAX_DISPLAY_NAME - len(suffix))]}{suffix}"


class GoogleBatchBackend(BatchBackend):
    """Gemini Batch API with inline and file submission."""

    provider_name = "google"
    default_max_requests = MAX_BATCH_REQUESTS
    default_max_bytes = MAX_BATCH_BYTES

    def _wrap(self, request: BatchRequest, body: dict[str, Any]) -> dict[str, Any]:
        if not body.get("model"):
            raise ValueError(f"Gemini batch body of {request.custom_id} has no model")
        return {"key": request.custom_id, "request": dict(body)}

    def _submit_part(self, entries: list[dict[str, Any]], *, label: str) -> BatchHandle:
        model = _model_of(entries)
        name = f"{label}-{int(time.time())}"
        uploaded_name: str | None = None
        if goes_inline(entries):
            src: Any = _inline_source(entries)
        else:
            uploaded = self.client.files.upload(
                file=io.BytesIO(_file_lines(entries)),
                config={"display_name": name, "mime_type": "jsonl"},
            )
            uploaded_name = str(uploaded.name)
            src = uploaded_name
        try:
            job = self.client.batches.create(
                model=model, src=src, config={"display_name": name}
            )
        except Exception:
            self._delete_files(uploaded_name)
            raise
        return BatchHandle(
            provider=self.provider_name,
            batch_id=str(job.name),
            custom_ids=[str(entry["key"]) for entry in entries],
            metadata={
                "request_count": len(entries),
                "input_file_name": uploaded_name,
            },
        )

    def request_bodies(self, part: BatchPart) -> dict[str, dict[str, Any]]:
        """Return each request of *part* as submitted.

        An inline part sends the ``InlinedRequest`` dicts the SDK receives;
        a part that carries a response schema or is too large for inline
        submission uploads the REST request of each file line
        (:func:`file_request`, :func:`goes_inline`).
        """
        if goes_inline(part.entries):
            sent = _inline_source(part.entries)
        else:
            sent = [file_request(entry["request"]) for entry in part.entries]
        return {
            request.custom_id: body
            for request, body in zip(part.requests, sent, strict=True)
        }

    def _upload_part(
        self, part: BatchPart, *, submission_id: str, label: str
    ) -> PartUpload:
        _model_of(part.entries)
        if goes_inline(part.entries):
            return PartUpload(self.provider_name)
        uploaded = self.client.files.upload(
            file=io.BytesIO(_file_lines(part.entries)),
            config={
                "display_name": display_name(label, submission_id, part.index),
                "mime_type": "jsonl",
            },
        )
        return PartUpload(self.provider_name, str(uploaded.name))

    def _create_part(
        self,
        part: BatchPart,
        upload: PartUpload,
        *,
        submission_id: str,
        label: str,
    ) -> BatchHandle:
        model = _model_of(part.entries)
        src: Any = upload.file_id or _inline_source(part.entries)
        config = {"display_name": display_name(label, submission_id, part.index)}
        job = self._guarded_create(
            part,
            upload,
            submission_id,
            lambda: self.client.batches.create(model=model, src=src, config=config),
        )
        return BatchHandle(
            provider=self.provider_name,
            batch_id=str(job.name),
            custom_ids=part.custom_ids,
            metadata={
                "request_count": len(part.entries),
                "input_file_name": upload.file_id,
            },
        )

    def _delete_upload(self, file_id: str) -> None:
        self.client.files.delete(name=file_id)

    def _list_page(
        self, page_size: int, cursor: str | None
    ) -> tuple[list[BatchListing], str | None]:
        config: dict[str, Any] = {"page_size": page_size}
        if cursor:
            config["page_token"] = cursor
        pager = self.client.batches.list(config=config)
        page = get_field(pager, "page")
        jobs = list(pager if page is None else page)
        listings = [self._listing(job) for job in jobs]
        if page is None:
            return listings, None
        token = get_field(get_field(pager, "config"), "page_token")
        return listings, str(token) if token else None

    def _listing(self, job: Any) -> BatchListing:
        state = _state_name(job)
        name = get_field(job, "display_name")
        return BatchListing(
            provider=self.provider_name,
            batch_id=str(get_field(job, "name") or ""),
            status=_STATUS_MAP.get(state, BatchStatus.UNKNOWN),
            provider_status=state,
            created_at=as_datetime(get_field(job, "create_time")),
            display_name=str(name) if name else None,
        )

    def status(self, handle: BatchHandle) -> BatchStatusInfo:
        """Return the status of the batch job from its state name."""
        job = self.client.batches.get(name=handle.batch_id)
        state = _state_name(job)
        status = _STATUS_MAP.get(state, BatchStatus.UNKNOWN)
        total = len(handle.custom_ids) or count(handle.metadata.get("request_count"))
        error_message = None
        if status is BatchStatus.UNKNOWN:
            error_message = f"Unmapped Google batch state: {state or 'none'}"
        elif status is BatchStatus.FAILED:
            error = get_field(job, "error")
            error_message = _error_text(error) if error else None
        dest = get_field(job, "dest")
        file_name = get_field(dest, "file_name")
        inline = get_field(dest, "inlined_responses")
        return BatchStatusInfo(
            status=status,
            provider_status=state,
            total_requests=total,
            error_message=error_message,
            results_available=status is BatchStatus.COMPLETED
            and bool(file_name or inline),
            output_file_id=str(file_name) if file_name else None,
        )

    def _fetch_results(
        self, handle: BatchHandle
    ) -> tuple[list[BatchResultItem], str | None]:
        scope = current_scope()
        mark = len(scope.responses) if scope is not None else 0
        job = self.client.batches.get(name=handle.batch_id)
        state = _state_name(job)
        if _STATUS_MAP.get(state) not in TERMINAL_STATUSES:
            raise BatchNotReadyError(
                f"Google batch {handle.batch_id} is {state or 'unknown'}"
            )
        dest = get_field(job, "dest")
        file_name = get_field(dest, "file_name")
        inline = get_field(dest, "inlined_responses")
        if file_name:
            data = self.client.files.download(file=file_name)
            text = data.decode("utf-8") if isinstance(data, bytes) else str(data)
            return self._parse_file(text), None
        if inline:
            wire = _wire_inline(scope.responses[mark:] if scope is not None else [])
            return self._parse_inline(handle, inline, wire), None
        error = get_field(job, "error")
        return [], _error_text(error) if error else f"batch {state} without results"

    @staticmethod
    def _parse_file(text: str) -> list[BatchResultItem]:
        items: list[BatchResultItem] = []
        for number, line in jsonl_lines(text):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = None
            if not isinstance(record, dict):
                items.append(malformed_line(number, line, "the Gemini result file"))
                continue
            custom_id = str(record.get("key") or "")
            error = record.get("error")
            if error:
                item = BatchResultItem(
                    custom_id=custom_id,
                    error=_error_text(error),
                    error_code=_code(error),
                    response={"error": error},
                )
            else:
                response = record.get("response")
                item = parse_response(
                    custom_id, response if isinstance(response, dict) else {}
                )
            item.raw, item.raw_kind = line, "line"
            items.append(item)
        return items

    @staticmethod
    def _parse_inline(
        handle: BatchHandle, responses: Any, wire: list[Any] | None = None
    ) -> list[BatchResultItem]:
        """Parse inline results; *wire* holds their elements of the GET body."""
        responses = list(responses)
        if wire is not None and len(wire) != len(responses):
            logger.warning(
                "Gemini batch %s: %d inline results on the wire, %d parsed; "
                "keeping the SDK dumps",
                handle.batch_id,
                len(wire),
                len(responses),
            )
            wire = None
        items: list[BatchResultItem] = []
        for index, inline in enumerate(responses):
            metadata = get_field(inline, "metadata")
            custom_id = get_field(metadata, "key") if metadata else None
            if not custom_id and index < len(handle.custom_ids):
                custom_id = handle.custom_ids[index]
            custom_id = str(custom_id or "")
            error = get_field(inline, "error")
            response = get_field(inline, "response")
            if error:
                item = BatchResultItem(
                    custom_id=custom_id,
                    error=_error_text(error),
                    error_code=_code(error),
                    response={"error": _compact_dict(error)},
                )
            elif response is None:
                item = BatchResultItem(
                    custom_id=custom_id,
                    error="Gemini inline result carried no response",
                )
            else:
                item = parse_response(custom_id, _compact_dict(response))
            if wire is None:
                item.raw, item.raw_kind = sdk_dump(inline), "sdk"
            else:
                # The element keeps the wire keys and values, not the
                # whitespace of the body it was cut from.
                item.raw = json.dumps(wire[index], ensure_ascii=False)
                item.raw_kind = "line"
            items.append(item)
        return items

    def _cancel(self, handle: BatchHandle) -> None:
        self.client.batches.cancel(name=handle.batch_id)

    def cleanup(self, handle: BatchHandle) -> None:
        """Delete the uploaded input file and the result file; never raises."""
        output_name = None
        try:
            job = self.client.batches.get(name=handle.batch_id)
        except Exception as exc:  # noqa: BLE001 - cleanup is best effort
            logger.warning("Could not look up batch %s: %s", handle.batch_id, exc)
        else:
            output_name = get_field(get_field(job, "dest"), "file_name")
        self._delete_files(handle.metadata.get("input_file_name"), output_name)

    def _delete_files(self, *names: Any) -> None:
        for name in names:
            if not name:
                continue
            if len(str(name).removeprefix("files/")) > _MAX_FILE_ID:
                # Gemini names batch result files files/batch-<batch id>, which
                # the Files API refuses to delete; the provider keeps them.
                logger.debug("Not deleting Google file %s: its ID is too long", name)
                continue
            try:
                self.client.files.delete(name=str(name))
            except Exception as exc:  # noqa: BLE001 - cleanup is best effort
                logger.warning("Failed to delete Google file %s: %s", name, exc)

    def list_active(self, *, page_size: int = 100) -> list[BatchSummary]:
        """Return the batch jobs that have not reached a terminal state."""
        active: list[BatchSummary] = []
        for job in self.client.batches.list(config={"page_size": page_size}):
            state = _state_name(job)
            status = _STATUS_MAP.get(state, BatchStatus.UNKNOWN)
            if status in TERMINAL_STATUSES:
                continue
            active.append(
                BatchSummary(
                    provider=self.provider_name,
                    batch_id=str(get_field(job, "name") or ""),
                    status=status,
                    provider_status=state,
                )
            )
        return active


def _wire_inline(captured: list[CapturedResponse]) -> list[Any] | None:
    """Return the ``inlinedResponses`` array of the captured batch GET body.

    None when no successful GET was captured or its body holds no array.
    """
    for response in reversed(captured):
        if response.method != "GET" or not 200 <= response.status < 300:
            continue
        try:
            body = json.loads(response.body)
        except ValueError:
            return None
        output = get_field(get_field(body, "metadata"), "output")
        inlined = get_field(get_field(output, "inlinedResponses"), "inlinedResponses")
        return inlined if isinstance(inlined, list) else None
    return None


def _compact_dict(obj: Any) -> dict[str, Any]:
    """Dump an SDK model without its unset fields, else as a plain dict."""
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json", exclude_none=True)
        except Exception:  # noqa: BLE001 - fall back to the generic dump
            logger.debug("model_dump failed on %s", type(obj).__name__)
        else:
            if isinstance(dumped, dict):
                return dumped
    return as_plain_dict(obj)


def _code(error: Any) -> str | None:
    code = get_field(error, "code") or get_field(error, "status")
    return str(code) if code else None
