"""Fake batch services of OpenAI, Anthropic and Google for characterization tests.

``FakeBatchService.install`` replaces the SDK client classes the batch
backends build (``openai.OpenAI``, ``anthropic.Anthropic`` and
``google.genai.Client``) wherever the tool binds them (see ``seams.replace``)
with fake clients that share one service. The real batch backends therefore
run, including request building and result parsing. Each fake client lists
and cancels the batches of its provider; a cancel for another provider's
batch fails with a 404.

The service keeps every submitted request, reports a batch completed on its
``polls_until_complete``-th status call and then builds one result per
request from the ``FakeLLM`` unit scripts. A request is matched to its unit
like a synchronous one: the page marker of its last image, else the first
record heading of its user text. The answer follows the schema the request
enforces (OpenAI ``text.format``, Gemini ``response_schema``) or, without
one, ``FakeLLM.schema``. A script reply that raises becomes a failed request
in the provider's result format: an error-file line (or an error line in the
output file) for OpenAI, an ``errored`` entry for Anthropic and an inline
error for Google; ``drop_units`` leaves requests out of the results. Every
API call is recorded in ``calls``; ``mark`` and ``request_records(since=...)``
scope the records to one run.
"""

from __future__ import annotations

import base64
import binascii
import copy
import io
import json
import re
import threading
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from PIL import Image

from tests.characterization.fakes import (
    ADJUST,
    EXTRACT,
    USAGE,
    FakeAPIError,
    FakeLLM,
    boundary_payload,
    is_boundary_request,
    page_unit,
    redact,
    render,
    text_unit,
)
from tests.characterization.inputs import decode_marker
from tests.characterization.seams import replace

Provider = Literal["openai", "anthropic", "google"]
OpenAIErrorForm = Literal["error_file", "output"]

CREATED_AT = 1_767_225_600
"""Fixed Unix time of every fake batch and file."""

LONG_TEXT = 120
"""Text values longer than this are recorded once under ``prompts``."""

SDK_CLIENTS: dict[str, tuple[str, str]] = {
    "openai": ("openai", "OpenAI"),
    "anthropic": ("anthropic", "Anthropic"),
    "google": ("google.genai", "Client"),
}
"""(public module, class name) of the SDK client class each provider fakes."""

_DISPLAY_TIME_RE = re.compile(r"-\d{9,}$")


@dataclass
class Outcome:
    """The scripted answer to one batch request: text or an error."""

    content: str | None = None
    status_code: int = 200
    error: str | None = None

    @property
    def failed(self) -> bool:
        """Return True when the request failed."""
        return self.error is not None


@dataclass
class BatchEntry:
    """One request of a submitted batch."""

    custom_id: str
    unit: str | None
    body: dict[str, Any]
    outcome: Outcome | None = None
    dropped: bool = False


@dataclass
class FakeBatch:
    """One submitted batch and its state."""

    provider: Provider
    batch_id: str
    entries: list[BatchEntry]
    info: dict[str, Any] = field(default_factory=dict)
    polls: int = 0
    completed: bool = False
    output_file_id: str | None = None
    error_file_id: str | None = None
    cancelled: bool = False


def _decode_image(data: bytes) -> tuple[dict[str, Any], int | None]:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return {"width": image.width, "height": image.height}, decode_marker(image)


def _image_from_data_url(url: str) -> tuple[dict[str, Any], int | None] | None:
    if not url.startswith("data:") or ";base64," not in url:
        return None
    header, data = url.split(",", 1)
    try:
        raw = base64.b64decode(data)
    except (binascii.Error, ValueError):
        return None
    info, page = _decode_image(raw)
    return {"mime": header[5:].split(";", 1)[0], **info, "page": page}, page


def _image_from_base64(data: Any, mime: Any) -> tuple[dict[str, Any], int | None]:
    raw = data if isinstance(data, bytes) else base64.b64decode(str(data))
    info, page = _decode_image(raw)
    return {"mime": mime, **info, "page": page}, page


def describe_body(value: Any) -> tuple[Any, list[int | None]]:
    """Return *value* with images replaced by descriptions, and their pages.

    Images are data URLs (OpenAI) and base64 ``data`` beside a
    ``media_type`` (Anthropic) or ``mime_type`` (Google).
    """
    pages: list[int | None] = []

    def walk(node: Any) -> Any:
        if isinstance(node, Mapping):
            mime = node.get("media_type", node.get("mime_type"))
            result: dict[str, Any] = {}
            for key, item in node.items():
                if key == "data" and mime is not None:
                    described, page = _image_from_base64(item, mime)
                    pages.append(page)
                    result[key] = {"image": described}
                else:
                    result[str(key)] = walk(item)
            return result
        if isinstance(node, list | tuple):
            return [walk(item) for item in node]
        if isinstance(node, str):
            image = _image_from_data_url(node)
            if image is not None:
                pages.append(image[1])
                return {"image": image[0]}
        return node

    return walk(value), pages


def _request_body(provider: Provider, body: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the request a JSONL line carries: OpenAI nests it under ``body``."""
    if provider == "openai":
        nested = body.get("body")
        if isinstance(nested, Mapping):
            return nested
    return body


def _system_text(provider: Provider, body: Mapping[str, Any]) -> str:
    if provider == "openai":
        parts = [
            str(block.get("text", ""))
            for message in body.get("input", [])
            if message.get("role") == "system"
            for block in message.get("content", [])
        ]
        return "\n".join(parts)
    if provider == "anthropic":
        system = body.get("system", "")
        if isinstance(system, list):
            return "\n".join(str(block.get("text", "")) for block in system)
        return str(system)
    config = body.get("config") or {}
    instruction = body.get("system_instruction") or config.get("system_instruction")
    if isinstance(instruction, Mapping):
        return "\n".join(
            str(part.get("text", "")) for part in instruction.get("parts", [])
        )
    return str(instruction or "")


def _texts(blocks: Iterable[Any]) -> list[str]:
    return [
        str(block["text"])
        for block in blocks
        if isinstance(block, Mapping) and "text" in block
    ]


def _user_text(provider: Provider, body: Mapping[str, Any]) -> str:
    parts: list[str] = []
    if provider == "openai":
        for message in body.get("input", []):
            if message.get("role") != "system":
                parts += _texts(message.get("content", []))
    elif provider == "anthropic":
        for message in body.get("messages", []):
            parts += _texts(message.get("content", []))
    else:
        for content in body.get("contents", []):
            parts += _texts(content.get("parts", []))
    return "\n".join(parts)


def _enforced_schema(provider: Provider, body: Mapping[str, Any]) -> Any:
    if provider == "openai":
        fmt = (body.get("text") or {}).get("format")
        return fmt.get("schema") if isinstance(fmt, Mapping) else None
    if provider == "google":
        config = body.get("config") or body.get("generation_config") or {}
        return config.get("response_schema") or config.get("response_json_schema")
    return None


class FakeBatchService:
    """Batch API stand-ins of three providers over ``FakeLLM`` unit scripts.

    ``calls`` holds one record per API call; ``batches`` every submitted
    batch by id; ``clients`` one record per client construction.
    ``fail_on`` maps a call name (such as ``"openai.batches.create"``) to an
    exception that call raises after it is recorded. The results leave out
    every request whose unit is in ``drop_units``.

    An OpenAI request line nests the Responses body under ``body``; the
    answer always follows that nested body's response format and system
    prompt, so a request that enforces a schema gets schema JSON, as a real
    batch returns.
    """

    def __init__(
        self,
        llm: FakeLLM,
        *,
        polls_until_complete: int = 1,
        openai_error_form: OpenAIErrorForm = "error_file",
    ) -> None:
        self.llm = llm
        self.polls_until_complete = polls_until_complete
        self.openai_error_form: OpenAIErrorForm = openai_error_form
        self.fail_on: dict[str, BaseException] = {}
        self.drop_units: set[str] = set()
        self.calls: list[dict[str, Any]] = []
        self.clients: list[dict[str, Any]] = []
        self.batches: dict[str, FakeBatch] = {}
        self.files: dict[str, bytes] = {}
        self._order: list[str] = []
        self._attempts: dict[str | None, int] = {}
        self._counter = 0
        self._lock = threading.RLock()

    # -- installation ----------------------------------------------------------
    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeBatchService:
        """Replace the SDK client classes."""
        providers: tuple[Provider, ...] = ("openai", "anthropic", "google")
        for provider in providers:
            module_name, name = SDK_CLIENTS[provider]
            replace(monkeypatch, module_name, name, self.client_class(provider))
        return self

    def client_class(self, provider: Provider) -> Any:
        """Return a stand-in for the SDK client class of *provider*."""
        builders = {
            "openai": FakeOpenAIClient,
            "anthropic": FakeAnthropicClient,
            "google": FakeGoogleClient,
        }
        builder = builders[provider]

        def construct(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                self.clients.append({"provider": provider, "kwargs": redact(kwargs)})
            return builder(self)

        return construct

    def reset(self) -> None:
        """Forget batches, files, records, attempts and id counters."""
        with self._lock:
            self.calls.clear()
            self.clients.clear()
            self.batches.clear()
            self.files.clear()
            self._order.clear()
            self._attempts.clear()
            self._counter = 0

    def reset_records(self) -> None:
        """Forget the call and client records but keep the batches."""
        with self._lock:
            self.calls.clear()
            self.clients.clear()

    def mark(self) -> int:
        """Return a mark for ``request_records(since=...)``: batches so far."""
        with self._lock:
            return len(self._order)

    # -- ids and records -------------------------------------------------------
    def new_id(self, prefix: str) -> str:
        """Return the next id with *prefix*; numbers are unique per service."""
        with self._lock:
            self._counter += 1
            return f"{prefix}{self._counter:04d}"

    def record(self, provider: Provider, method: str, /, **details: Any) -> None:
        """Record one API call; raise the exception ``fail_on`` holds for it."""
        call = f"{provider}.{method}"
        with self._lock:
            self.calls.append({"call": call, **details})
        failure = self.fail_on.get(call)
        if failure is not None:
            raise failure

    def id_replacements(self) -> dict[str, str]:
        """Return a placeholder for every batch and file id, in creation order."""
        replacements: dict[str, str] = {}
        counts: dict[str, int] = {}
        with self._lock:
            for item in self._order:
                if item in self.batches:
                    kind = f"{self.batches[item].provider}-batch"
                else:
                    kind = "file"
                counts[kind] = counts.get(kind, 0) + 1
                replacements[item] = f"<{kind}-{counts[kind]}>"
        return replacements

    def batch_states(self) -> list[dict[str, Any]]:
        """Return the id, provider and state of every batch, in creation order."""
        with self._lock:
            batches = [
                self.batches[item] for item in self._order if item in self.batches
            ]
            return [
                {
                    "batch_id": batch.batch_id,
                    "provider": batch.provider,
                    "state": (
                        "cancelled"
                        if batch.cancelled
                        else "completed"
                        if batch.completed
                        else "pending"
                    ),
                    "polls": batch.polls,
                }
                for batch in batches
            ]

    def request_records(self, since: int = 0) -> dict[str, Any]:
        """Return calls, clients and the requests of batches created after *since*.

        Images are described and long texts stored once under ``prompts``.
        """
        prompts: dict[str, str] = {}
        ids: dict[str, str] = {}

        def shorten(node: Any) -> Any:
            if isinstance(node, dict):
                return {key: shorten(item) for key, item in node.items()}
            if isinstance(node, list):
                return [shorten(item) for item in node]
            if isinstance(node, str) and len(node) > LONG_TEXT:
                prompt_id = ids.get(node)
                if prompt_id is None:
                    prompt_id = f"batch-prompt-{len(ids) + 1}"
                    ids[node] = prompt_id
                    prompts[prompt_id] = node
                return {"prompt": prompt_id, "chars": len(node)}
            return node

        requests: list[dict[str, Any]] = []
        with self._lock:
            for batch_id in self._order[since:]:
                batch = self.batches.get(batch_id)
                if batch is None:
                    continue
                for entry in batch.entries:
                    described, _ = describe_body(entry.body)
                    requests.append(
                        {
                            "provider": batch.provider,
                            "batch_id": batch.batch_id,
                            "custom_id": entry.custom_id,
                            "unit": entry.unit,
                            "body": shorten(described),
                        }
                    )
            calls = shorten(copy.deepcopy(self.calls))
            clients = copy.deepcopy(self.clients)
        return {
            "clients": clients,
            "calls": calls,
            "requests": requests,
            "prompts": prompts,
        }

    # -- batches ---------------------------------------------------------------
    def unit_of(self, provider: Provider, body: Mapping[str, Any]) -> str | None:
        """Return the unit of one request: its last image's marker or its text."""
        request = _request_body(provider, body)
        _, pages = describe_body(request)
        marked = [page for page in pages if page is not None]
        if marked:
            return page_unit(marked[-1])
        text = _user_text(provider, request)
        if is_boundary_request(_system_text(provider, request), None):
            return boundary_payload(text)[0]
        return text_unit(text)

    def submit(
        self,
        provider: Provider,
        prefix: str,
        requests: list[tuple[str, dict[str, Any]]],
        info: dict[str, Any] | None = None,
    ) -> FakeBatch:
        """Register a batch of (custom_id, body) requests and return it."""
        entries = [
            BatchEntry(custom_id, self.unit_of(provider, body), body)
            for custom_id, body in requests
        ]
        batch = FakeBatch(provider, self.new_id(prefix), entries, dict(info or {}))
        with self._lock:
            self.batches[batch.batch_id] = batch
            self._order.append(batch.batch_id)
        return batch

    def add_file(self, prefix: str, content: bytes) -> str:
        """Store *content* as a file and return its id."""
        file_id = self.new_id(prefix)
        with self._lock:
            self.files[file_id] = content
            self._order.append(file_id)
        return file_id

    def poll(self, batch: FakeBatch) -> FakeBatch:
        """Count one status call; complete the batch when its time has come."""
        with self._lock:
            batch.polls += 1
            if (
                not batch.completed
                and not batch.cancelled
                and batch.polls >= self.polls_until_complete
            ):
                self._complete(batch)
        return batch

    def lookup(self, batch_id: str, provider: Provider | None = None) -> FakeBatch:
        """Return the batch *batch_id* or raise a 404 error.

        With *provider*, a batch of another provider is not found either.
        """
        batch = self.batches.get(batch_id)
        if batch is None or (provider is not None and batch.provider != provider):
            raise FakeAPIError(404, f"Error code: 404 - batch {batch_id} not found")
        return batch

    def listed(self, provider: Provider) -> list[FakeBatch]:
        """Return *provider*'s batches, newest first, each counted as polled."""
        with self._lock:
            batches = [
                batch
                for batch in reversed(list(self.batches.values()))
                if batch.provider == provider
            ]
        return [self.poll(batch) for batch in batches]

    def cancel(self, provider: Provider, batch_id: str) -> FakeBatch:
        """Cancel *provider*'s batch *batch_id* unless it has already ended."""
        batch = self.lookup(batch_id, provider)
        with self._lock:
            batch.cancelled = batch.cancelled or not batch.completed
        return batch

    def _complete(self, batch: FakeBatch) -> None:
        for entry in batch.entries:
            entry.outcome = self.answer(batch.provider, entry)
            entry.dropped = entry.unit in self.drop_units
        batch.completed = True
        if batch.provider == "openai":
            self._openai_files(batch)
        if batch.provider == "google" and batch.info.get("src_file"):
            lines = [
                json.dumps({"key": entry.custom_id, **_google_rest_result(entry)})
                for entry in batch.entries
                if not entry.dropped
            ]
            batch.output_file_id = self.add_file(
                "files/fake-out", ("\n".join(lines) + "\n").encode("utf-8")
            )

    def answer(self, provider: Provider, entry: BatchEntry) -> Outcome:
        """Return the scripted outcome of one request (attempts count per unit)."""
        with self._lock:
            attempt = self._attempts.get(entry.unit, 0) + 1
            self._attempts[entry.unit] = attempt
        request = _request_body(provider, entry.body)
        system = _system_text(provider, request)
        schema = _enforced_schema(provider, request)
        role = ADJUST if is_boundary_request(system, schema) else EXTRACT
        reply, form = self.llm.scripted_reply(entry.unit, attempt)
        if reply is not None:
            if reply.exception is not None:
                status = getattr(reply.exception, "status_code", 500)
                return Outcome(status_code=int(status), error=str(reply.exception))
            if reply.text is not None:
                return Outcome(content=reply.text)
            if reply.form is not None:
                form = reply.form
        payload = self.llm.payload_for(
            role, entry.unit, _user_text(provider, request), schema
        )
        return Outcome(content=render(payload, form))

    # -- OpenAI result files ---------------------------------------------------
    def _openai_files(self, batch: FakeBatch) -> None:
        output: list[str] = []
        errors: list[str] = []
        model = ""
        for index, entry in enumerate(batch.entries, 1):
            if entry.dropped:
                continue
            outcome = entry.outcome or Outcome(content="")
            model = str(_request_body("openai", entry.body).get("model", model))
            line = _openai_result_line(index, entry, outcome, model)
            target = (
                errors
                if outcome.failed and self.openai_error_form == "error_file"
                else output
            )
            target.append(json.dumps(line, ensure_ascii=False))
        if output:
            batch.output_file_id = self.add_file(
                "file-fake", ("\n".join(output) + "\n").encode("utf-8")
            )
        if errors:
            batch.error_file_id = self.add_file(
                "file-fake", ("\n".join(errors) + "\n").encode("utf-8")
            )


def _openai_result_line(
    index: int, entry: BatchEntry, outcome: Outcome, model: str
) -> dict[str, Any]:
    request_id = f"req_fake{index:04d}"
    if outcome.failed:
        body: dict[str, Any] = {
            "error": {
                "message": outcome.error,
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_request",
            }
        }
    else:
        body = {
            "id": f"resp_fake{index:04d}",
            "object": "response",
            "created_at": CREATED_AT,
            "status": "completed",
            "model": model,
            "output": [
                {
                    "type": "message",
                    "id": f"msg_fake{index:04d}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": outcome.content or "",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": USAGE["input_tokens"],
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": USAGE["output_tokens"],
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": USAGE["total_tokens"],
            },
        }
    return {
        "id": f"batch_req_fake{index:04d}",
        "custom_id": entry.custom_id,
        "response": {
            "status_code": outcome.status_code,
            "request_id": request_id,
            "body": body,
        },
        "error": None,
    }


class _BinaryContent:
    """Stand-in for the SDK's binary response content."""

    def __init__(self, content: bytes) -> None:
        self.content = content

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")

    def read(self) -> bytes:
        return self.content


@dataclass
class _Page:
    """Stand-in for one page of a cursor listing; iterating walks its items."""

    data: list[Any]
    has_more: bool = False
    last_id: str | None = None

    def __iter__(self) -> Iterator[Any]:
        return iter(self.data)


class _OpenAIFiles:
    def __init__(self, service: FakeBatchService) -> None:
        self._service = service

    def create(self, *, file: Any, purpose: str, **kwargs: Any) -> Any:
        from openai.types import FileObject

        if isinstance(file, tuple):
            file = file[1]
        raw = file.read() if hasattr(file, "read") else bytes(file)
        lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
        self._service.record(
            "openai", "files.create", purpose=purpose, lines=len(lines)
        )
        file_id = self._service.add_file("file-fake", raw)
        return FileObject.model_validate(
            {
                "id": file_id,
                "bytes": len(raw),
                "created_at": CREATED_AT,
                "filename": "batch_input.jsonl",
                "object": "file",
                "purpose": purpose,
                "status": "processed",
            }
        )

    def content(self, file_id: str, **kwargs: Any) -> _BinaryContent:
        self._service.record("openai", "files.content", file_id=file_id)
        content = self._service.files.get(file_id)
        if content is None:
            raise FakeAPIError(404, f"Error code: 404 - file {file_id} not found")
        return _BinaryContent(content)

    def delete(self, file_id: str, **kwargs: Any) -> None:
        self._service.record("openai", "files.delete", file_id=file_id)
        with self._service._lock:
            self._service.files.pop(file_id, None)


class _OpenAIBatches:
    def __init__(self, service: FakeBatchService) -> None:
        self._service = service

    def create(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        service = self._service
        service.record(
            "openai",
            "batches.create",
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata,
        )
        raw = service.files.get(input_file_id, b"")
        lines = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
        batch = service.submit(
            "openai",
            "batch_fake",
            [(str(line["custom_id"]), line) for line in lines],
            {
                "input_file_id": input_file_id,
                "endpoint": endpoint,
                "completion_window": completion_window,
                "metadata": metadata,
            },
        )
        return _openai_batch(batch)

    def retrieve(self, batch_id: str, **kwargs: Any) -> Any:
        service = self._service
        service.record("openai", "batches.retrieve", batch_id=batch_id)
        return _openai_batch(service.poll(service.lookup(batch_id)))

    def list(self, *, limit: int | None = None, after: str | None = None) -> _Page:
        service = self._service
        service.record("openai", "batches.list", limit=limit, after=after)
        batches = [_openai_batch(batch) for batch in service.listed("openai")]
        last = batches[-1].id if batches else None
        return _Page(data=batches, has_more=False, last_id=last)

    def cancel(self, batch_id: str, **kwargs: Any) -> Any:
        service = self._service
        service.record("openai", "batches.cancel", batch_id=batch_id)
        return _openai_batch(service.cancel("openai", batch_id))


def _openai_batch(batch: FakeBatch) -> Any:
    from openai.types import Batch

    total = len(batch.entries)
    failed = sum(1 for entry in batch.entries if entry.outcome and entry.outcome.failed)
    if batch.cancelled:
        status = "cancelled"
    elif batch.completed:
        status = "completed"
    else:
        status = "validating" if batch.polls == 0 else "in_progress"
    done = batch.completed
    return Batch.model_validate(
        {
            "id": batch.batch_id,
            "object": "batch",
            "endpoint": batch.info.get("endpoint", "/v1/responses"),
            "input_file_id": batch.info.get("input_file_id", ""),
            "completion_window": batch.info.get("completion_window", "24h"),
            "created_at": CREATED_AT,
            "status": status,
            "completed_at": CREATED_AT + 60 if done else None,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "metadata": batch.info.get("metadata"),
            "request_counts": {
                "total": total,
                "completed": total - failed if done else 0,
                "failed": failed if done else 0,
            },
        }
    )


class _OpenAIModels:
    def __init__(self, service: FakeBatchService) -> None:
        self._service = service

    def list(self, **kwargs: Any) -> _Page:
        self._service.record("openai", "models.list")
        return _Page(data=[])


class FakeOpenAIClient:
    """Stand-in for ``openai.OpenAI``: files, batches and model listing."""

    def __init__(self, service: FakeBatchService) -> None:
        self.files = _OpenAIFiles(service)
        self.batches = _OpenAIBatches(service)
        self.models = _OpenAIModels(service)


class _AnthropicBatches:
    def __init__(self, service: FakeBatchService) -> None:
        self._service = service

    def create(self, *, requests: list[dict[str, Any]], **kwargs: Any) -> Any:
        service = self._service
        service.record("anthropic", "messages.batches.create", requests=len(requests))
        batch = service.submit(
            "anthropic",
            "msgbatch_fake",
            [(str(item["custom_id"]), dict(item["params"])) for item in requests],
        )
        return _anthropic_batch(batch)

    def retrieve(self, batch_id: str, **kwargs: Any) -> Any:
        service = self._service
        service.record("anthropic", "messages.batches.retrieve", batch_id=batch_id)
        return _anthropic_batch(service.poll(service.lookup(batch_id)))

    def results(self, batch_id: str, **kwargs: Any) -> Iterator[Any]:
        from anthropic.types.messages import MessageBatchIndividualResponse

        service = self._service
        service.record("anthropic", "messages.batches.results", batch_id=batch_id)
        batch = service.lookup(batch_id)
        if not batch.completed:
            raise FakeAPIError(400, f"Error code: 400 - batch {batch_id} not ended")
        responses = [
            MessageBatchIndividualResponse.model_validate(
                {
                    "custom_id": entry.custom_id,
                    "result": _anthropic_result(index, entry),
                }
            )
            for index, entry in enumerate(batch.entries, 1)
            if not entry.dropped
        ]
        return iter(responses)

    def list(
        self,
        *,
        limit: int | None = None,
        after_id: str | None = None,
        **kwargs: Any,
    ) -> _Page:
        service = self._service
        service.record(
            "anthropic", "messages.batches.list", limit=limit, after_id=after_id
        )
        batches = [_anthropic_batch(batch) for batch in service.listed("anthropic")]
        last = batches[-1].id if batches else None
        return _Page(data=batches, has_more=False, last_id=last)

    def cancel(self, batch_id: str, **kwargs: Any) -> Any:
        service = self._service
        service.record("anthropic", "messages.batches.cancel", batch_id=batch_id)
        return _anthropic_batch(service.cancel("anthropic", batch_id))


def _anthropic_result(index: int, entry: BatchEntry) -> dict[str, Any]:
    outcome = entry.outcome or Outcome(content="")
    if outcome.failed:
        return {
            "type": "errored",
            "error": {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": outcome.error},
                "request_id": f"req_fake{index:04d}",
            },
        }
    return {
        "type": "succeeded",
        "message": {
            "id": f"msg_fake{index:04d}",
            "type": "message",
            "role": "assistant",
            "model": str(entry.body.get("model", "")),
            "content": [{"type": "text", "text": outcome.content or ""}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": USAGE["input_tokens"],
                "output_tokens": USAGE["output_tokens"],
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    }


def _anthropic_batch(batch: FakeBatch) -> Any:
    from anthropic.types.messages import MessageBatch

    outcomes = [entry.outcome for entry in batch.entries]
    errored = sum(1 for outcome in outcomes if outcome and outcome.failed)
    total = len(batch.entries)
    ended = batch.completed or batch.cancelled
    created = "2026-01-01T00:00:00Z"
    return MessageBatch.model_validate(
        {
            "id": batch.batch_id,
            "type": "message_batch",
            "processing_status": "ended" if ended else "in_progress",
            "created_at": created,
            "expires_at": "2026-01-02T00:00:00Z",
            "ended_at": "2026-01-01T00:01:00Z" if ended else None,
            "archived_at": None,
            "cancel_initiated_at": None,
            "results_url": (
                f"https://fake-anthropic.invalid/v1/messages/batches/"
                f"{batch.batch_id}/results"
                if batch.completed
                else None
            ),
            "request_counts": {
                "processing": 0 if ended else total,
                "succeeded": total - errored if batch.completed else 0,
                "errored": errored if batch.completed else 0,
                "canceled": total if batch.cancelled else 0,
                "expired": 0,
            },
        }
    )


class _AnthropicMessages:
    def __init__(self, service: FakeBatchService) -> None:
        self.batches = _AnthropicBatches(service)


class FakeAnthropicClient:
    """Stand-in for ``anthropic.Anthropic``: the message batches API."""

    def __init__(self, service: FakeBatchService) -> None:
        self.messages = _AnthropicMessages(service)


class _GoogleBatches:
    def __init__(self, service: FakeBatchService) -> None:
        self._service = service

    def create(
        self, *, model: str, src: Any, config: dict[str, Any] | None = None
    ) -> Any:
        service = self._service
        recorded = dict(config or {})
        if isinstance(recorded.get("display_name"), str):
            recorded["display_name"] = _DISPLAY_TIME_RE.sub(
                "-<unix-time>", recorded["display_name"]
            )
        if isinstance(src, str):
            service.record(
                "google", "batches.create", model=model, src=src, config=recorded
            )
            raw = service.files.get(src)
            if raw is None:
                raise FakeAPIError(404, f"Error code: 404 - file {src} not found")
            lines = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
            batch = service.submit(
                "google",
                "batches/fake",
                [(str(line["key"]), dict(line["request"])) for line in lines if line],
                {
                    "model": model,
                    "display_name": recorded.get("display_name"),
                    "src_file": src,
                },
            )
            return _google_batch(batch)
        service.record(
            "google", "batches.create", model=model, src=len(src), config=recorded
        )
        requests: list[tuple[str, dict[str, Any]]] = []
        for index, item in enumerate(src):
            metadata = item.get("metadata") or {}
            key = metadata.get("key") if isinstance(metadata, Mapping) else None
            requests.append((str(key or f"inline-{index}"), dict(item)))
        batch = service.submit(
            "google",
            "batches/fake",
            requests,
            {"model": model, "display_name": recorded.get("display_name")},
        )
        return _google_batch(batch)

    def get(self, *, name: str, **kwargs: Any) -> Any:
        service = self._service
        service.record("google", "batches.get", name=name)
        return _google_batch(service.poll(service.lookup(name)))

    def list(self, *, config: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
        service = self._service
        service.record("google", "batches.list", config=dict(config or {}))
        return iter([_google_batch(batch) for batch in service.listed("google")])

    def cancel(self, *, name: str, **kwargs: Any) -> None:
        service = self._service
        service.record("google", "batches.cancel", name=name)
        service.cancel("google", name)


def _google_response(entry: BatchEntry) -> dict[str, Any]:
    outcome = entry.outcome or Outcome(content="")
    metadata = entry.body.get("metadata")
    echoed: dict[str, Any] = {"metadata": dict(metadata)} if metadata else {}
    if outcome.failed:
        return {
            **echoed,
            "error": {"code": outcome.status_code, "message": outcome.error},
        }
    return {
        **echoed,
        "response": {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": outcome.content}]},
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": USAGE["input_tokens"],
                "candidates_token_count": USAGE["output_tokens"],
                "total_token_count": USAGE["total_tokens"],
            },
        },
    }


def _google_rest_result(entry: BatchEntry) -> dict[str, Any]:
    """Return the result of one file request in REST (camel) keys."""
    outcome = entry.outcome or Outcome(content="")
    if outcome.failed:
        return {"error": {"code": outcome.status_code, "message": outcome.error}}
    return {
        "response": {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": outcome.content}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": USAGE["input_tokens"],
                "candidatesTokenCount": USAGE["output_tokens"],
                "totalTokenCount": USAGE["total_tokens"],
            },
        }
    }


def _google_batch(batch: FakeBatch) -> Any:
    from google.genai import types

    if batch.cancelled:
        state = "JOB_STATE_CANCELLED"
    elif batch.completed:
        state = "JOB_STATE_SUCCEEDED"
    else:
        state = "JOB_STATE_PENDING" if batch.polls == 0 else "JOB_STATE_RUNNING"
    data: dict[str, Any] = {
        "name": batch.batch_id,
        "display_name": batch.info.get("display_name"),
        "model": batch.info.get("model"),
        "state": state,
    }
    if batch.completed and batch.output_file_id:
        data["dest"] = {"file_name": batch.output_file_id}
    elif batch.completed:
        data["dest"] = {
            "inlined_responses": [
                _google_response(entry) for entry in batch.entries if not entry.dropped
            ]
        }
    return types.BatchJob.model_validate(data)


class _GoogleFiles:
    """The file API: JSONL request uploads and result downloads."""

    def __init__(self, service: FakeBatchService) -> None:
        self._service = service

    def upload(self, *, file: Any, config: Mapping[str, Any] | None = None) -> Any:
        raw = file.read() if hasattr(file, "read") else bytes(file)
        lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
        name = dict(config or {}).get("display_name")
        if isinstance(name, str):
            name = _DISPLAY_TIME_RE.sub("-<unix-time>", name)
        self._service.record(
            "google", "files.upload", display_name=name, lines=len(lines)
        )
        return SimpleNamespace(name=self._service.add_file("files/fake", raw))

    def download(self, *, file: str, **kwargs: Any) -> bytes:
        self._service.record("google", "files.download", file=file)
        content = self._service.files.get(file)
        if content is None:
            raise FakeAPIError(404, f"Error code: 404 - file {file} not found")
        return content

    def delete(self, *, name: str, **kwargs: Any) -> None:
        self._service.record("google", "files.delete", name=name)
        with self._service._lock:
            self._service.files.pop(name, None)


class FakeGoogleClient:
    """Stand-in for ``google.genai.Client``: the batches and file APIs."""

    def __init__(self, service: FakeBatchService) -> None:
        self.batches = _GoogleBatches(service)
        self.files = _GoogleFiles(service)


__all__ = [
    "CREATED_AT",
    "SDK_CLIENTS",
    "FakeAnthropicClient",
    "FakeBatch",
    "FakeBatchService",
    "FakeGoogleClient",
    "FakeOpenAIClient",
    "Outcome",
    "describe_body",
]
