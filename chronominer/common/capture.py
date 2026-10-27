"""Verbatim provider requests and responses captured at the HTTP transport.

A pipeline opens one :func:`capture_attempt` scope per provider attempt. The
hooks installed on every httpx client hand each request body, as it goes over
the wire, and each response body, as it came over the wire, to the scope of
the current attempt, which a context variable holds; outside a scope nothing
is captured. Scopes stay separate per asyncio task and follow
``asyncio.to_thread``, which copies the context.

The response hooks read the whole body before the SDK parses it, so a
streamed response inside a scope arrives buffered. A streamed request body,
such as a multipart file upload, is not captured.

The module also defines the attempt record a pipeline builds from a scope,
the recorder protocol through which the record is stored, and
:class:`HttpAttempt`, which captures one synchronous call and stores its
record; a pipeline installs its recorder with :func:`use_recorder` for the
calls it makes.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import threading
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import httpx

from .responses import inspect_response, stop_of, unwrap_raw_response

if TYPE_CHECKING:
    from .usage import Usage

RawKind = Literal["http", "sdk"]

_PAYLOAD_MIN_CHARS = 1024
_MEDIA_TYPE_KEYS = frozenset({"media_type", "mime_type", "mimeType"})
"""Keys that mark a mapping as a media part whose ``data`` is a payload."""


@dataclass(frozen=True)
class CapturedResponse:
    """One HTTP response seen inside a capture scope."""

    status: int
    method: str
    url: str
    body: bytes
    elapsed: float | None

    @property
    def text(self) -> str:
        """Return the body decoded as UTF-8 (invalid bytes replaced)."""
        return self.body.decode("utf-8", errors="replace")


class CaptureScope:
    """Hold the responses captured during one provider attempt.

    The responses are kept in arrival order, with the body of the last request
    sent.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._responses: list[CapturedResponse] = []
        self._request: bytes | None = None

    def add(self, response: CapturedResponse) -> None:
        """Append *response*; safe from any thread."""
        with self._lock:
            self._responses.append(response)

    def sent(self, body: bytes) -> None:
        """Keep *body* as the last request body sent; safe from any thread."""
        with self._lock:
            self._request = body

    @property
    def request(self) -> bytes | None:
        """Return the body of the last request sent, or None."""
        with self._lock:
            return self._request

    @property
    def responses(self) -> list[CapturedResponse]:
        """Return a copy of the captured responses."""
        with self._lock:
            return list(self._responses)

    @property
    def last(self) -> CapturedResponse | None:
        """Return the latest captured response, or None."""
        with self._lock:
            return self._responses[-1] if self._responses else None


_SCOPE: ContextVar[CaptureScope | None] = ContextVar("capture_scope", default=None)


def current_scope() -> CaptureScope | None:
    """Return the capture scope of the current context, or None."""
    return _SCOPE.get()


@contextlib.contextmanager
def capture_attempt() -> Iterator[CaptureScope]:
    """Open a capture scope for one provider attempt."""
    scope = CaptureScope()
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def _elapsed(response: httpx.Response) -> float | None:
    try:
        return response.elapsed.total_seconds()
    except RuntimeError:
        return None


def _store(scope: CaptureScope, response: httpx.Response) -> None:
    scope.add(
        CapturedResponse(
            status=response.status_code,
            method=response.request.method,
            url=str(response.request.url),
            body=response.content,
            elapsed=_elapsed(response),
        )
    )


def capture_response(response: httpx.Response) -> None:
    """Response hook for ``httpx.Client``: read the body into the current scope."""
    scope = _SCOPE.get()
    if scope is None:
        return
    response.read()
    _store(scope, response)


async def acapture_response(response: httpx.Response) -> None:
    """Response hook for ``httpx.AsyncClient``: read the body into the scope."""
    scope = _SCOPE.get()
    if scope is None:
        return
    await response.aread()
    _store(scope, response)


def capture_request(request: httpx.Request) -> None:
    """Request hook, and transport call: keep the body in the current scope.

    A streamed body, not yet read, is left alone.
    """
    scope = _SCOPE.get()
    if scope is None:
        return
    try:
        body = request.content
    except httpx.RequestNotRead:
        return
    scope.sent(body)


async def acapture_request(request: httpx.Request) -> None:
    """Request hook for ``httpx.AsyncClient``: see :func:`capture_request`."""
    capture_request(request)


def capture_hooks() -> dict[str, list[Callable[..., None]]]:
    """Return ``event_hooks`` for an ``httpx.Client`` that capture the traffic."""
    return {"request": [capture_request], "response": [capture_response]}


def async_capture_hooks() -> dict[str, list[Callable[..., Awaitable[None]]]]:
    """Return ``event_hooks`` for an ``httpx.AsyncClient`` that capture it."""
    return {"request": [acapture_request], "response": [acapture_response]}


def install_capture(client: httpx.Client | httpx.AsyncClient) -> None:
    """Add the matching capture hooks to *client* once."""
    wanted: dict[str, list[Any]] = (
        dict(async_capture_hooks())
        if isinstance(client, httpx.AsyncClient)
        else dict(capture_hooks())
    )
    hooks = {event: list(found) for event, found in client.event_hooks.items()}
    for event, added in wanted.items():
        present = hooks.setdefault(event, [])
        present.extend(hook for hook in added if hook not in present)
    client.event_hooks = hooks


@dataclass(frozen=True)
class AttemptRecord:
    """One provider attempt as stored: request, verbatim response and outcome.

    ``request`` holds the request as sent, without image or audio payloads
    (see :func:`sent_request`); None when no request was seen. ``raw`` is
    the captured body as bytes, unaltered, with ``raw_kind`` ``'http'``, or a
    dump of the SDK object with ``'sdk'`` where no hook could see the body.
    ``latency`` is in seconds.
    """

    provider: str
    model: str
    role: str
    attempt: int
    label: str | None = None
    request: Mapping[str, Any] | None = None
    raw: str | bytes | None = None
    raw_kind: RawKind | None = None
    usage: Mapping[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None
    http_status: int | None = None
    latency: float | None = None


class AttemptRecorder(Protocol):
    """Stores attempt records; returns the stored response id, or None."""

    async def record(self, record: AttemptRecord) -> int | None:
        """Store *record*; return the stored response id, or None."""
        ...


_RECORDER: ContextVar[AttemptRecorder | None] = ContextVar(
    "attempt_recorder", default=None
)


def current_recorder() -> AttemptRecorder | None:
    """Return the recorder installed for the current context, or None."""
    return _RECORDER.get()


@contextlib.contextmanager
def use_recorder(recorder: AttemptRecorder | None) -> Iterator[None]:
    """Install *recorder* for the calls made inside the block."""
    token = _RECORDER.set(recorder)
    try:
        yield
    finally:
        _RECORDER.reset(token)


async def record_attempt(record: AttemptRecord) -> int | None:
    """Store *record* through the current recorder; None without one."""
    recorder = _RECORDER.get()
    if recorder is None:
        return None
    return await recorder.record(record)


def sdk_dump(obj: Any) -> str:
    """Return a JSON dump of an SDK response object."""
    dump = getattr(obj, "model_dump_json", None)
    if callable(dump):
        return str(dump())
    to_json = getattr(obj, "to_json", None)
    if callable(to_json):
        return str(to_json())
    return json.dumps(obj, ensure_ascii=False, default=str)


_PLACEHOLDER = "<omitted "
"""The start of every placeholder :func:`redact_payloads` writes."""


def _is_base64(text: str) -> bool:
    if len(text) < _PAYLOAD_MIN_CHARS:
        return False
    for altchars in (None, b"-_"):
        try:
            base64.b64decode(text, altchars=altchars, validate=True)
        except (binascii.Error, ValueError):
            continue
        return True
    return False


def redact_payloads(value: Any) -> Any:
    """Return *value* with binary payloads replaced by a short placeholder.

    Data URLs, base64 strings (standard or URL-safe alphabet) of at least
    1,024 characters, bytes, and the ``data`` string of a media part (a
    mapping that names its media type, as Anthropic's and Google's image
    and audio parts do) of any length are replaced, recursively through
    mappings, lists and tuples. A placeholder stays as it is, so redacting
    twice keeps the original length.
    """
    if isinstance(value, bytes | bytearray):
        return f"<omitted {len(value)} bytes>"
    if isinstance(value, str) and value.startswith(_PLACEHOLDER):
        return value
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value[:256]:
            head = value.split(",", 1)[0]
            return f"<omitted {head}, {len(value)} chars>"
        if _is_base64(value):
            return f"<omitted base64, {len(value)} chars>"
        return value
    if isinstance(value, Mapping):
        media = not _MEDIA_TYPE_KEYS.isdisjoint(value)
        return {
            key: (
                f"<omitted base64, {len(item)} chars>"
                if media
                and key == "data"
                and isinstance(item, str)
                and not item.startswith(_PLACEHOLDER)
                else redact_payloads(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact_payloads(item) for item in value]
    return value


def sent_request(
    scope: CaptureScope | None,
    arguments: Callable[[], Mapping[str, Any]] | None = None,
) -> Mapping[str, Any] | None:
    """Return the request of an attempt as sent, without binary payloads.

    It is the JSON body of the last request *scope* saw go over HTTP, after
    every client-side rewrite. Where no JSON body was seen, *arguments*
    returns the arguments the pipeline passed to the SDK client, if given;
    else the result is None.
    """
    body = scope.request if scope is not None else None
    if body is not None:
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            return dict(redact_payloads(parsed))
    if arguments is None:
        return None
    return dict(redact_payloads(arguments()))


def carry_prompt_feedback(response: Any, body: bytes | str | None) -> None:
    """Copy a Gemini prompt block from the HTTP *body* onto *response*.

    langchain-google-genai keeps ``prompt_feedback`` in the result's
    ``llm_output`` and hands back an empty message without it when Gemini
    blocks the prompt, so :func:`~.responses.inspect_response` would read an
    empty answer. A ``promptFeedback`` of the body is set as the message's
    ``response_metadata["prompt_feedback"]`` unless the message carries one.
    """
    message = unwrap_raw_response(response)
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict) or body is None:
        return
    if metadata.get("prompt_feedback") is not None:
        return
    try:
        parsed = json.loads(body)
    except ValueError:
        return
    if not isinstance(parsed, dict):
        return
    feedback = parsed.get("promptFeedback", parsed.get("prompt_feedback"))
    if isinstance(feedback, Mapping):
        metadata["prompt_feedback"] = dict(feedback)


class HttpAttempt:
    """The capture scope and the stored record of one HTTP attempt.

    Without a recorder in the context it captures and stores nothing. The
    record's request is the JSON body sent over HTTP; where none was seen,
    *request* returns the arguments passed to the SDK client (see
    :func:`sent_request`). A response that
    :func:`~.responses.inspect_response` finds stopped (a content filter, a
    refusal, a truncated answer) is stored with the outcome's message as its
    error and the outcome's kind as its error type.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        role: str,
        attempt: int,
        label: str | None,
        request: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.active = current_recorder() is not None
        self._identity = (provider, model, role, attempt, label)
        self._request = request
        self._scope: CaptureScope | None = None
        self._started = time.monotonic()

    @contextlib.contextmanager
    def capturing(self) -> Iterator[None]:
        """Capture the responses of the block into this attempt's scope."""
        self._started = time.monotonic()
        if not self.active:
            yield
            return
        with capture_attempt() as scope:
            self._scope = scope
            yield

    async def store(
        self,
        *,
        response: Any = None,
        usage: Usage | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Build the attempt's record and await the recorder with it.

        The raw body is the last response captured; without one, the SDK
        object *response* is dumped and marked ``sdk``.
        """
        if not self.active:
            return
        latency = time.monotonic() - self._started
        captured = self._scope.last if self._scope is not None else None
        raw: str | bytes | None = None
        raw_kind: RawKind | None = None
        status = getattr(error, "status_code", None)
        if captured is not None:
            raw, raw_kind, status = captured.body, "http", captured.status
        elif response is not None:
            raw, raw_kind = sdk_dump(unwrap_raw_response(response)), "sdk"
        message = None if error is None else str(error)
        kind = None if error is None else type(error).__name__
        stopped = None if error is None else stop_of(error)
        if stopped is not None:
            kind = stopped.kind
        if response is not None and error is None:
            if captured is not None:
                carry_prompt_feedback(response, captured.body)
            outcome = inspect_response(response)
            if outcome.stops_page:
                message, kind = outcome.message, outcome.kind
        provider, model, role, attempt, label = self._identity
        await record_attempt(
            AttemptRecord(
                provider=provider,
                model=model,
                role=role,
                attempt=attempt,
                label=label,
                request=sent_request(self._scope, self._request),
                raw=raw,
                raw_kind=raw_kind,
                usage=None if usage is None else usage.as_dict(),
                error=message,
                error_type=kind,
                http_status=status if isinstance(status, int) else None,
                latency=latency,
            )
        )
