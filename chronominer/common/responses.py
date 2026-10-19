"""How a model response ended, read from its provider status fields.

``inspect_response`` reads every shape a LangChain response arrives in: an
``AIMessage``, or the ``with_structured_output(include_raw=True)`` dict with
its ``raw`` message and ``parsing_error``. ``outcome_of`` classifies the
same status fields given as a dict, as the batch backends read them from a
result record. Content-filter stops, refusals and answers cut at the
output-token limit end a unit at once, in synchronous calls and batch
results alike, because a retry with the same request would get the same
answer; none of them is ever kept as a success.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

__all__ = [
    "ProviderResponseError",
    "ResponseKind",
    "ResponseOutcome",
    "body_error_of",
    "finish_error_of",
    "get_response_metadata",
    "inspect_response",
    "outcome_of",
    "response_error_of",
    "stop_of",
    "truncation_of",
    "unwrap_raw_response",
]

ResponseKind = Literal["ok", "content_filter", "refusal", "truncated"]
"""How a response ended; every kind but ``ok`` fails the unit and is stored
as its ``error_type``."""

_GEMINI_FILTERS = frozenset(
    {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}
)
"""Gemini ``finish_reason`` values of a response its filters stopped."""

_UNSPECIFIED_BLOCKS = frozenset(
    {0, "", "BLOCKED_REASON_UNSPECIFIED", "BLOCK_REASON_UNSPECIFIED"}
)
"""Gemini ``block_reason`` values that block nothing."""

_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "content_filter": "Response stopped by the provider's content filter",
        "refusal": "Model refused the request",
        "truncated": "Response truncated at the output-token limit",
        "ok": "Response completed",
    }
)


@dataclass(frozen=True)
class ResponseOutcome:
    """The kind of ending and, when known, the reason the provider gave.

    ``content_filter``, ``refusal`` and ``truncated`` stop the unit; ``ok``
    leaves the response to the caller's empty-content and schema checks.
    """

    kind: ResponseKind
    reason: str | None = None

    @property
    def stops_page(self) -> bool:
        """Whether this outcome fails the unit without further retries.

        A truncated answer is never kept, not even as plain text: a retry
        with the same output limit would be cut again.
        """
        return self.kind != "ok"

    @property
    def message(self) -> str:
        """A readable error message including the reason."""
        label = _LABELS[self.kind]
        return f"{label}: {self.reason}" if self.reason else label


def unwrap_raw_response(response: Any) -> Any:
    """Return the raw message inside a ``with_structured_output`` wrapper dict."""
    if isinstance(response, dict) and "raw" in response:
        return response["raw"]
    return response


def get_response_metadata(response: Any) -> dict[str, Any]:
    """Return the ``response_metadata`` dict of a possibly wrapped response.

    Returns ``{}`` when the response carries no metadata dict.
    """
    meta = getattr(unwrap_raw_response(response), "response_metadata", None)
    return meta if isinstance(meta, dict) else {}


def _refusal_text(target: Any) -> str | None:
    """Return the refusal text of a message, ``""`` for an empty one, else None.

    Chat Completions put it in ``additional_kwargs["refusal"]``; the Responses
    API puts it in a content block of type ``refusal``.
    """
    kwargs = getattr(target, "additional_kwargs", None)
    if isinstance(kwargs, dict):
        refusal = kwargs.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            return refusal.strip()
    content = getattr(target, "content", None)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "refusal":
                text = block.get("refusal")
                return text.strip() if isinstance(text, str) else ""
    return None


def _incomplete_reason(meta: Mapping[str, Any]) -> str | None:
    """Return ``incomplete_details.reason`` of an incomplete Responses API call."""
    if meta.get("status") != "incomplete":
        return None
    details = meta.get("incomplete_details")
    if isinstance(details, dict) and isinstance(details.get("reason"), str):
        return str(details["reason"])
    return None


def _block_reason(meta: Mapping[str, Any]) -> str | None:
    """Return the Gemini ``prompt_feedback.block_reason`` of a blocked prompt.

    Snake and camel keys are read; an unspecified reason blocks nothing.
    """
    feedback = meta.get("prompt_feedback")
    if feedback is None:
        feedback = meta.get("promptFeedback")
    if not isinstance(feedback, Mapping):
        return None
    reason = feedback.get("block_reason")
    if reason is None:
        reason = feedback.get("blockReason")
    reason = getattr(reason, "value", reason)
    if isinstance(reason, bool) or not isinstance(reason, str | int):
        return None
    if reason in _UNSPECIFIED_BLOCKS:
        return None
    return str(reason)


def _truncation_reason(meta: Mapping[str, Any], incomplete: str | None) -> str | None:
    """Return the field that reports a stop at the output-token limit."""
    if incomplete == "max_output_tokens":
        return "incomplete_details.reason=max_output_tokens"
    if meta.get("stop_reason") == "max_tokens":
        return "stop_reason=max_tokens"
    if meta.get("finish_reason") in ("MAX_TOKENS", "length"):
        return f"finish_reason={meta['finish_reason']}"
    return None


def outcome_of(meta: Mapping[str, Any], refusal: str | None = None) -> ResponseOutcome:
    """Classify a response from its status fields and its refusal text.

    *meta* holds the fields as ``response_metadata`` names them: the
    Responses API ``status`` and ``incomplete_details``, the Chat
    Completions and Gemini ``finish_reason``, Gemini's ``prompt_feedback``
    and the Anthropic ``stop_reason``. *refusal* is the refusal text the
    response carries (``""`` for an empty one), else None. A content filter
    (Gemini's ``SAFETY``, ``RECITATION``, ``PROHIBITED_CONTENT``,
    ``BLOCKLIST`` and ``SPII`` included, and a prompt Gemini blocked
    outright) wins over a refusal, and a refusal over truncation (OpenAI's
    ``incomplete_details.reason`` ``max_output_tokens``, Anthropic's
    ``stop_reason`` ``max_tokens``, Gemini's ``finish_reason``
    ``MAX_TOKENS`` or the Chat Completions ``length``).
    """
    incomplete = _incomplete_reason(meta)
    finish = meta.get("finish_reason")
    if incomplete == "content_filter":
        return ResponseOutcome(
            "content_filter", "incomplete_details.reason=content_filter"
        )
    blocked = _block_reason(meta)
    if blocked is not None:
        return ResponseOutcome(
            "content_filter", f"prompt_feedback.block_reason={blocked}"
        )
    if isinstance(finish, str) and (
        finish == "content_filter" or finish in _GEMINI_FILTERS
    ):
        return ResponseOutcome("content_filter", f"finish_reason={finish}")
    if meta.get("stop_reason") == "refusal":
        return ResponseOutcome("refusal", "stop_reason=refusal")
    if refusal is not None:
        return ResponseOutcome("refusal", refusal or None)
    truncation = _truncation_reason(meta, incomplete)
    if truncation is not None:
        return ResponseOutcome("truncated", truncation)
    return ResponseOutcome("ok")


_TRUNCATION_ERRORS = frozenset({"LengthFinishReasonError"})
"""SDK errors raised instead of a response cut at the output-token limit
(the OpenAI SDK's structured-output parser, which OpenRouter calls use)."""


def truncation_of(exc: BaseException) -> ResponseOutcome | None:
    """Return the ``truncated`` outcome an SDK error stands for, else None.

    The error or one of its causes must be a parser error raised for an
    answer cut at the output-token limit; such an error is no transient
    failure, and a retry with the same limit would be cut again.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__
        if name in _TRUNCATION_ERRORS:
            return ResponseOutcome("truncated", f"{name}: finish_reason=length")
        current = current.__cause__ or current.__context__
    return None


_FILTER_WORDING = re.compile(
    r"content[ _-]?filter|moderation|flagged|safety|prohibited|inappropriate",
    re.IGNORECASE,
)
"""Wording of an in-body error that reports a content filter."""
_NULL_CHOICES = "null value for 'choices'"
"""The LangChain error for a Chat Completions body without choices."""


class ProviderResponseError(RuntimeError):
    """A provider answered HTTP 200 with an error instead of an answer.

    OpenRouter and other Chat Completions services put an ``error`` object
    in the body, or end a choice with ``finish_reason`` ``error``.
    *status_code* is the code the error names, else 502, so the retry ladder
    treats it as a server error; *outcome* is a ``content_filter`` outcome
    when the error's wording reports a filter, which ends the unit instead.
    *body* is the response body and *usage* the usage the response reported,
    so a billed attempt keeps its tokens
    (:func:`~.usage.usage_from_exception` reads both).
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        outcome: ResponseOutcome | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.outcome = outcome
        self.body: Mapping[str, Any] | None = None
        self.usage: Any = None


def _body_error(message: str, code: Any) -> ProviderResponseError:
    status = 502
    if isinstance(code, int) and not isinstance(code, bool) and 400 <= code <= 599:
        status = code
    outcome = (
        ResponseOutcome("content_filter", message)
        if _FILTER_WORDING.search(message)
        else None
    )
    return ProviderResponseError(message, status_code=status, outcome=outcome)


def response_error_of(exc: BaseException) -> ProviderResponseError | None:
    """Return the in-body provider error an SDK exception stands for, else None.

    LangChain raises ``ValueError`` with the body's ``error`` object and
    ``TypeError`` for a body whose ``choices`` are null.
    """
    if isinstance(exc, ProviderResponseError):
        return exc
    if isinstance(exc, ValueError) and exc.args and isinstance(exc.args[0], Mapping):
        body = exc.args[0]
        if "message" in body or "code" in body:
            message = str(body.get("message") or body)
            metadata = body.get("metadata")
            if isinstance(metadata, Mapping) and metadata.get("raw"):
                message = f"{message} ({metadata['raw']})"
            return _body_error(message, body.get("code"))
    if isinstance(exc, TypeError) and _NULL_CHOICES in str(exc):
        return _body_error("Response without choices", None)
    return None


def body_error_of(body: bytes | str | None) -> ProviderResponseError | None:
    """Return the error a Chat Completions response body carries, else None.

    Reads a captured HTTP-200 body: an ``error`` object, or a choice that
    ended with ``finish_reason`` ``error``, whose own ``error`` object names
    the message and code where it has one. The SDK's own parser may fail on
    such a body before LangChain sees its error.
    """
    data = _json_body(body)
    if data is None:
        return None
    error = data.get("error")
    found: ProviderResponseError | None = None
    if isinstance(error, Mapping):
        found = _mapping_error(error)
    elif (cause := _choice_error(data)) is not None:
        found = (
            _mapping_error(cause)
            if isinstance(cause, Mapping)
            else _body_error("finish_reason=error", None)
        )
    elif data.get("choices") is None and "choices" in data:
        found = _body_error("Response without choices", None)
    if found is not None:
        found.body = data
    return found


def _json_body(body: bytes | str | None) -> Mapping[str, Any] | None:
    """Return a captured body parsed as a JSON object, else None."""
    if not body:
        return None
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, Mapping) else None


def _choice_error(data: Mapping[str, Any]) -> Any:
    """Return the ``error`` of the first choice that ended with
    ``finish_reason`` ``error`` (``""`` when it has none), else None."""
    choices = data.get("choices")
    for choice in choices if isinstance(choices, list) else ():
        if isinstance(choice, Mapping) and choice.get("finish_reason") == "error":
            return choice.get("error") or ""
    return None


def _mapping_error(error: Mapping[str, Any]) -> ProviderResponseError:
    return _body_error(str(error.get("message") or error), error.get("code"))


def finish_error_of(
    response: Any, body: bytes | str | None = None
) -> ProviderResponseError | None:
    """Return the error of a response whose choice ended with ``finish_reason``
    ``error``, else None.

    The SDK keeps no error object of the choice in the response metadata;
    *body*, the captured HTTP body, supplies it where it has one.
    """
    meta = get_response_metadata(response)
    if meta.get("finish_reason") != "error":
        return None
    error = meta.get("error")
    if not isinstance(error, Mapping):
        data = _json_body(body)
        cause = _choice_error(data) if data is not None else None
        if isinstance(cause, Mapping):
            found = _mapping_error(cause)
            found.body = data
            return found
    if isinstance(error, Mapping):
        return _mapping_error(error)
    native = meta.get("native_finish_reason")
    reason = f"finish_reason=error ({native})" if native else "finish_reason=error"
    return _body_error(str(error) if error else reason, None)


def stop_of(exc: BaseException) -> ResponseOutcome | None:
    """Return the outcome that ends a unit an SDK error stands for, else None:
    a truncation (:func:`truncation_of`) or an in-body content filter."""
    truncated = truncation_of(exc)
    if truncated is not None:
        return truncated
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ProviderResponseError) and current.outcome is not None:
            return current.outcome
        current = current.__cause__ or current.__context__
    return None


def inspect_response(response: Any) -> ResponseOutcome:
    """Classify how a LangChain model response ended.

    Reads the status fields of :func:`outcome_of` from ``response_metadata``,
    and as the refusal text a ``parsing_error`` of class
    ``OpenAIRefusalError`` in an ``include_raw`` dict, else
    ``additional_kwargs["refusal"]`` or a refusal content block. Anything
    else is ``ok``.
    """
    refusal: str | None = None
    if isinstance(response, dict) and "raw" in response:
        parsing_error = response.get("parsing_error")
        if (
            parsing_error is not None
            and type(parsing_error).__name__ == "OpenAIRefusalError"
        ):
            refusal = str(parsing_error).strip()
    if refusal is None:
        refusal = _refusal_text(unwrap_raw_response(response))
    return outcome_of(get_response_metadata(response), refusal)
