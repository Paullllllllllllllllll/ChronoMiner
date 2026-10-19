"""In-body provider errors, answers without a JSON object and batch counts.

One rule in ``common``: an HTTP-200 body with an ``error`` object, a body
without choices and a choice that ended with ``finish_reason`` ``error`` are
provider errors the retry ladder retries, unless their wording reports a
content filter, which ends the unit; an answer in a JSON format that holds
no JSON object is never usable; a finalized submission counts its units.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from chronominer.common.batch import BatchStatus, BatchStatusInfo, outcome_counts
from chronominer.common.capture import CapturedResponse, capture_attempt
from chronominer.common.responses import (
    ProviderResponseError,
    body_error_of,
    finish_error_of,
    response_error_of,
    stop_of,
)
from chronominer.common.retry import ErrorKind, classify_error
from chronominer.common.structured import (
    PreparedRequest,
    StructuredRequest,
    default_invoke,
    structured_call,
)
from chronominer.common.usage import usage_from_exception

SPEC: dict[str, Any] = {
    "name": "answer",
    "schema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}


class _Model:
    """Answers each call with the next scripted response or raises it."""

    def __init__(self, answers: Sequence[Any]) -> None:
        self.answers = list(answers)
        self.calls = 0

    async def ainvoke(self, messages: Sequence[Any], **kwargs: Any) -> Any:
        self.calls += 1
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _prepared(model: _Model) -> PreparedRequest:
    return PreparedRequest(model, {}, False)


def _invoke(model: _Model) -> Any:
    return asyncio.run(default_invoke(_prepared(model), []))


def test_an_error_body_is_a_retryable_provider_error() -> None:
    body = {"message": "Provider returned error", "code": 502}
    model = _Model([ValueError(body)])

    with pytest.raises(ProviderResponseError) as caught:
        _invoke(model)

    assert caught.value.status_code == 502
    assert classify_error(caught.value) is ErrorKind.SERVER_ERROR
    assert stop_of(caught.value) is None


def test_an_error_body_keeps_its_rate_limit_code() -> None:
    error = response_error_of(ValueError({"message": "Rate limited", "code": 429}))

    assert error is not None
    assert classify_error(error) is ErrorKind.RATE_LIMIT


def test_a_body_without_choices_is_a_provider_error() -> None:
    exc = TypeError("Received response with null value for 'choices'. ...")

    error = response_error_of(exc)

    assert error is not None
    assert classify_error(error) is ErrorKind.SERVER_ERROR


def test_a_choice_ending_in_error_raises_a_provider_error() -> None:
    message = AIMessage(content="", response_metadata={"finish_reason": "error"})

    with pytest.raises(ProviderResponseError):
        _invoke(_Model([message]))
    assert finish_error_of(AIMessage(content="ok")) is None


def test_a_choice_ending_in_error_keeps_its_usage() -> None:
    message = AIMessage(
        content="",
        response_metadata={"finish_reason": "error"},
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )

    with pytest.raises(ProviderResponseError) as caught:
        _invoke(_Model([message]))

    usage = usage_from_exception(caught.value)
    assert usage is not None
    assert usage.total_tokens == 12


def test_a_captured_error_body_names_the_sdk_parser_failure() -> None:
    body = b'{"error": {"message": "Upstream timeout", "code": 504}, "choices": null}'
    model = _Model([TypeError("'NoneType' object is not iterable")])

    with capture_attempt() as scope:
        scope.add(CapturedResponse(200, "POST", "https://x/v1", body, None))
        with pytest.raises(ProviderResponseError) as caught:
            _invoke(model)

    assert caught.value.status_code == 504
    assert classify_error(caught.value) is ErrorKind.SERVER_ERROR
    assert body_error_of(b'{"choices": [{"finish_reason": "error"}]}') is not None
    assert body_error_of(b'{"choices": [{"finish_reason": "stop"}]}') is None


def test_filter_wording_ends_the_unit_as_a_content_filter() -> None:
    body = {"message": "Input was flagged by moderation", "code": 403}
    model = _Model([ValueError(body)])

    async def call() -> Any:
        request = StructuredRequest("openrouter", "json", SPEC)
        return await structured_call(model, [], request)

    result = asyncio.run(call())

    assert model.calls == 1
    assert result.stopped
    assert result.outcome.kind == "content_filter"
    error = response_error_of(ValueError(body))
    assert error is not None
    assert classify_error(error) is ErrorKind.CLIENT_ERROR


def test_a_choice_error_with_filter_wording_ends_the_unit() -> None:
    body = (
        b'\n  \n{"choices": [{"finish_reason": "error", "error": {"code": 502, '
        b'"message": "Upstream error from Alibaba: Output data may contain '
        b'inappropriate content."}, "message": {"content": null}}]}'
    )
    message = AIMessage(content="", response_metadata={"finish_reason": "error"})
    model = _Model([message])

    async def call() -> Any:
        request = StructuredRequest("openrouter", "json", SPEC)
        with capture_attempt() as scope:
            scope.add(CapturedResponse(200, "POST", "https://x/v1", body, None))
            return await structured_call(model, [], request)

    result = asyncio.run(call())

    assert model.calls == 1
    assert result.stopped
    assert result.outcome.kind == "content_filter"
    assert "inappropriate content" in str(result.outcome.reason)


def test_a_choice_without_its_own_error_keeps_the_native_reason() -> None:
    message = AIMessage(
        content="",
        response_metadata={"finish_reason": "error", "native_finish_reason": "x"},
    )
    body = b'{"choices": [{"finish_reason": "error", "message": {"content": null}}]}'

    error = finish_error_of(message, body)

    assert error is not None
    assert str(error) == "finish_reason=error (x)"


def test_other_value_errors_pass_through() -> None:
    with pytest.raises(ValueError, match="plain"):
        _invoke(_Model([ValueError("plain")]))


@pytest.mark.parametrize("answer", ["42", '"just a string"', "No JSON here."])
@pytest.mark.parametrize("response_format", ["schema", "json"])
def test_an_answer_without_an_object_is_never_usable(
    answer: str, response_format: str
) -> None:
    model = _Model([AIMessage(content=answer)])

    async def call() -> Any:
        request = StructuredRequest(
            "openai",
            response_format,  # type: ignore[arg-type]
            SPEC,
        )
        return await structured_call(model, [], request)

    result = asyncio.run(call())

    assert result.not_object
    assert not result.ok


def test_a_text_answer_or_an_object_is_not_flagged() -> None:
    async def call(fmt: str, text: str) -> Any:
        request = StructuredRequest(
            "openai",
            fmt,  # type: ignore[arg-type]
            None if fmt == "text" else SPEC,
        )
        return await structured_call(_Model([AIMessage(content=text)]), [], request)

    assert not asyncio.run(call("text", "42")).not_object
    assert not asyncio.run(call("json", '{"other": 1}')).not_object


def test_a_finalized_submission_counts_its_units() -> None:
    assert outcome_counts(3, 1) == {
        "total": 3,
        "completed": 2,
        "failed": 1,
        "pending": 0,
    }
    assert outcome_counts(2, 5)["completed"] == 0


def test_status_counts_are_the_provider_counts() -> None:
    info = BatchStatusInfo(
        BatchStatus.IN_PROGRESS,
        total_requests=4,
        completed_requests=1,
        failed_requests=1,
        pending_requests=2,
    )

    assert info.counts() == {"total": 4, "completed": 1, "failed": 1, "pending": 2}
