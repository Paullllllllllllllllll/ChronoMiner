"""A stopped answer (content filter, refusal, truncation) ends a unit at once.

One rule in ``common`` for synchronous calls: :func:`structured_call` never
validates or retries a stopped answer, an SDK error raised for a cut answer
counts as a truncation, and the attempt record carries the outcome.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from chronominer.common.capture import (
    AttemptRecord,
    HttpAttempt,
    carry_prompt_feedback,
    use_recorder,
)
from chronominer.common.responses import (
    ResponseOutcome,
    inspect_response,
    truncation_of,
)
from chronominer.common.retry import ErrorKind, classify_error
from chronominer.common.structured import (
    PreparedRequest,
    StructuredRequest,
    structured_call,
)

SPEC: dict[str, Any] = {
    "name": "answer",
    "schema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}


class LengthFinishReasonError(Exception):
    """Stands in for the OpenAI SDK's parser error of a cut answer."""


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


async def _invoke(request: PreparedRequest, messages: Sequence[Any]) -> Any:
    return await request.runnable.ainvoke(list(messages), **request.kwargs)


def _call(model: _Model, response_format: str = "json") -> Any:
    request = StructuredRequest(
        "openai",
        response_format,  # type: ignore[arg-type]
        None if response_format == "text" else SPEC,
        validation_retries=3,
    )
    return asyncio.run(structured_call(model, [], request, invoke=_invoke))


@pytest.mark.parametrize(
    "metadata",
    [
        {"stop_reason": "max_tokens"},
        {"finish_reason": "MAX_TOKENS"},
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
    ],
)
@pytest.mark.parametrize("response_format", ["json", "text"])
def test_a_truncated_answer_is_never_retried_or_kept(
    metadata: dict[str, Any], response_format: str
) -> None:
    cut = AIMessage(content='{"text": "cu', response_metadata=metadata)
    model = _Model([cut, AIMessage(content='{"text": "whole"}')])

    result = _call(model, response_format)

    assert model.calls == 1
    assert result.stopped and not result.ok
    assert result.outcome.kind == "truncated"


def test_an_sdk_error_for_a_cut_answer_is_a_truncation() -> None:
    model = _Model([LengthFinishReasonError("length limit reached")])

    result = _call(model)

    assert model.calls == 1
    assert result.outcome == ResponseOutcome(
        "truncated", "LengthFinishReasonError: finish_reason=length"
    )
    assert result.text == ""


def test_a_cut_answer_error_is_not_retried_as_a_server_error() -> None:
    error = LengthFinishReasonError(
        "Could not parse response content as the length limit was reached - "
        "CompletionUsage(completion_tokens=64, cost_details={"
        "'upstream_inference_cost': 0.01})"
    )
    wrapped = RuntimeError("call failed")
    wrapped.__cause__ = error
    assert truncation_of(wrapped) is not None
    assert classify_error(error) is ErrorKind.CLIENT_ERROR
    assert classify_error(Exception("upstream_inference_cost: 0")) is not (
        ErrorKind.SERVER_ERROR
    )
    assert classify_error(Exception("Upstream error from Alibaba")) is (
        ErrorKind.SERVER_ERROR
    )


class _Rows:
    def __init__(self) -> None:
        self.records: list[AttemptRecord] = []

    async def record(self, record: AttemptRecord) -> int | None:
        self.records.append(record)
        return len(self.records)


def _store(**kwargs: Any) -> AttemptRecord:
    rows = _Rows()

    async def run() -> None:
        with use_recorder(rows):
            attempt = HttpAttempt(
                provider="google", model="m", role="r", attempt=1, label=None
            )
            with attempt.capturing():
                pass
            await attempt.store(**kwargs)

    asyncio.run(run())
    (record,) = rows.records
    return record


def test_a_stopped_response_is_stored_with_its_outcome() -> None:
    filtered = AIMessage(content="", response_metadata={"finish_reason": "RECITATION"})

    record = _store(response=filtered)

    assert record.error_type == "content_filter"
    assert record.error == (
        "Response stopped by the provider's content filter: finish_reason=RECITATION"
    )


def test_a_completed_response_is_stored_without_an_error() -> None:
    record = _store(response=AIMessage(content="fine"))
    assert (record.error, record.error_type) == (None, None)


def test_a_cut_answer_error_is_stored_as_a_truncation() -> None:
    record = _store(error=LengthFinishReasonError("length limit reached"))
    assert record.error_type == "truncated"


def test_a_prompt_gemini_blocked_is_read_from_the_body() -> None:
    message = AIMessage(content="")
    body = json.dumps({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}})

    carry_prompt_feedback(message, body.encode("utf-8"))

    assert inspect_response(message) == ResponseOutcome(
        "content_filter", "prompt_feedback.block_reason=PROHIBITED_CONTENT"
    )


def test_a_body_without_feedback_changes_nothing() -> None:
    message = AIMessage(content="", response_metadata={"finish_reason": "STOP"})
    carry_prompt_feedback(message, b'{"candidates": []}')
    assert "prompt_feedback" not in message.response_metadata
