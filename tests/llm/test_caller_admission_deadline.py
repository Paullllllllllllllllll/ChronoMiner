"""ModelCaller: entry waits and WAIT pauses do not count against the page deadline;
requests and RETRY backoff do."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

import chronominer.llm.caller as caller_module
from chronominer import ext
from chronominer.common.retry import Decision, ErrorDecision, RetryPolicy
from chronominer.common.structured import StructuredRequest, StructuredResult
from chronominer.llm.caller import ModelCaller, PageTimeoutError
from tests.llm import helpers
from tests.llm.helpers import StatusError, model_spec

MESSAGES = [SystemMessage(content="Transcribe."), HumanMessage(content="page")]
# The failing requests answer with a client error, which leaves the rate
# limiter without an error penalty, so only the delay under test can matter.
DEADLINE = 0.2
LONG = 0.5


def _caller() -> ModelCaller:
    caller = ModelCaller("extract", model_spec(), helpers.env())
    caller.deadline = DEADLINE
    return caller


def _call(caller: ModelCaller) -> StructuredResult:
    request = StructuredRequest(
        provider="openai",
        response_format="json",
        schema={"name": "s", "schema": {"type": "object", "required": ["a"]}},
    )
    return asyncio.run(caller.call(MESSAGES, request, label="p1"))


def _patch_attempts(
    monkeypatch: pytest.MonkeyPatch,
    *,
    entry_wait: float = 0.0,
    decisions: list[ErrorDecision] | None = None,
) -> None:
    """Make entered attempts wait before admission and decide from *decisions*."""
    real = ext.attempt
    pending = list(decisions or [])

    def attempt(
        role: str, provider: str, estimate: int | None = None, **kwargs: Any
    ) -> ext.Attempt[Any]:
        opened: ext.Attempt[Any] = real(role, provider, estimate, **kwargs)
        refresh, on_error = opened.refresh, opened.on_error

        async def waiting_refresh() -> Any:
            await asyncio.sleep(entry_wait)
            return await refresh()

        def deciding(exc: BaseException) -> Any:
            decision = on_error(exc)
            return pending.pop(0) if pending else decision

        opened.refresh = waiting_refresh  # type: ignore[method-assign]
        opened.on_error = deciding  # type: ignore[method-assign]
        return opened

    monkeypatch.setattr(ext, "attempt", attempt)


def test_an_entry_wait_past_the_deadline_does_not_end_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_attempts(monkeypatch, entry_wait=LONG)
    helpers.install(monkeypatch, [{"a": 1}])

    result = _call(_caller())

    assert result.ok and result.data == {"a": 1}


def test_a_wait_decision_past_the_deadline_does_not_end_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_attempts(monkeypatch, decisions=[ErrorDecision(Decision.WAIT, LONG)])
    script = helpers.install(monkeypatch, [StatusError(400), {"a": 1}])

    result = _call(_caller())

    assert result.ok and result.data == {"a": 1}
    assert len(script.requests) == 2


def test_a_request_past_the_deadline_still_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helpers.install(monkeypatch, [{"a": 1}])
    real = caller_module.default_invoke

    async def hanging(model: Any, messages: Any) -> Any:
        await asyncio.sleep(LONG)
        return await real(model, messages)

    monkeypatch.setattr(caller_module, "default_invoke", hanging)

    with pytest.raises(PageTimeoutError):
        _call(_caller())


def test_a_request_cut_by_the_deadline_reports_a_timeout_to_its_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[BaseException] = []
    real_attempt = ext.attempt

    def attempt(
        role: str, provider: str, estimate: int | None = None, **kwargs: Any
    ) -> ext.Attempt[Any]:
        opened: ext.Attempt[Any] = real_attempt(role, provider, estimate, **kwargs)
        on_error = opened.on_error

        def recording(exc: BaseException) -> Any:
            errors.append(exc)
            return on_error(exc)

        opened.on_error = recording  # type: ignore[method-assign]
        return opened

    monkeypatch.setattr(ext, "attempt", attempt)
    helpers.install(monkeypatch, [{"a": 1}])

    async def hanging(model: Any, messages: Any) -> Any:
        await asyncio.sleep(LONG)

    monkeypatch.setattr(caller_module, "default_invoke", hanging)

    with pytest.raises(PageTimeoutError):
        _call(_caller())

    assert [type(exc) for exc in errors] == [TimeoutError]


def test_retry_backoff_past_the_deadline_still_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_attempts(monkeypatch, decisions=[ErrorDecision(Decision.RETRY)])
    script = helpers.install(monkeypatch, [StatusError(400), {"a": 1}])
    monkeypatch.setattr(RetryPolicy, "delay", lambda *args, **kwargs: LONG)

    with pytest.raises(PageTimeoutError):
        _call(_caller())

    assert len(script.requests) == 1
