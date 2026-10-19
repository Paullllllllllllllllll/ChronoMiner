"""Tests for the attempt decisions handed to the retry ladder."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from chronominer import ext
from chronominer.common.retry import Decision, default_decision
from chronominer.llm.retrying import AttemptDecisions

_IDENTITY = ext.Identity(provider="openai", api_key_env="OPENAI_API_KEY", role="t")


class _Attempt(ext.Attempt[object]):
    """An attempt whose ``on_error`` returns a fixed decision."""

    def __init__(self, decision: Decision, *, fail_entry: bool = False) -> None:
        super().__init__(_IDENTITY, None, self._build)
        self.decision = decision
        self.fail_entry = fail_entry

    def _build(self, identity: ext.Identity) -> object:
        if self.fail_entry:
            raise OSError("key variable unset")
        return object()

    def on_error(self, exc: BaseException) -> Decision:
        super().on_error(exc)
        return self.decision


async def test_entry_failure_records_decision_then_runs_callback() -> None:
    decisions = AttemptDecisions()
    attempt = _Attempt(Decision.SWITCH_KEY, fail_entry=True)
    seen: list[Decision] = []
    with pytest.raises(OSError) as info:
        async with contextlib.AsyncExitStack() as stack:
            await decisions.enter(
                stack,
                attempt,
                on_failure=lambda: seen.append(decisions.decide(attempt.errors[0])),
            )
    assert attempt.errors == [info.value]
    assert seen == [Decision.SWITCH_KEY]


async def test_entry_failure_decision_reaches_decide() -> None:
    decisions = AttemptDecisions()
    with pytest.raises(OSError) as info:
        async with contextlib.AsyncExitStack() as stack:
            await decisions.enter(stack, _Attempt(Decision.FAIL, fail_entry=True))
    assert decisions.decide(info.value) is Decision.FAIL


async def test_entered_attempt_is_returned() -> None:
    decisions = AttemptDecisions()
    attempt = _Attempt(Decision.RETRY)
    async with contextlib.AsyncExitStack() as stack:
        entered = await decisions.enter(stack, attempt)
        assert entered is attempt
        assert attempt.client is not None
    assert attempt.released


def test_call_failure_decision_is_popped_once() -> None:
    decisions = AttemptDecisions()
    exc = ValueError("bad answer")
    decisions.failed(_Attempt(Decision.WAIT), exc)
    assert decisions.decide(exc) is Decision.WAIT
    assert decisions.decide(exc) is default_decision(exc)


def test_unrecorded_exception_gets_default_decision() -> None:
    exc = ConnectionError("reset")
    assert AttemptDecisions().decide(exc) is default_decision(exc)


async def test_concurrent_attempts_keep_their_own_decisions() -> None:
    decisions = AttemptDecisions()
    started = asyncio.Event()
    errors: dict[Decision, Exception] = {}

    async def one(decision: Decision, wait: bool) -> None:
        attempt = _Attempt(decision)
        async with contextlib.AsyncExitStack() as stack:
            entered = await decisions.enter(stack, attempt)
            if wait:
                await started.wait()
            else:
                started.set()
            exc = RuntimeError(decision.value)
            decisions.failed(entered, exc)
            errors[decision] = exc

    await asyncio.gather(one(Decision.RETRY, True), one(Decision.FAIL, False))
    assert decisions.decide(errors[Decision.RETRY]) is Decision.RETRY
    assert decisions.decide(errors[Decision.FAIL]) is Decision.FAIL
