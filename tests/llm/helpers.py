"""Builders for LLM-layer tests: model specs, scripted chat models, call envs.

``install`` replaces the client factory used by ``ModelCaller`` with one that
returns a :class:`ScriptedChatModel`, so callers run their real structured
routing, retry ladder and attempt contexts offline.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable
from types import MappingProxyType
from typing import Any

import pytest
from langchain_core.messages import AIMessage

import chronominer.llm.caller as caller_module
from chronominer import ext
from chronominer.common.rate_limit import RateLimiterRegistry
from chronominer.common.retry import Decision
from chronominer.common.testing import FakeRequest, ScriptedChatModel
from chronominer.common.usage import Usage
from chronominer.events import RunEvents
from chronominer.llm.caller import CallEnv
from chronominer.settings import Endpoint, RetrySettings, Settings
from chronominer.spec import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SERVICE_TIER,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_VERBOSITY,
    ModelSpec,
)

USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}

CUSTOM_ENDPOINT = Endpoint(
    "fake",
    "https://fake-endpoint.invalid/v1/",
    "CM_FAKE_CUSTOM_API_KEY",
    supports_vision=True,
    supports_schema=True,
)


def model_spec(
    model: str = DEFAULT_MODEL, provider: str = "openai", **changes: Any
) -> ModelSpec:
    """Return a model spec with the code defaults and *changes* applied."""
    values: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "endpoint": None,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "verbosity": DEFAULT_VERBOSITY,
        "max_output_tokens": None,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "service_tier": DEFAULT_SERVICE_TIER,
        "response_format": None,
    }
    values.update(changes)
    return ModelSpec(**values)


def custom_spec(**changes: Any) -> ModelSpec:
    """Return the spec of a model behind :data:`CUSTOM_ENDPOINT`."""
    changes.setdefault("endpoint", CUSTOM_ENDPOINT.name)
    return model_spec("org/fake-model", "custom", **changes)


def settings(**values: Any) -> Settings:
    """Return code-default settings with :data:`CUSTOM_ENDPOINT` and *values*."""
    values.setdefault(
        "endpoints", MappingProxyType({CUSTOM_ENDPOINT.name: CUSTOM_ENDPOINT})
    )
    return dataclasses.replace(Settings(), **values)


def message(content: str = "", **fields: Any) -> AIMessage:
    """Return an answer message with the default usage."""
    fields.setdefault("usage_metadata", dict(USAGE))
    return AIMessage(content=content, **fields)


class Script:
    """Answers handed out in order; records every request.

    An answer is a string (message content), a dict (sent as JSON), an
    ``AIMessage``, an exception to raise, or a callable that turns the
    request into one of those. The last answer repeats.
    """

    def __init__(self, answers: Iterable[Any]) -> None:
        self.answers = list(answers)
        self.requests: list[FakeRequest] = []
        self.constructions: list[dict[str, Any]] = []

    def record(self, request: FakeRequest) -> None:
        self.requests.append(request)

    def respond(self, request: FakeRequest) -> AIMessage:
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if callable(answer) and not isinstance(answer, type):
            answer = answer(request)
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, AIMessage):
            return answer
        if isinstance(answer, dict):
            return message(json.dumps(answer))
        return message(str(answer))

    def model(self, **kwargs: Any) -> ScriptedChatModel:
        self.constructions.append(kwargs)
        return ScriptedChatModel(responder=self.respond, recorder=self.record)


def install(monkeypatch: pytest.MonkeyPatch, answers: Iterable[Any]) -> Script:
    """Make every client built by ``ModelCaller`` answer from *answers*."""
    script = Script(answers)

    def build(model: ModelSpec, settings: Any, **kwargs: Any) -> ScriptedChatModel:
        return script.model(spec=model, **kwargs)

    monkeypatch.setattr(caller_module, "build_chat_model", build)
    return script


class StatusError(Exception):
    """A provider error with an HTTP status and, optionally, usage in its body."""

    def __init__(self, status: int, usage: dict[str, int] | None = None) -> None:
        super().__init__(f"status {status}")
        self.status_code = status
        self.body = {"usage": usage} if usage else None


class RecordingExt:
    """Wrap ``ext.attempt`` and record attempts, commits and ``on_error`` calls.

    ``attempts`` holds the attempts that were entered, ``opened`` every
    attempt, including those only built. *decision*, when set, replaces the
    decision ``on_error`` returns. *entry_errors* are raised, in order, by
    the first entries, as a failing reservation would.
    """

    def __init__(
        self,
        decision: Decision | None = None,
        entry_errors: Iterable[BaseException] = (),
    ) -> None:
        self.decision = decision
        self.entry_errors = list(entry_errors)
        self.opened: list[ext.Attempt[Any]] = []
        self.attempts: list[ext.Attempt[Any]] = []
        self.commits: list[Usage | None] = []
        self.errors: list[BaseException] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> RecordingExt:
        real = ext.attempt

        def attempt(
            role: str, provider: str, estimate: int | None = None, **kwargs: Any
        ) -> ext.Attempt[Any]:
            opened: ext.Attempt[Any] = real(role, provider, estimate, **kwargs)
            commit, on_error, refresh = opened.commit, opened.on_error, opened.refresh

            async def recording_refresh() -> Any:
                if opened not in self.attempts:
                    self.attempts.append(opened)
                if self.entry_errors:
                    raise self.entry_errors.pop(0)
                return await refresh()

            def recording_commit(usage: Usage | None) -> None:
                self.commits.append(usage)
                commit(usage)

            def recording_on_error(exc: BaseException) -> Decision:
                self.errors.append(exc)
                decision = on_error(exc)
                return self.decision or decision

            opened.commit = recording_commit  # type: ignore[method-assign]
            opened.on_error = recording_on_error  # type: ignore[method-assign]
            opened.refresh = recording_refresh  # type: ignore[method-assign]
            self.opened.append(opened)
            return opened

        monkeypatch.setattr(ext, "attempt", attempt)
        return self

    @property
    def roles(self) -> list[str]:
        """The role of every entered attempt, in order."""
        return [attempt.identity.role for attempt in self.attempts]


def env(
    retry: RetrySettings | None = None,
    *,
    clock: Callable[[], float] | None = None,
    events: RunEvents | None = None,
    **values: Any,
) -> CallEnv:
    """Return a call env with wide rate limits.

    *clock* drives the rate limiters, so their error penalty ends in fake
    time when the test also fakes the sleeps.
    """
    values.setdefault("rate_limits", ((100_000, 1),))
    if retry is not None:
        values["retry"] = retry
    created = CallEnv.create(settings(**values), events)
    if clock is None:
        return created
    limiters = RateLimiterRegistry(created.settings.rate_limits, clock=clock)
    return dataclasses.replace(created, limiters=limiters)


__all__ = [
    "CUSTOM_ENDPOINT",
    "USAGE",
    "RecordingExt",
    "Script",
    "StatusError",
    "custom_spec",
    "env",
    "install",
    "message",
    "model_spec",
    "settings",
]
