"""ModelCaller: attempt contexts, the retry ladder, usage totals, waits and the
response format each provider receives."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

import chronominer.llm.caller as caller_module
from chronominer.common.retry import Decision
from chronominer.common.structured import (
    StructuredRequest,
    StructuredResult,
    prepare_request,
    sanitize_schema_for_anthropic,
)
from chronominer.common.testing import FakeRequest, ScriptedChatModel
from chronominer.common.usage import Usage, UsageTotals
from chronominer.events import Waiting
from chronominer.ext import Identity
from chronominer.llm.caller import CallEnv, ModelCaller, collect_usage
from chronominer.llm.client import build_chat_model
from chronominer.settings import RetrySettings
from chronominer.spec import ModelSpec
from tests.llm import helpers
from tests.llm.conftest import Sleeps
from tests.llm.helpers import RecordingExt, StatusError, custom_spec, model_spec
from tests.recording_events import RecordingEvents

MESSAGES = [SystemMessage(content="Extract."), HumanMessage(content="chunk")]
ANSWER_USAGE = Usage(input_tokens=100, output_tokens=20, total_tokens=120)

SELECTED_SCHEMA: dict[str, Any] = {
    "name": "ExtractionSchema",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "contains_no_content_of_requested_type": {"type": "boolean"},
            "entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"item": {"type": ["string", "null"]}},
                    "required": ["item"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["contains_no_content_of_requested_type", "entries"],
        "additionalProperties": False,
    },
}
ANSWER = {
    "contains_no_content_of_requested_type": False,
    "entries": [{"item": "bread"}],
}


def _caller(env: CallEnv | None = None, spec: ModelSpec | None = None) -> ModelCaller:
    return ModelCaller("extract", spec or model_spec(), env or helpers.env())


def _request(
    fmt: str = "json", retries: int = 0, provider: str = "openai"
) -> StructuredRequest:
    return StructuredRequest(
        provider=provider,
        response_format=fmt,  # type: ignore[arg-type]
        schema={"name": "s", "schema": {"type": "object", "required": ["a"]}},
        validation_retries=retries,
    )


def _call(
    caller: ModelCaller, request: StructuredRequest, messages: Any = MESSAGES
) -> StructuredResult:
    return asyncio.run(caller.call(messages, request, label="Extraction of chunk 1"))


def test_a_failed_then_retried_call_opens_one_attempt_per_try(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    recorder = RecordingExt().install(monkeypatch)
    events = RecordingEvents()
    helpers.install(monkeypatch, [StatusError(503), {"a": 1}])
    caller = _caller(helpers.env(clock=sleeps.clock, events=events))

    result = _call(caller, _request())

    assert result.ok and result.data == {"a": 1}
    assert len(recorder.attempts) == 2
    assert recorder.roles == ["extract", "extract"]
    assert all(attempt.released for attempt in recorder.attempts)
    assert [type(error) for error in recorder.errors] == [StatusError]
    assert recorder.commits == [ANSWER_USAGE]
    assert recorder.attempts[0].identity == Identity(
        "openai", "OPENAI_API_KEY", "extract", model="gpt-6-luna"
    )
    (waiting,) = events.of("waiting")
    assert isinstance(waiting, Waiting)
    assert waiting.seconds == sleeps[0] >= 1.0
    assert "server_error" in waiting.reason
    assert "Extraction of chunk 1" in waiting.reason


def test_an_entry_failure_is_consulted_and_retried(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    entry_error = StatusError(503)
    recorder = RecordingExt(entry_errors=[entry_error]).install(monkeypatch)
    events = RecordingEvents()
    script = helpers.install(monkeypatch, [{"a": 1}])
    caller = _caller(helpers.env(clock=sleeps.clock, events=events))

    result = _call(caller, _request())

    assert result.ok
    assert len(recorder.attempts) == 2
    assert recorder.attempts[0].released
    assert recorder.errors == [entry_error]
    assert recorder.commits == [ANSWER_USAGE]
    assert len(script.requests) == 1
    assert len(events.of("waiting")) == 1


def test_an_entry_failure_can_end_the_call(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    recorder = RecordingExt(
        decision=Decision.FAIL, entry_errors=[StatusError(503)]
    ).install(monkeypatch)
    script = helpers.install(monkeypatch, [{"a": 1}])

    with pytest.raises(StatusError):
        _call(_caller(helpers.env(clock=sleeps.clock)), _request())

    assert len(recorder.attempts) == 1
    assert len(recorder.errors) == 1
    assert recorder.commits == []
    assert script.requests == []


def test_json_validation_retries_are_attempts_of_their_own(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    recorder = RecordingExt().install(monkeypatch)
    events = RecordingEvents()
    helpers.install(monkeypatch, ["not json", {"a": 1}])
    env = helpers.env(events=events)

    result = _call(_caller(env), _request(retries=1))

    assert result.ok and result.validation_retries == 1
    assert len(recorder.attempts) == 2
    assert recorder.errors == []
    assert recorder.commits == [ANSWER_USAGE, ANSWER_USAGE]
    assert env.usage.get("extract") == ANSWER_USAGE + ANSWER_USAGE
    assert events.of("waiting") == []


def test_usage_recovered_from_a_failed_attempt_reaches_the_totals(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    usage = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    helpers.install(monkeypatch, [StatusError(500, usage), {"a": 1}])
    env = helpers.env(clock=sleeps.clock)

    _call(_caller(env), _request())

    assert env.usage.roles == ("extract",)
    assert env.usage.get("extract") == ANSWER_USAGE + Usage(7, 3, 10)


def test_the_on_error_decision_drives_the_ladder(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    recorder = RecordingExt(decision=Decision.FAIL).install(monkeypatch)
    script = helpers.install(monkeypatch, [StatusError(503), {"a": 1}])

    with pytest.raises(StatusError):
        _call(_caller(helpers.env(clock=sleeps.clock)), _request())

    assert len(recorder.attempts) == 1
    assert len(script.requests) == 1
    assert sleeps == []


def test_a_switch_key_decision_retries_at_once(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    recorder = RecordingExt(decision=Decision.SWITCH_KEY).install(monkeypatch)
    events = RecordingEvents()
    helpers.install(monkeypatch, [StatusError(429), {"a": 1}])
    env = helpers.env(clock=sleeps.clock, events=events)

    result = _call(_caller(env), _request())

    assert result.ok
    assert len(recorder.attempts) == 2
    (waiting,) = events.of("waiting")
    assert waiting.seconds == 0.0
    assert "switch_key" in waiting.reason


def test_a_client_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    recorder = RecordingExt().install(monkeypatch)
    helpers.install(monkeypatch, [StatusError(400), {"a": 1}])

    with pytest.raises(StatusError):
        _call(_caller(), _request())

    assert len(recorder.attempts) == 1
    assert recorder.commits == []


def test_retries_stop_at_the_attempt_limit(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    recorder = RecordingExt().install(monkeypatch)
    events = RecordingEvents()
    helpers.install(monkeypatch, [StatusError(429)])
    env = helpers.env(RetrySettings(max_attempts=3), clock=sleeps.clock, events=events)

    with pytest.raises(StatusError):
        _call(_caller(env), _request())

    assert len(recorder.attempts) == 3
    assert len(recorder.errors) == 3
    assert len(events.of("waiting")) == 2


def test_cancellation_releases_the_open_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RecordingExt().install(monkeypatch)
    helpers.install(monkeypatch, [{"a": 1}])
    started = asyncio.Event()

    async def hang(*_args: Any) -> Any:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(caller_module, "default_invoke", hang)
    caller = _caller()

    async def scenario() -> None:
        task = asyncio.create_task(caller.call(MESSAGES, _request(), label="p"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert len(recorder.attempts) == 1
    assert recorder.attempts[0].released
    assert recorder.errors == []


def test_collect_usage_counts_the_calls_of_its_block_only(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    helpers.install(monkeypatch, [{"a": 1}, {"a": 2}, {"a": 3}])
    env = helpers.env()
    caller = _caller(env)

    async def main() -> tuple[UsageTotals, UsageTotals, UsageTotals]:
        with collect_usage() as outer:
            await caller.call(MESSAGES, _request(), label="outer")

            async def page() -> UsageTotals:
                with collect_usage() as totals:
                    await caller.call(MESSAGES, _request(), label="p1")
                return totals

            child = await asyncio.create_task(page())
        with collect_usage() as after:
            await caller.call(MESSAGES, _request(), label="after")
        return outer, child, after

    outer, child, after = asyncio.run(main())

    assert outer.get("extract") == ANSWER_USAGE + ANSWER_USAGE
    assert child.get("extract") == ANSWER_USAGE
    assert after.get("extract") == ANSWER_USAGE
    assert env.usage.get("extract") == ANSWER_USAGE + ANSWER_USAGE + ANSWER_USAGE


def test_every_attempt_passes_the_provider_rate_limiter(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(clock=sleeps.clock)
    helpers.install(monkeypatch, [StatusError(503), {"a": 1}])

    _call(_caller(env), _request())

    assert env.limiters.get("openai").total_requests == 2
    assert "anthropic" not in env.limiters


def test_one_client_per_caller_and_one_more_per_other_key_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = helpers.install(monkeypatch, [{"a": 1}])
    caller = _caller()

    same = caller.client_for(Identity("openai", "OPENAI_API_KEY", "extract"))
    other = caller.client_for(Identity("openai", "SECOND_KEY", "extract"))

    assert same is caller.model
    assert other is not caller.model
    assert caller.client_for(Identity("openai", "SECOND_KEY", "x")) is other
    assert [c.get("api_key_env") for c in script.constructions] == [
        "OPENAI_API_KEY",
        "SECOND_KEY",
    ]


def test_a_custom_caller_takes_the_endpoint_key_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RecordingExt().install(monkeypatch)
    script = helpers.install(monkeypatch, [{"a": 1}])

    caller = _caller(spec=custom_spec())
    _call(caller, _request(provider="custom"))

    assert script.constructions[0]["api_key_env"] == "CM_FAKE_CUSTOM_API_KEY"
    assert recorder.attempts[0].identity.api_key_env == "CM_FAKE_CUSTOM_API_KEY"


def test_aclose_closes_every_client_and_swallows_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helpers.install(monkeypatch, [{"a": 1}])
    caller = _caller()
    closed: list[str] = []

    class Client:
        def __init__(self, name: str, fails: bool = False) -> None:
            self.name = name
            self.fails = fails

        async def close(self) -> None:
            closed.append(self.name)
            if self.fails:
                raise RuntimeError("already closed")

    model = SimpleNamespace(
        root_async_client=Client("async", fails=True), root_client=Client("sync")
    )
    monkeypatch.setattr(caller, "_clients", {"OPENAI_API_KEY": model})

    asyncio.run(caller.aclose())

    assert closed == ["async", "sync"]
    assert caller._clients == {}


PROVIDER_SPECS = {
    "openai": model_spec(),
    "anthropic": model_spec("claude-sonnet-4-5", "anthropic"),
    "google": model_spec("gemini-3-flash", "google"),
    "openrouter": model_spec("anthropic/claude-sonnet-4.5", "openrouter"),
    "custom": custom_spec(),
}
_SCHEMA_KWARGS = ("text", "response_format", "response_schema", "response_mime_type")


def _formatted(
    monkeypatch: pytest.MonkeyPatch, provider: str, fmt: str
) -> tuple[FakeRequest, StructuredResult, list[Any]]:
    answer = "Bread prices rose." if fmt == "text" else json.dumps(ANSWER)
    script = helpers.install(monkeypatch, [answer])
    caller = _caller(spec=PROVIDER_SPECS[provider])
    prompt = f"Extract. The JSON schema:\n{json.dumps(SELECTED_SCHEMA['schema'])}"
    messages = [SystemMessage(content=prompt), HumanMessage(content="chunk")]
    request = StructuredRequest(
        provider=provider,
        response_format=fmt,  # type: ignore[arg-type]
        schema=SELECTED_SCHEMA,
    )
    result = _call(caller, request, messages)
    (sent,) = script.requests
    return sent, result, messages


def _enforced_schema(sent: FakeRequest, provider: str) -> Any:
    if provider == "openai":
        assert sent.structured is None
        text_format = sent.kwargs["text"]["format"]
        assert text_format["name"] == SELECTED_SCHEMA["name"]
        assert text_format["strict"] is True
        return text_format["schema"]
    if provider == "custom":
        assert sent.structured is None
        return sent.kwargs["response_format"]["json_schema"]["schema"]
    if provider == "google":
        assert sent.structured is None
        assert sent.kwargs["response_mime_type"] == "application/json"
        return sent.kwargs["response_schema"]
    assert sent.structured is not None
    schema = dict(sent.structured.schema)
    assert schema.pop("title") == SELECTED_SCHEMA["name"]
    if provider == "anthropic":
        assert sent.structured.method == "json_schema"
    return schema


@pytest.mark.parametrize("provider", PROVIDER_SPECS)
def test_schema_format_enforces_the_selected_schema(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    sent, result, _messages = _formatted(monkeypatch, provider, "schema")

    expected = SELECTED_SCHEMA["schema"]
    if provider == "anthropic":
        expected = sanitize_schema_for_anthropic(expected)
    assert _enforced_schema(sent, provider) == expected
    assert result.enforced
    assert result.data == ANSWER


@pytest.mark.parametrize("provider", PROVIDER_SPECS)
def test_json_format_prompts_the_schema_without_binding(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    sent, result, messages = _formatted(monkeypatch, provider, "json")

    assert sent.structured is None
    assert not set(_SCHEMA_KWARGS) & set(sent.kwargs)
    assert sent.messages == messages
    assert "The JSON schema:" in str(sent.messages[0].content)
    assert not result.enforced
    assert result.data == ANSWER


@pytest.mark.parametrize("provider", PROVIDER_SPECS)
def test_text_format_has_no_structured_binding(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    sent, result, _messages = _formatted(monkeypatch, provider, "text")

    assert sent.structured is None
    assert not set(_SCHEMA_KWARGS) & set(sent.kwargs)
    assert result.text == "Bread prices rose."
    assert result.data is None


def test_the_selected_schema_reaches_the_openai_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real client merges the schema format with the verbosity it holds."""
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    model = build_chat_model(model_spec(), helpers.settings())
    prepared = prepare_request(model, "openai", "schema", SELECTED_SCHEMA)

    payload = model._get_request_payload(MESSAGES, **prepared.kwargs)  # type: ignore[attr-defined]

    assert payload["text"] == {
        "format": {
            "type": "json_schema",
            "name": SELECTED_SCHEMA["name"],
            "schema": SELECTED_SCHEMA["schema"],
            "strict": True,
        },
        "verbosity": "medium",
    }
    assert payload["reasoning"] == {"effort": "xhigh"}
    assert payload["service_tier"] == "default"


def _anthropic_payload(
    monkeypatch: pytest.MonkeyPatch, model: str, effort: str, fmt: str = "text"
) -> dict[str, Any]:
    """The Messages API payload the real client builds for one page."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    spec = model_spec(model, "anthropic", reasoning_effort=effort)
    chat = build_chat_model(spec, helpers.settings())
    prepared = prepare_request(chat, "anthropic", fmt, SELECTED_SCHEMA)  # type: ignore[arg-type]
    model_call, kwargs = prepared.runnable, dict(prepared.kwargs)
    if prepared.enforced:
        binding = prepared.runnable.first.steps__["raw"]
        model_call = binding.bound
        kwargs.update(binding.kwargs)
        kwargs.pop("ls_structured_output_format", None)
    payload: dict[str, Any] = model_call._get_request_payload(MESSAGES, **kwargs)
    return payload


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-4-6"])
def test_the_anthropic_payload_carries_the_effort(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    low = _anthropic_payload(monkeypatch, model, "low")
    high = _anthropic_payload(monkeypatch, model, "high")

    assert low != high
    assert low["output_config"] == {"effort": "low"}
    assert high["output_config"] == {"effort": "high"}
    assert low["thinking"] == high["thinking"] == {"type": "adaptive"}


def test_the_anthropic_payload_keeps_the_effort_beside_the_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _anthropic_payload(monkeypatch, "claude-opus-5-5", "medium", "schema")

    assert payload["output_config"]["effort"] == "medium"
    assert payload["output_config"]["format"]["type"] == "json_schema"


@pytest.mark.parametrize("model", ["claude-3-7-sonnet", "claude-haiku-4-5"])
def test_an_anthropic_model_without_effort_sends_none(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    low = _anthropic_payload(monkeypatch, model, "low")
    high = _anthropic_payload(monkeypatch, model, "high")

    assert "output_config" not in low
    assert "output_config" not in high
    if model == "claude-3-7-sonnet":
        assert low == high


def test_an_attempt_on_another_client_rebuilds_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = helpers.install(monkeypatch, [json.dumps(ANSWER)])
    caller = _caller(spec=PROVIDER_SPECS["anthropic"])
    second = ScriptedChatModel(
        responder=script.respond, recorder=script.record, tag="second"
    )

    def other_client(identity: Identity) -> Any:
        return second

    monkeypatch.setattr(caller, "client_for", other_client)
    request = StructuredRequest(
        provider="anthropic", response_format="schema", schema=SELECTED_SCHEMA
    )

    result = _call(caller, request)

    assert result.enforced
    (sent,) = script.requests
    assert sent.tag == "second"
    assert sent.structured is not None and sent.structured.method == "json_schema"
