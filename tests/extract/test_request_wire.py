"""Invoke-time schema enforcement sends what ``with_structured_output`` sends.

The characterization fake records a schema passed as an invoke kwarg as the
``json_schema`` structured-output call (``tests/characterization/fakes.py``).
These tests build the real provider payloads of the extraction schema offline
and check that both forms produce the same request.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from chronominer.common.structured import (
    prepare_request,
    sanitize_schema_for_anthropic,
)
from chronominer.extract.extraction import (
    ANTHROPIC_SCHEMA_DESCRIPTION,
    anthropic_schema,
)
from chronominer.llm.client import build_chat_model
from chronominer.llm.prompts import EXTRACTION_SCHEMA_NAME, extraction_schema
from chronominer.spec import resolve_schema
from tests.llm.helpers import CUSTOM_ENDPOINT, custom_spec, model_spec, settings

PROVIDER_SPECS = {
    "openai": model_spec(),
    "anthropic": model_spec("claude-sonnet-4-5", "anthropic"),
    "google": model_spec("gemini-3-flash", "google"),
    "openrouter": model_spec("anthropic/claude-sonnet-4.5", "openrouter"),
    "custom": custom_spec(),
}
MESSAGES = [SystemMessage(content="Extract."), HumanMessage(content="chunk")]
SCHEMA: dict[str, Any] = extraction_schema(resolve_schema("WageRecords").schema)


def _public(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return *kwargs* without the tracing keys."""
    return {key: value for key, value in kwargs.items() if not key.startswith("ls_")}


def _bound_kwargs(model: Any, schema: Any) -> dict[str, Any]:
    """Return the kwargs ``with_structured_output`` binds, tracing keys excluded."""
    runnable = model.with_structured_output(
        schema, method="json_schema", include_raw=True
    )
    return _public(runnable.first.steps__["raw"].kwargs)


def test_openai_text_format_is_the_json_schema_request() -> None:
    model = ChatOpenAI(
        api_key="unused",  # type: ignore[arg-type]
        model="gpt-5.6-luna",
        use_responses_api=True,
        verbosity="medium",
    )
    prepared = prepare_request(model, "openai", "schema", SCHEMA)

    structured = model._get_request_payload(MESSAGES, **_bound_kwargs(model, SCHEMA))
    invoked = model._get_request_payload(MESSAGES, **prepared.kwargs)

    assert invoked == structured
    assert invoked["text"]["format"]["name"] == EXTRACTION_SCHEMA_NAME


def test_a_custom_response_format_is_the_json_schema_request() -> None:
    model = ChatOpenAI(
        api_key="unused",  # type: ignore[arg-type]
        model="org/model",
        base_url="https://fake-endpoint.invalid/v1",
    )
    prepared = prepare_request(model, "custom", "schema", SCHEMA)

    structured = model._get_request_payload(MESSAGES, **_bound_kwargs(model, SCHEMA))
    invoked = model._get_request_payload(MESSAGES, **prepared.kwargs)

    assert invoked == structured
    assert invoked["response_format"]["json_schema"]["name"] == EXTRACTION_SCHEMA_NAME


def test_a_gemini_response_schema_is_the_json_schema_request() -> None:
    model = ChatGoogleGenerativeAI(model="gemini-3-flash", google_api_key="unused")
    prepared = prepare_request(model, "google", "schema", SCHEMA)

    structured = model._prepare_request(
        MESSAGES, **_bound_kwargs(model, SCHEMA["schema"])
    )
    invoked = model._prepare_request(MESSAGES, **prepared.kwargs)

    assert invoked == structured


def test_an_anthropic_output_format_is_the_json_schema_request() -> None:
    """The extractor titles and describes the schema; the request turns its
    nullable fields into ``anyOf``."""
    model = ChatAnthropic(
        api_key="unused",  # type: ignore[arg-type]
        model_name="claude-sonnet-4-5",
        timeout=None,
        stop=None,
    )
    prepared = prepare_request(model, "anthropic", "schema", anthropic_schema(SCHEMA))
    expected = {
        **sanitize_schema_for_anthropic(SCHEMA["schema"]),
        "title": EXTRACTION_SCHEMA_NAME,
        "description": ANTHROPIC_SCHEMA_DESCRIPTION,
    }

    binding = prepared.runnable.first.steps__["raw"]
    invoked = binding.bound._get_request_payload(
        MESSAGES, **{**prepared.kwargs, **_public(binding.kwargs)}
    )
    structured = model._get_request_payload(MESSAGES, **_bound_kwargs(model, expected))

    assert invoked == structured
    assert invoked["output_config"]["format"]["schema"] == expected


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        CUSTOM_ENDPOINT.api_key_env,
    ):
        monkeypatch.setenv(name, "unused")


@pytest.mark.usefixtures("keys")
def test_an_openrouter_response_format_carries_the_schema() -> None:
    """LangChain sends the schema title as the name and drops it from the body."""
    model = build_chat_model(PROVIDER_SPECS["openrouter"], settings())
    prepared = prepare_request(model, "openrouter", "schema", SCHEMA)
    binding = prepared.runnable.first.steps__["raw"]
    payload = binding.bound._get_request_payload(
        MESSAGES, **{**prepared.kwargs, **_public(binding.kwargs)}
    )
    json_schema = payload["response_format"]["json_schema"]

    assert prepared.enforced is True
    assert json_schema["name"] == EXTRACTION_SCHEMA_NAME
    assert json_schema["schema"] == SCHEMA["schema"]


def _payload(model: Any, provider: str, **kwargs: Any) -> Any:
    if provider == "google":
        return model._prepare_request(MESSAGES, **kwargs)
    return model._get_request_payload(MESSAGES, **kwargs)


@pytest.mark.usefixtures("keys")
@pytest.mark.parametrize("provider", list(PROVIDER_SPECS))
def test_the_json_format_sends_no_schema(provider: str) -> None:
    model = build_chat_model(PROVIDER_SPECS[provider], settings())
    prepared = prepare_request(model, provider, "json", SCHEMA)

    invoked = _payload(prepared.runnable, provider, **prepared.kwargs)

    assert not prepared.enforced
    assert invoked == _payload(model, provider)
    assert EXTRACTION_SCHEMA_NAME not in str(invoked)
    assert "contains_no_content_of_requested_type" not in str(invoked)
