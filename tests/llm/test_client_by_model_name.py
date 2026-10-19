"""A bare model name selects its provider and that provider's client class."""

from __future__ import annotations

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from chronominer.llm.capabilities import detect_capabilities
from chronominer.llm.client import build_chat_model
from chronominer.providers import infer_provider
from chronominer.spec import parse
from tests.llm.helpers import CUSTOM_ENDPOINT, custom_spec, model_spec, settings

_KEY_VARIABLES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    CUSTOM_ENDPOINT.api_key_env,
)


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _KEY_VARIABLES:
        monkeypatch.setenv(name, f"secret-{name}")


@pytest.mark.parametrize(
    ("name", "provider", "cls"),
    [
        ("gpt-5.6-luna", "openai", ChatOpenAI),
        ("o3-mini", "openai", ChatOpenAI),
        ("claude-sonnet-4-5", "anthropic", ChatAnthropic),
        ("gemini-3-flash", "google", ChatGoogleGenerativeAI),
        ("anthropic/claude-sonnet-4.5", "openrouter", ChatOpenAI),
        ("my-finetune", "openai", ChatOpenAI),
    ],
)
def test_a_bare_model_name_selects_the_provider_and_its_client(
    name: str, provider: str, cls: type
) -> None:
    assert infer_provider(name) == provider

    model = build_chat_model(model_spec(name, provider), settings())

    assert type(model) is cls


@pytest.mark.parametrize("name", ["models/gemini-2.5-flash", "gemma-4-31b-it"])
def test_google_ids_from_the_model_flag_reach_the_google_client(name: str) -> None:
    """Routing and capability resolution agree on the provider."""
    spec = parse(["--input", "x", "--schema", "s", "--model", name])

    assert spec.model.provider == "google"
    assert detect_capabilities(name).provider_name == "google"
    model = build_chat_model(spec.model, settings())
    assert type(model) is ChatGoogleGenerativeAI


def test_an_explicit_prefix_still_wins_over_detection() -> None:
    argv = ["--input", "x", "--schema", "s", "--model", "openrouter:google/gemma-3"]
    spec = parse(argv)
    assert spec.model.provider == "openrouter"
    assert spec.model.model == "google/gemma-3"


def test_a_custom_endpoint_is_never_inferred_from_a_name() -> None:
    assert infer_provider(custom_spec().model) == "openrouter"

    model = build_chat_model(custom_spec(), settings())

    assert type(model) is ChatOpenAI
    assert str(model.openai_api_base).startswith(CUSTOM_ENDPOINT.base_url)
