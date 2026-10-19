"""The client factory: key variables, provider classes, overrides, failures."""

from __future__ import annotations

import os
import warnings
from typing import Any
from unittest.mock import patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from chronominer.llm.client import (
    build_chat_model,
    default_key_env,
    get_api_key,
    suppress_pydantic_serializer_warnings,
)
from chronominer.settings import SettingsError
from tests.llm.helpers import custom_spec, model_spec, settings


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "CM_FAKE_CUSTOM_API_KEY",
    ):
        monkeypatch.setenv(name, f"secret-{name}")


class TestKeys:
    """Which variable holds the key and what happens when it is unset."""

    def test_the_identity_variable_wins(
        self, keys: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SECOND_KEY", "second")
        with patch("langchain_openai.ChatOpenAI") as cls:
            build_chat_model(model_spec(), settings(), api_key_env="SECOND_KEY")
        assert cls.call_args.kwargs["api_key"] == "second"

    def test_the_settings_name_the_default_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALT_OPENAI", "alt")
        custom = settings(api_keys={"openai": "ALT_OPENAI"})
        with patch("langchain_openai.ChatOpenAI") as cls:
            build_chat_model(model_spec(), custom)
        assert cls.call_args.kwargs["api_key"] == "alt"

    def test_an_unset_variable_is_an_oserror_naming_it(self) -> None:
        with pytest.raises(OSError, match="OPENAI_API_KEY"):
            build_chat_model(model_spec(), settings())

    def test_a_custom_endpoint_reads_its_own_variable(self, keys: None) -> None:
        with patch("langchain_openai.ChatOpenAI") as cls:
            build_chat_model(custom_spec(), settings())
        kwargs = cls.call_args.kwargs
        assert kwargs["api_key"] == "secret-CM_FAKE_CUSTOM_API_KEY"
        assert kwargs["base_url"] == "https://fake-endpoint.invalid/v1/"

    def test_get_api_key(self) -> None:
        with patch.dict(os.environ, {"K": "v"}, clear=True):
            assert get_api_key("openai", "K") == "v"
            with pytest.raises(OSError, match="api_key_env"):
                get_api_key("custom")
        assert default_key_env("google") == "GOOGLE_API_KEY"
        assert default_key_env("custom") is None


class TestFailures:
    """Specs the factory cannot build."""

    def test_unsupported_provider(self, keys: None) -> None:
        with pytest.raises(ValueError, match="Unsupported provider"):
            build_chat_model(model_spec("x", "acme"), settings())

    def test_unknown_endpoint(self, keys: None) -> None:
        spec = custom_spec(endpoint="missing")
        with pytest.raises(SettingsError, match="unknown endpoint"):
            build_chat_model(spec, settings())


@pytest.mark.parametrize(
    ("spec", "cls"),
    [
        (model_spec(), ChatOpenAI),
        (model_spec("gpt-4o"), ChatOpenAI),
        (model_spec("claude-sonnet-4-5", "anthropic"), ChatAnthropic),
        (model_spec("claude-fable-5", "anthropic"), ChatAnthropic),
        (model_spec("gemini-3-flash", "google"), ChatGoogleGenerativeAI),
        (model_spec("anthropic/claude-sonnet-4.5", "openrouter"), ChatOpenAI),
        (custom_spec(), ChatOpenAI),
    ],
)
def test_the_provider_classes_accept_the_kwargs(
    keys: None, spec: Any, cls: type
) -> None:
    """The real classes take every keyword, without SDK retries."""
    model = build_chat_model(spec, settings())

    assert isinstance(model, cls)
    assert model.max_retries == 0  # type: ignore[attr-defined]


def test_overrides_replace_constructor_kwargs(keys: None) -> None:
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as cls:
        build_chat_model(
            model_spec("gemini-3-flash", "google"),
            settings(),
            overrides={"timeout": None},
        )
    assert cls.call_args.kwargs["timeout"] is None


def _warn(message: str, module: str) -> None:
    warnings.warn_explicit(
        message, UserWarning, f"{module}.py", 1, module=module, registry={}
    )


def test_only_the_pydantic_serializer_warning_is_silenced() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        suppress_pydantic_serializer_warnings()

        _warn("Pydantic serializer warnings:\n  Unexpected(...)", "pydantic.main")
        _warn("some other pydantic warning", "pydantic.main")
        _warn("Pydantic serializer warnings: from elsewhere", "somewhere.else")

    assert [str(warning.message) for warning in record] == [
        "some other pydantic warning",
        "Pydantic serializer warnings: from elsewhere",
    ]
