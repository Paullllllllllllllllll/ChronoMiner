"""Per-phase HTTP timeouts must survive the trip from config to provider.

A scalar float handed to an httpx-backed client applies to every timeout
phase, so a 900 s read budget silently becomes a 900 s connect budget.
These tests pin the config parsing on both ProviderConfig constructors and
the per-provider construction: ChatOpenAI, OpenRouter, and custom
endpoints must receive an ``httpx.Timeout``; ChatAnthropic and
ChatGoogleGenerativeAI must keep receiving a plain float.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from modules.llm.http_timeouts import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_POOL_TIMEOUT,
    DEFAULT_WRITE_TIMEOUT,
)
from modules.llm.langchain_provider import LangChainLLM, ProviderConfig

_FULL_TIMEOUTS = {"total": 900, "connect": 5, "write": 60, "pool": 15}
_GARBAGE_TIMEOUTS = {
    "total": 900,
    "connect": "soon",
    "write": None,
    "pool": -1,
}


def _model_config(name: str = "gpt-4o") -> dict[str, Any]:
    return {
        "extraction_model": {
            "name": name,
            "max_output_tokens": 4096,
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
    }


def _concurrency_config(timeouts: dict[str, Any]) -> dict[str, Any]:
    return {"concurrency": {"extraction": {"timeouts": timeouts}}}


def _live_config(timeouts: dict[str, Any]) -> ProviderConfig:
    """Build a ProviderConfig through the live LLMExtractor path."""
    with patch("modules.llm.openai_utils.get_config_loader") as mock_loader:
        mock_loader.return_value.get_model_config.return_value = _model_config()
        mock_loader.return_value.get_concurrency_config.return_value = (
            _concurrency_config(timeouts)
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            from modules.llm.openai_utils import LLMExtractor

            extractor = LLMExtractor(model="gpt-4o")

    assert extractor._llm is not None
    return extractor._llm.config


class TestProviderConfigTimeoutFields:
    """The three per-phase fields default to the http_timeouts constants."""

    @pytest.mark.unit
    def test_per_phase_defaults(self):
        config = ProviderConfig(provider="openai", model="gpt-4o")

        assert config.connect_timeout == 10.0
        assert config.write_timeout == 30.0
        assert config.pool_timeout == 30.0

    @pytest.mark.unit
    def test_defaults_track_the_shared_constants(self):
        config = ProviderConfig(provider="openai", model="gpt-4o")

        assert config.connect_timeout == DEFAULT_CONNECT_TIMEOUT
        assert config.write_timeout == DEFAULT_WRITE_TIMEOUT
        assert config.pool_timeout == DEFAULT_POOL_TIMEOUT

    @pytest.mark.unit
    def test_timeout_stays_a_plain_float(self):
        config = ProviderConfig(provider="openai", model="gpt-4o", timeout=900.0)

        assert isinstance(config.timeout, float)
        assert config.timeout == 900.0


class TestFromConfigParsing:
    """ProviderConfig.from_config reads the per-phase override keys."""

    @pytest.mark.unit
    def test_parses_all_phases(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            config = ProviderConfig.from_config(
                _model_config(),
                concurrency_config=_concurrency_config(_FULL_TIMEOUTS),
            )

        assert config.timeout == 900.0
        assert config.connect_timeout == 5.0
        assert config.write_timeout == 60.0
        assert config.pool_timeout == 15.0

    @pytest.mark.unit
    def test_garbage_values_fall_back_to_defaults(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            config = ProviderConfig.from_config(
                _model_config(),
                concurrency_config=_concurrency_config(_GARBAGE_TIMEOUTS),
            )

        assert config.timeout == 900.0
        assert config.connect_timeout == DEFAULT_CONNECT_TIMEOUT
        assert config.write_timeout == DEFAULT_WRITE_TIMEOUT
        assert config.pool_timeout == DEFAULT_POOL_TIMEOUT

    @pytest.mark.unit
    def test_absent_timeouts_block_yields_defaults(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            config = ProviderConfig.from_config(
                _model_config(),
                concurrency_config={"concurrency": {"extraction": {}}},
            )

        assert config.connect_timeout == DEFAULT_CONNECT_TIMEOUT
        assert config.write_timeout == DEFAULT_WRITE_TIMEOUT
        assert config.pool_timeout == DEFAULT_POOL_TIMEOUT


class TestLiveExtractorParsing:
    """LLMExtractor._initialize_llm reads the same keys as from_config."""

    @pytest.mark.unit
    def test_parses_all_phases(self):
        config = _live_config(_FULL_TIMEOUTS)

        assert config.timeout == 900.0
        assert config.connect_timeout == 5.0
        assert config.write_timeout == 60.0
        assert config.pool_timeout == 15.0

    @pytest.mark.unit
    def test_garbage_values_fall_back_to_defaults(self):
        config = _live_config(_GARBAGE_TIMEOUTS)

        assert config.timeout == 900.0
        assert config.connect_timeout == DEFAULT_CONNECT_TIMEOUT
        assert config.write_timeout == DEFAULT_WRITE_TIMEOUT
        assert config.pool_timeout == DEFAULT_POOL_TIMEOUT

    @pytest.mark.unit
    def test_absent_timeouts_block_yields_defaults(self):
        config = _live_config({})

        assert config.timeout == 600.0
        assert config.connect_timeout == DEFAULT_CONNECT_TIMEOUT
        assert config.write_timeout == DEFAULT_WRITE_TIMEOUT
        assert config.pool_timeout == DEFAULT_POOL_TIMEOUT


def _capture_openai_kwargs(config: ProviderConfig) -> dict[str, Any]:
    """Build the chat model with ChatOpenAI patched, returning its kwargs."""
    captured: dict[str, Any] = {}

    def _fake_chat_openai(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    with patch("langchain_openai.ChatOpenAI", _fake_chat_openai):
        LangChainLLM(config)._create_chat_model()

    return captured


class TestOpenAIFamilyReceivesHttpxTimeout:
    """openai, openrouter, and custom endpoints get a per-phase timeout."""

    @staticmethod
    def _assert_per_phase(timeout: Any) -> None:
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read == 900.0
        assert timeout.connect == 5.0
        assert timeout.write == 60.0
        assert timeout.pool == 15.0

    @pytest.mark.unit
    def test_openai(self):
        config = ProviderConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            timeout=900.0,
            connect_timeout=5.0,
            write_timeout=60.0,
            pool_timeout=15.0,
        )
        self._assert_per_phase(_capture_openai_kwargs(config)["timeout"])

    @pytest.mark.unit
    def test_openrouter(self):
        config = ProviderConfig(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.5",
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            timeout=900.0,
            connect_timeout=5.0,
            write_timeout=60.0,
            pool_timeout=15.0,
        )
        self._assert_per_phase(_capture_openai_kwargs(config)["timeout"])

    @pytest.mark.unit
    def test_custom_endpoint(self):
        config = ProviderConfig(
            provider="custom",
            model="org/model",
            api_key="test-key",
            base_url="https://example.com/v1",
            timeout=900.0,
            connect_timeout=5.0,
            write_timeout=60.0,
            pool_timeout=15.0,
        )
        self._assert_per_phase(_capture_openai_kwargs(config)["timeout"])

    @pytest.mark.unit
    def test_defaults_reach_the_openai_client(self):
        config = ProviderConfig(
            provider="openai", model="gpt-4o", api_key="test-key", timeout=600.0
        )
        timeout = _capture_openai_kwargs(config)["timeout"]

        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read == 600.0
        assert timeout.connect == DEFAULT_CONNECT_TIMEOUT
        assert timeout.write == DEFAULT_WRITE_TIMEOUT
        assert timeout.pool == DEFAULT_POOL_TIMEOUT


class TestAnthropicAndGoogleKeepPlainFloat:
    """Both wrappers do arithmetic/comparison on the raw timeout value."""

    @staticmethod
    def _capture(target: str, config: ProviderConfig) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        def _fake(**kwargs: Any) -> object:
            captured.update(kwargs)
            return object()

        with patch(target, _fake):
            LangChainLLM(config)._create_chat_model()

        return captured

    @pytest.mark.unit
    def test_anthropic(self):
        config = ProviderConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
            timeout=900.0,
            connect_timeout=5.0,
        )
        kwargs = self._capture("langchain_anthropic.ChatAnthropic", config)

        assert isinstance(kwargs["timeout"], float)
        assert kwargs["timeout"] == 900.0

    @pytest.mark.unit
    def test_google(self):
        config = ProviderConfig(
            provider="google",
            model="gemini-2.0-flash",
            api_key="test-key",
            timeout=900.0,
            connect_timeout=5.0,
        )
        kwargs = self._capture("langchain_google_genai.ChatGoogleGenerativeAI", config)

        assert isinstance(kwargs["timeout"], float)
        assert kwargs["timeout"] == 900.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
