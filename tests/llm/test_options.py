"""Constructor options per provider, output limits and the response format."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from chronominer.llm.options import (
    constructor_kwargs,
    disabled_params,
    model_capabilities,
    output_limit,
    response_format_for,
)
from chronominer.settings import PromptCaching
from chronominer.spec import REASONING_EFFORTS, ModelSpec
from tests.llm.helpers import custom_spec, model_spec, settings


def _kwargs(spec: ModelSpec, **values: Any) -> dict[str, Any]:
    return constructor_kwargs(spec, settings(**values))


def _limit(spec: ModelSpec) -> int | None:
    return output_limit(spec, model_capabilities(spec, settings()))


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        pytest.param(model_spec(), 128000, id="openai-family"),
        pytest.param(model_spec("my-finetune"), None, id="openai-unknown"),
        pytest.param(model_spec("claude-sonnet-4-5", "anthropic"), 64000, id="claude"),
        pytest.param(model_spec("my-model", "anthropic"), 4096, id="claude-unknown"),
        pytest.param(model_spec("gemini-2.5-flash", "google"), 65536, id="gemini"),
        pytest.param(model_spec("my-model", "google"), None, id="google-unknown"),
        pytest.param(
            model_spec("anthropic/claude-sonnet-4.5", "openrouter"),
            128000,
            id="routed",
        ),
        pytest.param(custom_spec(), 128000, id="custom"),
    ],
)
def test_an_unset_limit_takes_the_family_limit(
    spec: ModelSpec, expected: int | None
) -> None:
    assert _limit(spec) == expected


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        pytest.param(model_spec(max_output_tokens=200000), 200000, id="openai"),
        pytest.param(
            model_spec("claude-sonnet-4-5", "anthropic", max_output_tokens=200000),
            64000,
            id="anthropic-clamped",
        ),
        pytest.param(
            model_spec("gemini-3-flash", "google", max_output_tokens=5000),
            5000,
            id="google-below-limit",
        ),
        pytest.param(
            model_spec("gemini-3-flash", "google", max_output_tokens=200000),
            65536,
            id="google-clamped",
        ),
        pytest.param(
            model_spec("org/model", "openrouter", max_output_tokens=200000),
            200000,
            id="openrouter",
        ),
        pytest.param(custom_spec(max_output_tokens=5000), 5000, id="custom"),
    ],
)
def test_a_set_limit_is_clamped_only_on_anthropic_and_google(
    spec: ModelSpec, expected: int
) -> None:
    assert _limit(spec) == expected


def test_routed_and_custom_models_send_the_routed_limit_when_unset() -> None:
    routed = _kwargs(model_spec("anthropic/claude-sonnet-4.5", "openrouter"))
    custom = _kwargs(custom_spec())

    assert routed["max_tokens"] == 128000
    assert routed["extra_body"] == {"reasoning": {"max_tokens": 32000}}
    assert custom["max_tokens"] == 128000


def test_openai_standard_models_take_samplers_and_max_tokens() -> None:
    kwargs = _kwargs(model_spec("gpt-4o", temperature=0.5, top_p=0.9))

    assert kwargs["max_tokens"] == 16384
    assert kwargs["temperature"] == 0.5
    assert kwargs["top_p"] == 0.9
    assert kwargs["disabled_params"] == {
        "frequency_penalty": None,
        "presence_penalty": None,
    }
    assert "reasoning" not in kwargs
    assert "verbosity" not in kwargs
    assert "max_completion_tokens" not in kwargs


@pytest.mark.parametrize(
    ("model", "sent"),
    [("gpt-4o", False), ("gpt-4.1", False), ("o1", True), ("gpt-5.6-luna", True)],
)
def test_the_service_tier_reaches_reasoning_models_only(model: str, sent: bool) -> None:
    kwargs = _kwargs(model_spec(model, service_tier="priority"))

    assert ("service_tier" in kwargs) is sent
    if sent:
        assert kwargs["service_tier"] == "priority"


def test_openai_reasoning_without_effort_support_sends_no_reasoning() -> None:
    kwargs = _kwargs(model_spec("o1"))

    assert "reasoning" not in kwargs
    assert kwargs["max_completion_tokens"] == 100000


@pytest.mark.parametrize(
    ("caching", "model_kwargs"),
    [
        (
            PromptCaching(enabled=True, openai_retention="24h"),
            {"prompt_cache_retention": "24h"},
        ),
        (PromptCaching(enabled=True, openai_retention=None), None),
        (PromptCaching(enabled=False, openai_retention="24h"), None),
    ],
    ids=["set", "null", "disabled"],
)
def test_openai_prompt_cache_retention_is_sent_only_when_set_and_on(
    caching: PromptCaching, model_kwargs: dict[str, str] | None
) -> None:
    """gpt-5.5 is documented for extended retention (prompt_cache_retention)."""
    kwargs = _kwargs(model_spec("gpt-5.5"), prompt_caching=caching)

    assert kwargs.get("model_kwargs") == model_kwargs
    assert "prompt_cache_options" not in kwargs


def test_adaptive_models_take_adaptive_thinking_and_no_samplers() -> None:
    kwargs = _kwargs(model_spec("claude-fable-5", "anthropic"))

    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["max_tokens"] == 128000
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_effort_none_turns_budget_thinking_off() -> None:
    kwargs = _kwargs(
        model_spec("claude-sonnet-4-5", "anthropic", reasoning_effort="none")
    )

    assert "thinking" not in kwargs
    assert kwargs["temperature"] == 1.0
    assert "top_p" not in kwargs


@pytest.mark.parametrize(
    ("limit", "thinking"),
    [
        (2000, {"type": "enabled", "budget_tokens": 1744}),
        (1200, None),
    ],
)
def test_the_thinking_budget_leaves_room_for_the_answer(
    limit: int, thinking: dict[str, Any] | None
) -> None:
    kwargs = _kwargs(
        model_spec("claude-sonnet-4-5", "anthropic", max_output_tokens=limit)
    )

    assert kwargs.get("thinking") == thinking


def test_models_without_thinking_keep_their_samplers() -> None:
    kwargs = _kwargs(model_spec("claude-3-5-sonnet", "anthropic", top_p=0.5))

    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 0.5
    assert kwargs["max_tokens"] == 8192


@pytest.mark.parametrize(
    ("model", "levels"),
    [
        ("claude-opus-5-5", [None, "low", "low", "medium", "high", "xhigh", "max"]),
        ("claude-sonnet-4-6", [None, "low", "low", "medium", "high", "high", "max"]),
        ("claude-opus-4-5", [None, "low", "low", "medium", "high", "high", "high"]),
    ],
)
def test_effort_maps_every_level_to_one_the_family_takes(
    model: str, levels: list[str | None]
) -> None:
    sent = [
        _kwargs(model_spec(model, "anthropic", reasoning_effort=effort)).get("effort")
        for effort in REASONING_EFFORTS
    ]

    assert sent == levels


@pytest.mark.parametrize("model", ["claude-sonnet-4-5", "claude-3-7-sonnet"])
def test_families_without_effort_send_none(model: str) -> None:
    for effort in REASONING_EFFORTS:
        spec = model_spec(model, "anthropic", reasoning_effort=effort)
        assert "effort" not in _kwargs(spec)


@pytest.mark.parametrize(
    ("model", "effort", "level"),
    [
        ("gemini-3.5-flash", "max", "high"),
        ("gemini-3.5-flash", "none", None),
        ("gemini-3.5-flash", "xhigh", "high"),
        ("gemini-2.5-pro", "high", None),
        ("gemini-2.0-flash", "high", None),
    ],
)
def test_thinking_level_is_sent_to_gemini_3_and_later(
    model: str, effort: str, level: str | None
) -> None:
    kwargs = _kwargs(model_spec(model, "google", reasoning_effort=effort))

    assert kwargs.get("thinking_level") == level


@pytest.mark.parametrize(
    ("model", "effort", "limit", "budget"),
    [
        ("gemini-2.5-flash", "low", 8192, 1024),
        ("gemini-2.5-flash", "medium", None, 16384),
        ("gemini-2.5-flash", "high", None, 24576),
        ("gemini-2.5-pro", "high", None, 32768),
        ("gemini-2.5-flash", "high", 1500, 1024),
        ("gemini-2.5-flash", "high", 1024, None),
        ("gemini-2.5-flash", "none", None, None),
        ("gemini-3.5-flash", "high", None, None),
    ],
)
def test_thinking_budget_is_sent_to_gemini_2(
    model: str, effort: str, limit: int | None, budget: int | None
) -> None:
    spec = model_spec(model, "google", reasoning_effort=effort, max_output_tokens=limit)

    assert _kwargs(spec).get("thinking_budget") == budget


def test_gemini_models_without_sampling_get_no_sampler_kwargs() -> None:
    kwargs = _kwargs(model_spec("gemini-3.7-flash", "google"))

    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert kwargs["top_k"] is None


def test_deepseek_reasoning_is_switched_on_or_off() -> None:
    on = _kwargs(model_spec("deepseek/deepseek-r1", "openrouter"))
    off = _kwargs(
        model_spec("deepseek/deepseek-r1", "openrouter", reasoning_effort="none")
    )

    assert on["extra_body"] == {"reasoning": {"enabled": True}}
    assert off["extra_body"] == {"reasoning": {"enabled": False}}
    # OpenRouter lists temperature and top_p for deepseek-r1.
    assert "temperature" in on and "top_p" in on


def test_routed_gemini_takes_a_reasoning_token_limit() -> None:
    kwargs = _kwargs(
        model_spec(
            "google/gemini-3-flash",
            "openrouter",
            reasoning_effort="low",
            max_output_tokens=20000,
        )
    )

    assert kwargs["extra_body"] == {"reasoning": {"max_tokens": 4000}}


def test_routed_penalties_are_sent_where_supported() -> None:
    kwargs = _kwargs(model_spec("openai/gpt-oss-120b", "openrouter"))

    assert kwargs["frequency_penalty"] == 0.0
    assert kwargs["presence_penalty"] == 0.0
    assert kwargs["disabled_params"] is None
    assert kwargs["extra_body"] == {"reasoning": {"effort": "xhigh"}}


def test_routed_models_without_reasoning_get_no_extra_body() -> None:
    kwargs = _kwargs(model_spec("mistralai/mistral-large", "openrouter"))

    assert "extra_body" not in kwargs


def test_custom_capabilities_come_from_the_endpoint() -> None:
    caps = model_capabilities(custom_spec(), settings())

    assert caps.family == "custom"
    assert caps.supports_vision is True
    assert caps.supports_structured_output is True


def test_a_custom_model_needs_its_endpoint() -> None:
    with pytest.raises(ValueError, match="named endpoint"):
        _kwargs(model_spec("org/model", "custom"))


def test_an_unknown_provider_is_an_error() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        _kwargs(model_spec("x", "acme"))


def test_disabled_params_follow_the_sampler_flags() -> None:
    caps = model_capabilities(model_spec("claude-sonnet-4-5", "anthropic"), settings())

    assert disabled_params(caps) == {
        "top_p": None,
        "frequency_penalty": None,
        "presence_penalty": None,
    }


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (model_spec(), "schema"),
        (model_spec("o3-deep-research"), "json"),
        (model_spec("claude-sonnet-4-5", "anthropic"), "schema"),
        (model_spec("my-finetune"), "json"),
        (custom_spec(), "schema"),
        (model_spec(response_format="json"), "json"),
        (model_spec("gemini-2.5-flash", "google", response_format="text"), "text"),
    ],
)
def test_response_format_for(spec: ModelSpec, expected: str) -> None:
    caps = model_capabilities(spec, settings())
    assert response_format_for(spec, caps) == expected


def test_a_custom_endpoint_keeps_a_requested_schema() -> None:
    spec = custom_spec(response_format="schema")
    caps = model_capabilities(spec, settings())
    unsupported = type(caps)(provider_name="custom", model_name=spec.model)

    assert response_format_for(spec, unsupported) == "schema"


def test_schema_without_structured_output_falls_back_to_json(
    caplog: pytest.LogCaptureFixture,
) -> None:
    spec = model_spec("o3-deep-research", response_format="schema")
    caps = model_capabilities(spec, settings())

    with caplog.at_level(logging.WARNING):
        assert response_format_for(spec, caps) == "json"
    assert "prompted JSON" in caplog.text


def test_unknown_response_format_is_an_error() -> None:
    spec = model_spec(response_format="xml")
    with pytest.raises(ValueError, match="Unknown response format"):
        response_format_for(spec, model_capabilities(spec, settings()))
