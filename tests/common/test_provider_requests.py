"""The shared request builder: one set of parameters per provider.

Synchronous calls send :func:`request_params` as invoke kwargs and batch
bodies carry the same parameters, so both paths send the same thinking,
effort, output limit, schema and media resolution.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from chronominer.common import provider_requests
from chronominer.common.capabilities import resolve_capabilities
from chronominer.common.capability_table import TABLE
from chronominer.common.image_presets import MAX_IMAGE_BYTES
from chronominer.common.provider_requests import (
    GOOGLE_REQUEST_BYTES,
    Part,
    RequestPlan,
    RequestTooLargeError,
    batch_body,
    check_request_size,
    message_content,
    request_params,
)
from chronominer.common.request_options import OutputLimits, RequestRules
from chronominer.common.settings_schema import PromptCaching

RULES = RequestRules(
    limits=OutputLimits(
        conservative=8192, clamped=frozenset({"anthropic", "google"}), routed=128000
    )
)
SCHEMA: dict[str, Any] = {
    "name": "answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"word": {"type": "string"}},
        "required": ["word"],
        "additionalProperties": False,
    },
}
IMAGE = Part(data="aGVsbG8=", mime="image/jpeg", detail="high")
SAMPLERS: dict[str, Any] = {
    "presence_penalty": 1.5,
    "frequency_penalty": -0.5,
    "top_k": 20,
    "min_p": 0.05,
    "repetition_penalty": 1.1,
}
"""A value for every sampler without a code default."""


def _plan(
    provider: str, model: str, effort: str | None = "low", **values: Any
) -> RequestPlan:
    caps = resolve_capabilities(f"{provider}:{model}", TABLE)
    return RequestPlan.resolve(provider, model, caps, RULES, effort=effort, **values)


def test_an_adaptive_claude_model_gets_adaptive_thinking_and_the_effort() -> None:
    params = request_params(_plan("anthropic", "claude-haiku-5-5", "low"))

    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "low"}
    assert "temperature" not in params


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "claude-sonnet-4-5"])
def test_effort_none_sends_neither_thinking_nor_effort(model: str) -> None:
    params = request_params(_plan("anthropic", model, "none"))

    assert "thinking" not in params
    assert "output_config" not in params


def test_a_claude_model_without_adaptive_thinking_gets_a_budget() -> None:
    params = request_params(_plan("anthropic", "claude-sonnet-4-5", "low"))

    assert params["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_a_gemini_batch_body_enforces_the_schema_with_thinking_on() -> None:
    body = batch_body(
        _plan("google", "gemini-3.1-flash-lite", "low"), "Answer.", [IMAGE], SCHEMA
    )

    config = body["config"]
    assert config["thinking_config"] == {"thinking_level": "low"}
    assert config["response_json_schema"] == SCHEMA["schema"]
    assert config["response_mime_type"] == "application/json"
    image = body["contents"][0]["parts"][0]
    assert image["media_resolution"] == {"level": "MEDIA_RESOLUTION_HIGH"}


def test_a_gemini_image_carries_its_media_resolution_in_a_sync_call() -> None:
    (block,) = message_content(_plan("google", "gemini-3.1-flash-lite"), [IMAGE])

    assert isinstance(block, dict)
    assert block["media_resolution"] == "MEDIA_RESOLUTION_HIGH"
    assert block["type"] == "image"


def test_a_gemini_2_model_gets_a_budget_and_no_media_resolution() -> None:
    plan = _plan("google", "gemini-2.5-flash", "low")

    assert "thinking_budget" in request_params(plan)["thinking_config"]
    (block,) = message_content(plan, [IMAGE])
    assert "media_resolution" not in block


def test_an_openrouter_model_with_a_documented_limit_is_clamped_to_it() -> None:
    unset = request_params(_plan("openrouter", "qwen/qwen3.5-flash-02-23", "max"))
    raised = request_params(
        _plan(
            "openrouter",
            "qwen/qwen3.5-flash-02-23",
            "low",
            requested_limit=128000,
        )
    )

    assert unset["max_tokens"] == 65536
    assert unset["extra_body"] == {"reasoning": {"effort": "xhigh"}}
    assert raised["max_tokens"] == 65536


def test_a_pinned_openrouter_provider_joins_the_reasoning_without_fallback() -> None:
    model = "qwen/qwen3.5-flash-02-23"
    routed = request_params(_plan("openrouter", model, "high"))
    pinned = request_params(
        _plan("openrouter", model, "high", openrouter_provider="deepinfra")
    )

    assert "provider" not in routed["extra_body"]
    assert pinned["extra_body"] == {
        "reasoning": routed["extra_body"]["reasoning"],
        "provider": {
            "order": ["deepinfra"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }
    other = request_params(_plan("openai", "gpt-5.6-luna", openrouter_provider="x"))
    assert "extra_body" not in other and "provider" not in other


def _sent_samplers(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the samplers of *params* at the top level and in ``extra_body``."""
    extra = params.get("extra_body", {})
    return (
        {name: params[name] for name in SAMPLERS if name in params},
        {name: extra[name] for name in SAMPLERS if name in extra},
    )


_ROUTED = {"top_k": 20, "min_p": 0.05, "repetition_penalty": 1.1}
_PENALTIES = {"presence_penalty": 1.5, "frequency_penalty": -0.5}


@pytest.mark.parametrize(
    ("provider", "model", "effort", "top", "body"),
    [
        ("openrouter", "qwen/qwen3.5-flash-02-23", "high", _PENALTIES, _ROUTED),
        ("custom", "qwen-vl", None, _PENALTIES, _ROUTED),
        ("google", "gemini-3.1-flash-lite", "low", {**_PENALTIES, "top_k": 20}, {}),
        ("anthropic", "claude-sonnet-4-5", "none", {"top_k": 20}, {}),
        ("openai", "gpt-4o", None, {}, {}),
    ],
)
def test_set_samplers_reach_the_providers_that_take_them(
    provider: str,
    model: str,
    effort: str | None,
    top: dict[str, Any],
    body: dict[str, Any],
) -> None:
    unset = request_params(_plan(provider, model, effort))
    params = request_params(_plan(provider, model, effort, **SAMPLERS))

    assert _sent_samplers(unset) == ({}, {})
    assert _sent_samplers(params) == (top, body)
    trimmed = {key: value for key, value in params.items() if key not in SAMPLERS}
    extra = {k: v for k, v in trimmed.pop("extra_body", {}).items() if k not in body}
    if extra:
        trimmed["extra_body"] = extra
    assert trimmed == unset


def test_a_set_penalty_replaces_the_zero_of_a_penalty_model() -> None:
    model = "openai/gpt-4o"
    unset = request_params(_plan("openrouter", model, None))
    params = request_params(_plan("openrouter", model, None, presence_penalty=1.5))

    assert (unset["presence_penalty"], unset["frequency_penalty"]) == (0.0, 0.0)
    assert (params["presence_penalty"], params["frequency_penalty"]) == (1.5, 0.0)


@pytest.mark.parametrize(
    ("model", "effort"), [("claude-sonnet-4-5", "low"), ("claude-haiku-5-5", "none")]
)
def test_anthropic_drops_top_k_with_thinking_or_without_a_temperature(
    model: str, effort: str
) -> None:
    params = request_params(_plan("anthropic", model, effort, **SAMPLERS))

    assert _sent_samplers(params) == ({}, {})


def test_a_fixed_sampling_gemini_model_gets_no_set_samplers() -> None:
    plan = _plan("google", "gemini-3.6-flash", "low", **SAMPLERS)

    assert not plan.caps.supports_temperature
    assert _sent_samplers(request_params(plan)) == ({}, {})
    config = batch_body(plan, "Answer.", [IMAGE], SCHEMA)["config"]
    assert _sent_samplers(config) == ({}, {})


def test_batch_bodies_carry_the_set_samplers_their_provider_takes() -> None:
    def body(provider: str, model: str, effort: str | None) -> dict[str, Any]:
        plan = _plan(provider, model, effort, **SAMPLERS)
        return batch_body(plan, "Answer.", [IMAGE], SCHEMA)

    config = body("google", "gemini-3.1-flash-lite", "low")["config"]
    assert _sent_samplers(config) == ({**_PENALTIES, "top_k": 20}, {})
    assert _sent_samplers(body("anthropic", "claude-sonnet-4-5", "none")) == (
        {"top_k": 20},
        {},
    )
    assert _sent_samplers(body("openai", "gpt-4o", None)) == ({}, {})


def test_an_openai_batch_body_carries_the_sync_parameters() -> None:
    plan = _plan(
        "openai",
        "gpt-5.6-luna",
        "minimal",
        service_tier="flex",
        verbosity="low",
        caching=PromptCaching(openai_retention="24h"),
    )

    params = request_params(plan)
    body = batch_body(plan, "Answer.", [IMAGE], SCHEMA)

    assert params["service_tier"] == "flex"
    assert body["service_tier"] == "auto"
    assert body["reasoning"] == params["reasoning"]
    assert body["max_output_tokens"] == params["max_output_tokens"]
    assert body["text"]["format"]["name"] == "answer"
    assert body["text"]["verbosity"] == "low"
    assert body["prompt_cache_options"] == {"ttl": "30m"}


def test_an_anthropic_batch_body_carries_thinking_effort_and_the_schema() -> None:
    plan = _plan("anthropic", "claude-haiku-5-5", "high", caching=PromptCaching())

    body = batch_body(plan, "Answer.", [IMAGE], SCHEMA)

    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "high"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}


def _with_context_image(context: str, page: str) -> list[Part]:
    """The user parts of a page request that sends a context image first."""
    return [
        Part(text="Context image:"),
        Part(data=context, mime="image/png", detail="high"),
        Part(text="The image:"),
        Part(data=page, mime="image/jpeg", detail="high"),
    ]


def test_a_gemini_request_over_its_limit_with_a_context_image_is_not_sent() -> None:
    plan = _plan("google", "gemini-3.1-flash-lite")
    # Each image fits the 20 MB image limit; together they exceed the request's.
    parts = _with_context_image("A" * 4_000_000, "B" * 16_000_000)

    with pytest.raises(RequestTooLargeError, match="Gemini API limit"):
        check_request_size(plan, "Answer.", parts)
    with pytest.raises(RequestTooLargeError):
        batch_body(plan, "Answer.", parts, SCHEMA)
    check_request_size(plan, "Answer.", parts[2:])


def test_a_context_image_over_the_anthropic_image_limit_is_not_sent() -> None:
    plan = _plan("anthropic", "claude-haiku-5-5")
    parts = _with_context_image("A" * (MAX_IMAGE_BYTES["anthropic"] + 1), "aGVsbG8=")

    with pytest.raises(RequestTooLargeError, match="image 1 of 2"):
        check_request_size(plan, "Answer.", parts)
    with pytest.raises(RequestTooLargeError):
        batch_body(plan, "Answer.", parts, SCHEMA)


@pytest.mark.parametrize(
    "provider,model",
    [
        ("openai", "gpt-5.6-luna"),
        ("anthropic", "claude-haiku-5-5"),
        ("google", "gemini-3.1-flash-lite"),
    ],
)
def test_a_request_within_the_limits_is_built_unchanged(
    provider: str, model: str
) -> None:
    plan = _plan(provider, model)
    # Near the per-image limit, within the Gemini request limit.
    near = "A" * (min(MAX_IMAGE_BYTES[provider], GOOGLE_REQUEST_BYTES) - 4_000)
    parts = _with_context_image("Y29udGV4dA==", near)

    check_request_size(plan, "Answer.", parts)
    body = batch_body(plan, "Answer.", parts, SCHEMA)

    unchecked = provider_requests._BODIES[provider](plan, "Answer.", parts, SCHEMA)
    assert json.dumps(body) == json.dumps(unchecked)
