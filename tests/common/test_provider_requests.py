"""The shared request builder: one set of parameters per provider.

Synchronous calls send :func:`request_params` as invoke kwargs and batch
bodies carry the same parameters, so both paths send the same thinking,
effort, output limit, schema and media resolution.
"""

from __future__ import annotations

from typing import Any

import pytest

from chronominer.common.capabilities import resolve_capabilities
from chronominer.common.capability_table import TABLE
from chronominer.common.provider_requests import (
    Part,
    RequestPlan,
    batch_body,
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
