"""Capability edge cases the recorded family table cannot express.

Per-model values live in ``data/capability_expectations.json`` and are checked
by test_capability_families.py.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from chronominer.llm.capabilities import (
    Capabilities,
    CapabilityError,
    custom_capabilities,
    detect_capabilities,
    ensure_image_support,
)
from chronominer.spec import MODEL_SUGGESTIONS


@pytest.mark.parametrize(
    ("model", "family", "limit"),
    [
        ("claude-opus-4-1-20250805", "claude-opus-4.1", 32000),
        ("claude-sonnet-4-5-20250929", "claude-sonnet-4.5", 64000),
        ("claude-opus-4-6", "claude-opus-4.6", 128000),
    ],
)
def test_claude_4_rules_keep_specific_versions_first(
    model: str, family: str, limit: int
) -> None:
    """Sonnet 4 and Opus 4 have their own rows; 4.1 to 4.6 still match first."""
    caps = detect_capabilities(model)
    assert caps.family == family
    assert caps.max_output_tokens == limit


@pytest.mark.parametrize(
    ("model", "family", "limit"),
    [
        ("claude-sonnet-5-5", "claude-sonnet-5.5", 128000),
        ("claude-sonnet-5.5", "claude-sonnet-5.5", 128000),
        ("claude-sonnet-5", "claude-sonnet-5", 128000),
        ("claude-haiku-5-5", "claude-haiku-5.5", 128000),
        ("claude-haiku-5.5-2026-01-01", "claude-haiku-5.5", 128000),
    ],
)
def test_claude_5_5_ids_resolve_to_their_own_rows(
    model: str, family: str, limit: int
) -> None:
    caps = detect_capabilities(model)
    assert (caps.provider_name, caps.family) == ("anthropic", family)
    assert caps.supports_vision
    assert caps.supports_structured_output
    assert caps.supports_reasoning_effort
    assert caps.uses_adaptive_thinking
    assert caps.image_high_res_tier
    assert caps.max_context_tokens == 1000000
    assert caps.max_output_tokens == limit


def test_model_suggestions_resolve_to_known_rows() -> None:
    for provider, names in MODEL_SUGGESTIONS.items():
        for name in names:
            assert detect_capabilities(name, provider).family != "unknown", name


@pytest.mark.parametrize("model", ["gpt-6.1-sol", "gpt-6.1-sol-2026-01-01"])
def test_gpt_6_1_sol_takes_the_gpt_6_sol_profile(model: str) -> None:
    caps = detect_capabilities(model)
    sol = detect_capabilities("gpt-6-sol")
    assert caps.family == "gpt-6.1-sol"
    assert sol.family == "gpt-6-sol"
    assert caps.max_output_tokens == 128000
    assert caps.max_context_tokens == 1050000
    assert caps.supports_original_image_detail
    assert caps.image_original_patch_cap_30k
    assert caps.supports_reasoning_effort
    assert caps.supports_structured_output
    # GPT-6.1 Sol takes no none effort; GPT-6 Sol does.
    assert caps.reasoning_efforts == sol.reasoning_efforts[1:]
    same = replace(
        caps,
        model_name="gpt-6-sol",
        family="gpt-6-sol",
        reasoning_efforts=sol.reasoning_efforts,
    )
    assert same == sol


@pytest.mark.parametrize("model", ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4-pro"])
def test_longer_prefix_wins_over_shorter_family(model: str) -> None:
    """A variant id matches its own family, not the 'gpt-5.4' prefix."""
    assert detect_capabilities(model).family == model
    assert detect_capabilities("gpt-5.4").family == "gpt-5.4"


def test_a_provider_override_keeps_the_family_and_names_the_provider() -> None:
    caps = detect_capabilities("claude-sonnet-4-5", "openrouter")

    assert caps.family == "claude-sonnet-4.5"
    assert caps.provider_name == "anthropic"


def test_routed_rules_need_a_vendor_id() -> None:
    assert detect_capabilities("deepseek/deepseek-r1").family == "openrouter-deepseek"
    assert detect_capabilities("deepseek-r1").family == "unknown"
    assert detect_capabilities("org/model", "custom").family == "unknown"


def test_the_table_has_no_audio_flags() -> None:
    """Speech-to-text ids resolve to a text-only profile without audio flags."""
    whisper = detect_capabilities("whisper-1")

    assert whisper.family == "openai-audio"
    assert not whisper.supports_vision
    assert whisper.extras == {}
    assert detect_capabilities("gemini-3.6-flash").extras == {}


def test_custom_capabilities_carry_the_declared_support() -> None:
    caps = custom_capabilities("org/m", supports_vision=False, supports_schema=True)

    assert caps.provider_name == "custom"
    assert caps.family == "custom"
    assert not caps.supports_vision
    assert caps.supports_structured_output and caps.supports_json_mode


class TestCapabilityErrors:
    """ensure_image_support."""

    def test_image_support_raises_for_a_text_model(self) -> None:
        caps = Capabilities(provider_name="openai", model_name="text-only")
        with pytest.raises(CapabilityError, match="does not support image inputs"):
            ensure_image_support("text-only", caps)

    def test_image_support_passes_for_a_vision_model(self) -> None:
        ensure_image_support("gpt-5.6-luna", detect_capabilities("gpt-5.6-luna"))
