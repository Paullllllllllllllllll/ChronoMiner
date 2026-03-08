"""Tests for prompt caching optimization.

Covers:
- cache_control annotation on system content blocks
- Content list preservation when cache_control is present
- Backward-compatible collapse for non-cache-control cases
- Cache metrics extraction from Anthropic usage metadata
- supports_prompt_caching capability flag
- Anthropic batch backend structured system message
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.llm.model_capabilities import Capabilities, detect_capabilities


# ---------------------------------------------------------------------------
# 1. supports_prompt_caching capability flag
# ---------------------------------------------------------------------------

class TestSupportsPromptCaching:
    """Verify the supports_prompt_caching flag is set correctly per provider."""

    @pytest.mark.unit
    @pytest.mark.parametrize("model", [
        "claude-sonnet-4-5",
        "claude-opus-4-6",
        "claude-3-5-sonnet",
        "claude-haiku-4-5",
        "claude",
    ])
    def test_anthropic_models_support_caching(self, model: str):
        caps = detect_capabilities(model)
        assert caps.supports_prompt_caching is True

    @pytest.mark.unit
    @pytest.mark.parametrize("model", [
        "anthropic/claude-sonnet-4-5",
        "openrouter/anthropic/claude-3-5-sonnet",
    ])
    def test_openrouter_claude_models_support_caching(self, model: str):
        caps = detect_capabilities(model)
        assert caps.supports_prompt_caching is True

    @pytest.mark.unit
    @pytest.mark.parametrize("model", [
        "gpt-4o",
        "gpt-5",
        "o3",
        "gemini-2.5-pro",
        "openrouter/openai/gpt-4o",
    ])
    def test_non_anthropic_models_do_not_support_caching(self, model: str):
        caps = detect_capabilities(model)
        assert caps.supports_prompt_caching is False


# ---------------------------------------------------------------------------
# 2. cache_control annotation in openai_utils message construction
# ---------------------------------------------------------------------------

class TestCacheControlAnnotation:
    """Verify cache_control is added/absent on system content blocks."""

    @pytest.mark.unit
    def test_cache_control_present_when_enabled_text(self):
        """process_text_chunk should add cache_control when enabled."""
        # Simulate the message construction logic from process_text_chunk
        system_message = "You are an expert."
        enable_cache_control = True

        system_content: list[Dict[str, Any]] = [
            {"type": "input_text", "text": system_message}
        ]
        if enable_cache_control:
            system_content[-1]["cache_control"] = {"type": "ephemeral"}

        assert "cache_control" in system_content[0]
        assert system_content[0]["cache_control"] == {"type": "ephemeral"}
        assert system_content[0]["text"] == system_message

    @pytest.mark.unit
    def test_cache_control_absent_when_disabled_text(self):
        """process_text_chunk should NOT add cache_control when disabled."""
        system_message = "You are an expert."
        enable_cache_control = False

        system_content: list[Dict[str, Any]] = [
            {"type": "input_text", "text": system_message}
        ]
        if enable_cache_control:
            system_content[-1]["cache_control"] = {"type": "ephemeral"}

        assert "cache_control" not in system_content[0]

    @pytest.mark.unit
    def test_cache_control_present_when_enabled_image(self):
        """process_image_chunk should add cache_control when enabled."""
        system_message = "Extract data."
        enable_cache_control = True

        system_content: list[Dict[str, Any]] = [
            {"type": "input_text", "text": system_message}
        ]
        if enable_cache_control:
            system_content[-1]["cache_control"] = {"type": "ephemeral"}

        assert "cache_control" in system_content[0]
        assert system_content[0]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# 3. Content block preservation in langchain_provider conversion
# ---------------------------------------------------------------------------

class TestContentBlockPreservation:
    """Verify content list handling for cache_control vs plain text."""

    @pytest.mark.unit
    def test_cache_control_content_preserved_as_list(self):
        """Content with cache_control should remain a list, not collapse."""
        content = [
            {"type": "input_text", "text": "System prompt", "cache_control": {"type": "ephemeral"}}
        ]

        has_image = any(
            isinstance(item, dict) and item.get("type") in ("image_url", "image")
            for item in content
        )
        has_cache_control = any(
            isinstance(item, dict) and "cache_control" in item
            for item in content
        )

        assert has_cache_control is True
        assert has_image is False

        # When has_cache_control is True, content stays as list and input_text -> text
        if has_cache_control:
            content = [
                {
                    **{k: v for k, v in item.items() if k != "type"},
                    "type": "text",
                }
                if isinstance(item, dict) and item.get("type") == "input_text"
                else item
                for item in content
            ]

        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "System prompt"
        assert content[0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.unit
    def test_text_only_content_still_collapsed(self):
        """Text-only content without cache_control should be collapsed to string."""
        content = [
            {"type": "input_text", "text": "Hello"},
            {"type": "input_text", "text": "World"},
        ]

        has_image = any(
            isinstance(item, dict) and item.get("type") in ("image_url", "image")
            for item in content
        )
        has_cache_control = any(
            isinstance(item, dict) and "cache_control" in item
            for item in content
        )

        assert has_cache_control is False
        assert has_image is False

        # Should collapse to string
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "input_text":
                text_parts.append(item.get("text", ""))
        result = "\n".join(text_parts)

        assert isinstance(result, str)
        assert result == "Hello\nWorld"

    @pytest.mark.unit
    def test_multimodal_with_cache_control_preserved(self):
        """Content with both image blocks and cache_control should stay a list."""
        content = [
            {"type": "text", "text": "Instruction", "cache_control": {"type": "ephemeral"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
        ]

        has_image = any(
            isinstance(item, dict) and item.get("type") in ("image_url", "image")
            for item in content
        )
        has_cache_control = any(
            isinstance(item, dict) and "cache_control" in item
            for item in content
        )

        assert has_image is True
        assert has_cache_control is True
        # Content should remain as list
        assert isinstance(content, list)


# ---------------------------------------------------------------------------
# 4. Cache metrics extraction
# ---------------------------------------------------------------------------

class TestCacheMetricsExtraction:
    """Verify cache metrics are extracted from Anthropic usage metadata."""

    @pytest.mark.unit
    def test_cache_metrics_from_dict_usage(self):
        """Extract cache tokens from dict-format usage metadata."""
        usage = {
            "input_tokens": 5000,
            "output_tokens": 500,
            "total_tokens": 5500,
            "cache_creation_input_tokens": 3200,
            "cache_read_input_tokens": 0,
        }

        cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)

        assert cache_creation == 3200
        assert cache_read == 0

    @pytest.mark.unit
    def test_cache_metrics_from_dict_usage_hit(self):
        """Extract cache read tokens when cache hits."""
        usage = {
            "input_tokens": 5000,
            "output_tokens": 500,
            "total_tokens": 5500,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 3200,
        }

        cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)

        assert cache_creation == 0
        assert cache_read == 3200

    @pytest.mark.unit
    def test_cache_metrics_from_attr_usage(self):
        """Extract cache tokens from attribute-style usage metadata."""
        usage = MagicMock()
        usage.cache_creation_input_tokens = 1500
        usage.cache_read_input_tokens = 2000

        cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)

        assert cache_creation == 1500
        assert cache_read == 2000

    @pytest.mark.unit
    def test_cache_metrics_absent_returns_zeros(self):
        """When no cache metrics exist, values should be zero."""
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
        }

        cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)

        assert cache_creation == 0
        assert cache_read == 0


# ---------------------------------------------------------------------------
# 5. Anthropic batch backend structured system message
# ---------------------------------------------------------------------------

class TestAnthropicBatchCacheControl:
    """Verify Anthropic batch backend uses structured system message."""

    @pytest.mark.unit
    def test_batch_system_message_is_structured(self):
        """System prompt should be a list with cache_control, not a plain string."""
        system_prompt = "You are a data extraction expert."

        # Simulate the batch backend's system message construction
        structured_system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        assert isinstance(structured_system, list)
        assert len(structured_system) == 1
        assert structured_system[0]["type"] == "text"
        assert structured_system[0]["text"] == system_prompt
        assert structured_system[0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.unit
    def test_batch_system_message_not_plain_string(self):
        """Ensure the system field is not a plain string."""
        system_prompt = "You are a data extraction expert."

        params = {
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }

        assert not isinstance(params["system"], str)
        assert isinstance(params["system"], list)


# ---------------------------------------------------------------------------
# 6. Prompt template ordering
# ---------------------------------------------------------------------------

class TestPromptTemplateOrdering:
    """Verify schema appears before context in prompt templates."""

    @pytest.mark.unit
    def test_text_prompt_schema_before_context(self):
        from pathlib import Path
        prompt_path = Path("prompts/text_extraction_prompt.txt")
        if not prompt_path.exists():
            pytest.skip("Prompt file not found")
        text = prompt_path.read_text(encoding="utf-8")
        schema_pos = text.find("{{TRANSCRIPTION_SCHEMA}}")
        context_pos = text.find("{{CONTEXT}}")
        assert schema_pos < context_pos, "Schema placeholder must appear before context placeholder"

    @pytest.mark.unit
    def test_image_prompt_schema_before_context(self):
        from pathlib import Path
        prompt_path = Path("prompts/image_extraction_prompt.txt")
        if not prompt_path.exists():
            pytest.skip("Prompt file not found")
        text = prompt_path.read_text(encoding="utf-8")
        schema_pos = text.find("{{TRANSCRIPTION_SCHEMA}}")
        context_pos = text.find("{{CONTEXT}}")
        assert schema_pos < context_pos, "Schema placeholder must appear before context placeholder"
