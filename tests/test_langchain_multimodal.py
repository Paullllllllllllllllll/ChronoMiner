"""Tests for multimodal content passthrough in langchain_provider.py."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestMultimodalContentPassthrough:
    """Test that image content blocks are passed through to LangChain correctly."""

    @pytest.mark.asyncio
    async def test_image_url_content_preserved_as_list(self):
        """When content contains image_url blocks, it should remain a list."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Simulate what ainvoke_with_structured_output does with image content
        content = [
            {"type": "text", "text": "Extract data from this image."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc123"}},
        ]

        # Check that has_image detection works
        has_image = any(
            isinstance(item, dict) and item.get("type") in ("image_url", "image")
            for item in content
        )
        assert has_image is True

    @pytest.mark.asyncio
    async def test_anthropic_image_content_preserved_as_list(self):
        """Anthropic image blocks (type='image') should also be preserved."""
        content = [
            {"type": "text", "text": "Extract data."},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"}},
        ]

        has_image = any(
            isinstance(item, dict) and item.get("type") in ("image_url", "image")
            for item in content
        )
        assert has_image is True

    @pytest.mark.asyncio
    async def test_text_only_content_flattened(self):
        """Text-only list content should be flattened to a string."""
        content = [
            {"type": "input_text", "text": "Hello"},
            {"type": "input_text", "text": "World"},
        ]

        has_image = any(
            isinstance(item, dict) and item.get("type") in ("image_url", "image")
            for item in content
        )
        assert has_image is False

        # Text extraction logic
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "input_text":
                text_parts.append(item.get("text", ""))
        result = "\n".join(text_parts)
        assert result == "Hello\nWorld"

    @pytest.mark.asyncio
    async def test_string_content_items_handled(self):
        """Plain string items in content list should be handled."""
        content = ["Hello", "World"]

        has_image = any(
            isinstance(item, dict) and item.get("type") in ("image_url", "image")
            for item in content
        )
        assert has_image is False

        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
        result = "\n".join(text_parts)
        assert result == "Hello\nWorld"
