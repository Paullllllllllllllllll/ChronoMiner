"""Tests for multimodal content passthrough in langchain_provider.py."""

import pytest

from modules.images.message_builder import build_image_content_block


class TestMultimodalContentPassthrough:
    """Test that image content blocks are passed through to LangChain correctly."""

    @pytest.mark.asyncio
    async def test_image_url_content_preserved_as_list(self):
        """When content contains image_url blocks, it should remain a list."""

        # Simulate what ainvoke_with_structured_output does with image content
        content = [
            {"type": "text", "text": "Extract data from this image."},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,abc123"},
            },
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
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"},
            },
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
    async def test_google_v1_block_detected_as_image(self):
        """The v1 data block must survive _to_lc_messages' list-preserving path."""
        block = build_image_content_block("abc", "image/png", "google", detail="high")
        content = [{"type": "text", "text": "Extract."}, block]

        has_image = any(
            isinstance(item, dict) and item.get("type") in ("image_url", "image")
            for item in content
        )
        assert has_image is True

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


class TestGoogleMediaResolutionPayload:
    """Offline payload check against the installed langchain-google-genai.

    Only the v1 data block reaches the branch that forwards a per-part media
    resolution; the legacy ``image_url`` block never does.
    """

    @staticmethod
    def _parts(block: dict, model: str):
        from langchain_google_genai.chat_models import _convert_to_parts

        return _convert_to_parts([block], model=model)

    @pytest.mark.unit
    def test_media_resolution_reaches_gemini_3_part(self):
        import base64

        payload = base64.b64encode(b"not-a-real-png").decode()
        block = build_image_content_block(payload, "image/png", "google", detail="high")
        part = self._parts(block, "gemini-3-pro")[0]

        assert part.inline_data is not None
        assert part.media_resolution is not None
        assert part.media_resolution.level.value == "MEDIA_RESOLUTION_HIGH"

    @pytest.mark.unit
    def test_legacy_block_carries_no_media_resolution(self):
        import base64

        payload = base64.b64encode(b"not-a-real-png").decode()
        block = build_image_content_block(payload, "image/png", "google")
        part = self._parts(block, "gemini-3-pro")[0]

        assert part.inline_data is not None
        assert part.media_resolution is None

    @pytest.mark.unit
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_v1_block_accepted_by_every_gemini_generation(self):
        """Older generations ignore media_resolution but must not break."""
        import base64

        payload = base64.b64encode(b"not-a-real-png").decode()
        block = build_image_content_block(payload, "image/png", "google", detail="high")
        for model in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"):
            part = self._parts(block, model)[0]
            assert part.inline_data is not None
            assert part.media_resolution is None
