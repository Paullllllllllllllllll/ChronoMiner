"""Tests for modules/llm/image_message_builder.py."""

from modules.images.message_builder import build_image_content_block


class TestBuildImageContentBlockOpenAI:
    def test_openai_basic(self):
        result = build_image_content_block(
            image_base64="abc123",
            mime_type="image/jpeg",
            provider="openai",
        )
        assert result["type"] == "image_url"
        assert "data:image/jpeg;base64,abc123" in result["image_url"]["url"]

    def test_openai_with_detail(self):
        result = build_image_content_block(
            image_base64="abc123",
            mime_type="image/png",
            provider="openai",
            detail="high",
            supports_image_detail=True,
        )
        assert result["image_url"]["detail"] == "high"

    def test_openai_detail_not_set_without_support(self):
        result = build_image_content_block(
            image_base64="abc123",
            mime_type="image/png",
            provider="openai",
            detail="high",
            supports_image_detail=False,
        )
        assert "detail" not in result["image_url"]

    def test_original_downgraded_on_chat_completions_route(self):
        """Chat Completions accepts auto/low/high only — 'original' 400s."""
        result = build_image_content_block(
            image_base64="abc123",
            mime_type="image/png",
            provider="openai",
            detail="original",
            supports_image_detail=True,
            supports_original_detail=False,
        )
        assert result["image_url"]["detail"] == "high"

    def test_original_preserved_on_responses_route(self):
        result = build_image_content_block(
            image_base64="abc123",
            mime_type="image/png",
            provider="openai",
            detail="original",
            supports_image_detail=True,
            supports_original_detail=True,
        )
        assert result["image_url"]["detail"] == "original"

    def test_other_details_unaffected_by_original_flag(self):
        for detail in ("low", "high", "auto"):
            result = build_image_content_block(
                image_base64="abc123",
                mime_type="image/png",
                provider="openai",
                detail=detail,
                supports_image_detail=True,
                supports_original_detail=False,
            )
            assert result["image_url"]["detail"] == detail

    def test_openrouter_original_downgraded(self):
        """OpenRouter is Chat-Completions-compatible: never 'original'."""
        result = build_image_content_block(
            image_base64="abc123",
            mime_type="image/jpeg",
            provider="openrouter",
            detail="original",
            supports_image_detail=True,
        )
        assert result["image_url"]["detail"] == "high"


class TestBuildImageContentBlockAnthropic:
    def test_anthropic_format(self):
        result = build_image_content_block(
            image_base64="abc123",
            mime_type="image/jpeg",
            provider="anthropic",
        )
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/jpeg"
        assert result["source"]["data"] == "abc123"

    def test_anthropic_no_detail(self):
        result = build_image_content_block(
            image_base64="abc123",
            mime_type="image/jpeg",
            provider="anthropic",
            detail="high",
            supports_image_detail=True,
        )
        # Anthropic format doesn't use detail
        assert "detail" not in result
        assert "image_url" not in result


class TestBuildImageContentBlockGoogle:
    def test_google_format(self):
        result = build_image_content_block(
            image_base64="abc123",
            mime_type="image/png",
            provider="google",
        )
        assert result["type"] == "image_url"
        # Google uses data URL directly (not nested dict)
        assert result["image_url"] == "data:image/png;base64,abc123"

    def test_google_media_resolution_uses_v1_data_block(self):
        """Only the v1 data block carries per-part media_resolution."""
        result = build_image_content_block(
            image_base64="abc123",
            mime_type="image/png",
            provider="google",
            detail="high",
        )
        assert result == {
            "type": "image",
            "base64": "abc123",
            "mime_type": "image/png",
            "media_resolution": "MEDIA_RESOLUTION_HIGH",
        }

    def test_google_media_resolution_levels_normalized(self):
        expected = {
            "low": "MEDIA_RESOLUTION_LOW",
            "medium": "MEDIA_RESOLUTION_MEDIUM",
            "high": "MEDIA_RESOLUTION_HIGH",
            "ultra_high": "MEDIA_RESOLUTION_ULTRA_HIGH",
            "MEDIA_RESOLUTION_HIGH": "MEDIA_RESOLUTION_HIGH",
        }
        for configured, canonical in expected.items():
            result = build_image_content_block(
                image_base64="abc123",
                mime_type="image/png",
                provider="google",
                detail=configured,
            )
            assert result["media_resolution"] == canonical

    def test_google_unknown_detail_keeps_legacy_block(self):
        """An unmappable value must not become an invalid enum member."""
        result = build_image_content_block(
            image_base64="abc123",
            mime_type="image/png",
            provider="google",
            detail="original",
        )
        assert result["type"] == "image_url"
        assert result["image_url"] == "data:image/png;base64,abc123"


class TestBuildImageContentBlockOpenRouter:
    def test_openrouter_uses_openai_format(self):
        result = build_image_content_block(
            image_base64="abc123",
            mime_type="image/jpeg",
            provider="openrouter",
            detail="high",
            supports_image_detail=True,
        )
        assert result["type"] == "image_url"
        assert result["image_url"]["detail"] == "high"
        assert "data:image/jpeg;base64,abc123" in result["image_url"]["url"]
