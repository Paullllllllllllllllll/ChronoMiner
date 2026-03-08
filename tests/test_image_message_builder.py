"""Tests for modules/llm/image_message_builder.py."""

from modules.llm.image_message_builder import build_image_content_block


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
