"""Tests for model capabilities detection and gating."""

import pytest
from modules.config.capabilities import (
    Capabilities,
    detect_provider,
    detect_capabilities,
)


class TestDetectProvider:
    """Test provider detection from model names."""
    
    @pytest.mark.unit
    def test_detect_openai_gpt_models(self):
        """Test OpenAI GPT model detection."""
        assert detect_provider("gpt-4o") == "openai"
        assert detect_provider("gpt-4o-mini") == "openai"
        assert detect_provider("gpt-4.1") == "openai"
        assert detect_provider("gpt-5") == "openai"
        assert detect_provider("gpt-5.1") == "openai"
        assert detect_provider("gpt-5.3-chat-latest") == "openai"
        assert detect_provider("gpt-5.3-codex") == "openai"
        assert detect_provider("gpt-5.4") == "openai"
        assert detect_provider("gpt-5.4-pro") == "openai"
    
    @pytest.mark.unit
    def test_detect_openai_reasoning_models(self):
        """Test OpenAI o-series model detection."""
        assert detect_provider("o1") == "openai"
        assert detect_provider("o1-mini") == "openai"
        assert detect_provider("o3") == "openai"
        assert detect_provider("o3-mini") == "openai"
        assert detect_provider("o3-pro") == "openai"
        assert detect_provider("o4-mini") == "openai"
    
    @pytest.mark.unit
    def test_detect_anthropic_models(self):
        """Test Anthropic Claude model detection."""
        assert detect_provider("claude-3-5-sonnet-20241022") == "anthropic"
        assert detect_provider("claude-3.5-sonnet") == "anthropic"
        assert detect_provider("claude-opus-4") == "anthropic"
        assert detect_provider("claude-sonnet-4") == "anthropic"
        assert detect_provider("claude-haiku-4.5") == "anthropic"
    
    @pytest.mark.unit
    def test_detect_google_models(self):
        """Test Google Gemini model detection."""
        assert detect_provider("gemini-2.0-flash") == "google"
        assert detect_provider("gemini-1.5-pro") == "google"
        assert detect_provider("gemini-pro") == "google"
    
    @pytest.mark.unit
    def test_detect_openrouter_models(self):
        """Test OpenRouter model detection."""
        assert detect_provider("openrouter/anthropic/claude-3.5-sonnet") == "openrouter"
        assert detect_provider("anthropic/claude-3.5-sonnet") == "openrouter"
        assert detect_provider("google/gemini-pro") == "openrouter"
    
    @pytest.mark.unit
    def test_detect_unknown_provider(self):
        """Test unknown provider detection."""
        assert detect_provider("unknown-model") == "unknown"
        assert detect_provider("some-random-model") == "unknown"
    
    @pytest.mark.unit
    def test_detect_provider_case_insensitive(self):
        """Test that provider detection is case-insensitive."""
        assert detect_provider("GPT-4o") == "openai"
        assert detect_provider("CLAUDE-3.5-sonnet") == "anthropic"
        assert detect_provider("Gemini-2.0-flash") == "google"


class TestDetectCapabilities:
    """Test capability detection for different models."""
    
    @pytest.mark.unit
    def test_gpt4o_capabilities(self):
        """Test GPT-4o capabilities."""
        caps = detect_capabilities("gpt-4o")
        
        assert caps.model == "gpt-4o"
        assert caps.family == "gpt-4o"
        assert caps.provider == "openai"
        assert caps.is_reasoning_model is False
        assert caps.supports_sampler_controls is True
        assert caps.supports_structured_outputs is True
        assert caps.supports_image_input is True
        assert caps.supports_function_calling is True
    
    @pytest.mark.unit
    def test_gpt41_capabilities(self):
        """Test GPT-4.1 capabilities."""
        caps = detect_capabilities("gpt-4.1")
        
        assert caps.family == "gpt-4.1"
        assert caps.is_reasoning_model is False
        assert caps.supports_sampler_controls is True
        assert caps.supports_image_input is True
    
    @pytest.mark.unit
    def test_gpt5_capabilities(self):
        """Test GPT-5 reasoning model capabilities."""
        caps = detect_capabilities("gpt-5")
        
        assert caps.family == "gpt-5"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_sampler_controls is False
        assert caps.max_context_tokens == 400000
    
    @pytest.mark.unit
    def test_gpt54_capabilities(self):
        """Test GPT-5.4 capabilities."""
        caps = detect_capabilities("gpt-5.4")

        assert caps.family == "gpt-5.4"
        assert caps.provider == "openai"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_sampler_controls is False
        assert caps.supports_structured_outputs is True
        assert caps.max_context_tokens == 1050000

    @pytest.mark.unit
    def test_gpt54_pro_capabilities(self):
        """Test GPT-5.4-pro capabilities (no structured outputs)."""
        caps = detect_capabilities("gpt-5.4-pro")

        assert caps.family == "gpt-5.4-pro"
        assert caps.provider == "openai"
        assert caps.is_reasoning_model is True
        assert caps.supports_structured_outputs is False
        assert caps.max_context_tokens == 1050000

    @pytest.mark.unit
    def test_gpt53_chat_capabilities(self):
        """Test GPT-5.3-chat-latest is a standard (non-reasoning) model."""
        caps = detect_capabilities("gpt-5.3-chat-latest")

        assert caps.family == "gpt-5.3-chat"
        assert caps.provider == "openai"
        assert caps.is_reasoning_model is False
        assert caps.supports_sampler_controls is True
        assert caps.supports_structured_outputs is True
        assert caps.max_context_tokens == 128000

    @pytest.mark.unit
    def test_gpt53_codex_capabilities(self):
        """Test GPT-5.3-codex is a reasoning model with 400K context."""
        caps = detect_capabilities("gpt-5.3-codex")

        assert caps.family == "gpt-5.3-codex"
        assert caps.provider == "openai"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_sampler_controls is False
        assert caps.max_context_tokens == 400000

    @pytest.mark.unit
    def test_gpt51_capabilities(self):
        """Test GPT-5.1 capabilities."""
        caps = detect_capabilities("gpt-5.1")

        assert caps.family == "gpt-5.1"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_sampler_controls is False
    
    @pytest.mark.unit
    def test_o1_capabilities(self):
        """Test o1 reasoning model capabilities."""
        caps = detect_capabilities("o1")
        
        assert caps.family == "o1"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is False
        assert caps.supports_sampler_controls is False
        assert caps.supports_image_input is True
    
    @pytest.mark.unit
    def test_o1_mini_capabilities(self):
        """Test o1-mini reasoning model capabilities."""
        caps = detect_capabilities("o1-mini")
        
        assert caps.family == "o1-mini"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is False
        assert caps.supports_sampler_controls is False
        assert caps.supports_image_input is False
        assert caps.supports_function_calling is False
    
    @pytest.mark.unit
    def test_o3_capabilities(self):
        """Test o3 reasoning model capabilities."""
        caps = detect_capabilities("o3")
        
        assert caps.family == "o3"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_sampler_controls is False
        assert caps.supports_image_input is True
    
    @pytest.mark.unit
    def test_o3_mini_capabilities(self):
        """Test o3-mini capabilities."""
        caps = detect_capabilities("o3-mini")
        
        assert caps.family == "o3-mini"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_image_input is False
        assert caps.supports_structured_outputs is True
    
    @pytest.mark.unit
    def test_o3_pro_capabilities(self):
        """Test o3-pro capabilities."""
        caps = detect_capabilities("o3-pro")
        
        assert caps.family == "o3-pro"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_image_input is True
    
    @pytest.mark.unit
    def test_o4_mini_capabilities(self):
        """Test o4-mini capabilities."""
        caps = detect_capabilities("o4-mini")

        assert caps.family == "o4-mini"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_image_input is True
        assert caps.supports_structured_outputs is True

    @pytest.mark.unit
    def test_o4_capabilities(self):
        """Test o4 (non-mini) capabilities — separate family from o4-mini."""
        caps = detect_capabilities("o4")

        assert caps.family == "o4"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_image_input is True

    @pytest.mark.unit
    def test_claude_opus_47_capabilities(self):
        """Test Claude Opus 4.7 capabilities — 1M context."""
        caps = detect_capabilities("claude-opus-4-7")

        assert caps.family == "claude-opus-4.7"
        assert caps.provider == "anthropic"
        assert caps.supports_image_input is True
        assert caps.supports_prompt_caching is True
        assert caps.max_context_tokens == 1000000

    @pytest.mark.unit
    def test_claude_opus_47_dot_notation(self):
        """Test Claude Opus 4.7 with dot notation."""
        caps = detect_capabilities("claude-opus-4.7")

        assert caps.family == "claude-opus-4.7"
        assert caps.max_context_tokens == 1000000

    @pytest.mark.unit
    def test_gpt41_mini_capabilities(self):
        """Test GPT-4.1-mini capabilities — 1.05M context, non-reasoning."""
        caps = detect_capabilities("gpt-4.1-mini")

        assert caps.family == "gpt-4.1-mini"
        assert caps.provider == "openai"
        assert caps.is_reasoning_model is False
        assert caps.supports_sampler_controls is True
        assert caps.supports_image_input is True
        assert caps.max_context_tokens == 1050000

    @pytest.mark.unit
    def test_gpt41_nano_capabilities(self):
        """Test GPT-4.1-nano capabilities — 1.05M context, non-reasoning."""
        caps = detect_capabilities("gpt-4.1-nano")

        assert caps.family == "gpt-4.1-nano"
        assert caps.provider == "openai"
        assert caps.is_reasoning_model is False
        assert caps.max_context_tokens == 1050000

    @pytest.mark.unit
    def test_gpt41_context_window(self):
        """Test GPT-4.1 has correct 1.05M context window."""
        caps = detect_capabilities("gpt-4.1")

        assert caps.max_context_tokens == 1050000

    @pytest.mark.unit
    def test_gemini_31_pro_capabilities(self):
        """Test Gemini 3.1 Pro capabilities."""
        caps = detect_capabilities("gemini-3.1-pro-preview")

        assert caps.family == "gemini-3.1-pro"
        assert caps.provider == "google"
        assert caps.is_reasoning_model is True
        assert caps.max_context_tokens == 1048576

    @pytest.mark.unit
    def test_gemini_31_flash_lite_capabilities(self):
        """Test Gemini 3.1 Flash-Lite capabilities."""
        caps = detect_capabilities("gemini-3.1-flash-lite-preview")

        assert caps.family == "gemini-3.1-flash-lite"
        assert caps.provider == "google"
        assert caps.max_context_tokens == 1048576

    @pytest.mark.unit
    def test_claude_35_sonnet_capabilities(self):
        """Test Claude 3.5 Sonnet capabilities."""
        caps = detect_capabilities("claude-3-5-sonnet-20241022")
        
        assert caps.family == "claude-3.5-sonnet"
        assert caps.provider == "anthropic"
        assert caps.is_reasoning_model is False
        assert caps.supports_sampler_controls is True
        assert caps.supports_image_input is True
        assert caps.max_context_tokens == 200000
    
    @pytest.mark.unit
    def test_claude_opus_4_capabilities(self):
        """Test Claude Opus 4 capabilities."""
        caps = detect_capabilities("claude-opus-4")
        
        assert caps.family == "claude-opus-4"
        assert caps.provider == "anthropic"
        assert caps.supports_structured_outputs is True
    
    @pytest.mark.unit
    def test_claude_sonnet_4_capabilities(self):
        """Test Claude Sonnet 4 capabilities."""
        caps = detect_capabilities("claude-sonnet-4")
        
        assert caps.family == "claude-sonnet-4"
        assert caps.provider == "anthropic"
    
    @pytest.mark.unit
    def test_gemini_20_flash_capabilities(self):
        """Test Gemini 2.0 Flash capabilities."""
        caps = detect_capabilities("gemini-2.0-flash")
        
        assert caps.provider == "google"
        assert caps.supports_image_input is True
    
    @pytest.mark.unit
    def test_gemini_15_pro_capabilities(self):
        """Test Gemini 1.5 Pro capabilities."""
        caps = detect_capabilities("gemini-1.5-pro")
        
        assert caps.provider == "google"
        assert caps.supports_image_input is True


class TestCustomProviderCapabilities:
    """Test capability detection for custom endpoints."""

    @pytest.mark.unit
    def test_custom_provider_returns_custom_caps(self):
        """Custom provider should return _CUSTOM_BASE capabilities."""
        caps = detect_capabilities("rednote-hilab/dots.mocr", provider="custom")

        assert caps.provider == "custom"
        assert caps.family == "custom"
        assert caps.model == "rednote-hilab/dots.mocr"
        assert caps.supports_structured_outputs is True
        assert caps.supports_image_input is True
        assert caps.supports_function_calling is False
        assert caps.is_reasoning_model is False
        assert caps.supports_sampler_controls is True

    @pytest.mark.unit
    def test_custom_provider_overrides_slash_detection(self):
        """A model name with '/' should NOT be detected as OpenRouter
        when provider='custom' is explicitly given."""
        caps_custom = detect_capabilities("org/model", provider="custom")
        caps_auto = detect_capabilities("org/model")

        assert caps_custom.provider == "custom"
        assert caps_auto.provider == "openrouter"

    @pytest.mark.unit
    def test_detect_capabilities_backward_compat(self):
        """Calling without provider kwarg should produce identical results
        for established models."""
        for model in ("gpt-4o", "claude-3-5-sonnet-20241022", "gemini-2.0-flash"):
            caps_old = detect_capabilities(model)
            caps_new = detect_capabilities(model, provider=None)
            assert caps_old == caps_new


class TestCapabilitiesDataclass:
    """Test the Capabilities dataclass."""
    
    @pytest.mark.unit
    def test_capabilities_defaults(self):
        """Test default capability values."""
        caps = Capabilities(model="test-model", family="test")
        
        assert caps.provider == "openai"
        assert caps.supports_responses_api is True
        assert caps.supports_chat_completions is True
        assert caps.api_preference == "langchain"
        assert caps.is_reasoning_model is False
        assert caps.supports_reasoning_effort is False
        assert caps.supports_developer_messages is True
        assert caps.supports_image_input is False
        assert caps.supports_image_detail is False
        assert caps.default_ocr_detail == "high"
        assert caps.supports_structured_outputs is True
        assert caps.supports_function_calling is True
        assert caps.supports_sampler_controls is True
        assert caps.max_context_tokens == 128000
    
    @pytest.mark.unit
    def test_capabilities_immutable(self):
        """Test that Capabilities is immutable (frozen)."""
        caps = Capabilities(model="test", family="test")
        
        with pytest.raises(AttributeError):
            caps.model = "changed"
    
    @pytest.mark.unit
    def test_capabilities_custom_values(self):
        """Test custom capability values."""
        caps = Capabilities(
            model="custom-model",
            family="custom",
            provider="anthropic",
            is_reasoning_model=True,
            supports_sampler_controls=False,
            max_context_tokens=200000,
        )
        
        assert caps.model == "custom-model"
        assert caps.provider == "anthropic"
        assert caps.is_reasoning_model is True
        assert caps.supports_sampler_controls is False
        assert caps.max_context_tokens == 200000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
