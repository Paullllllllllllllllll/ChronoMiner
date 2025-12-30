"""Tests for LangChain-based provider abstraction."""

import pytest
from unittest.mock import patch, MagicMock
from modules.llm.langchain_provider import (
    ProviderConfig,
    _normalize_schema_for_anthropic,
    _compute_openrouter_reasoning_max_tokens,
    _build_openrouter_reasoning_payload,
)


class TestProviderConfig:
    """Test ProviderConfig dataclass and factory."""
    
    @pytest.mark.unit
    def test_provider_config_defaults(self):
        """Test default ProviderConfig values."""
        config = ProviderConfig(provider="openai", model="gpt-4o")
        
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
        assert config.top_p == 1.0
        assert config.timeout == 600.0
        assert config.max_retries == 5
        assert config.extra_params == {}
    
    @pytest.mark.unit
    def test_provider_config_from_config_openai(self):
        """Test creating ProviderConfig from model_config for OpenAI."""
        model_config = {
            "transcription_model": {
                "name": "gpt-4o",
                "max_output_tokens": 8192,
                "temperature": 0.5,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2,
            }
        }
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            config = ProviderConfig.from_config(model_config)
        
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.max_tokens == 8192
        assert config.temperature == 0.5
        assert config.top_p == 0.9
        assert config.extra_params["frequency_penalty"] == 0.1
        assert config.extra_params["presence_penalty"] == 0.2
    
    @pytest.mark.unit
    def test_provider_config_from_config_anthropic(self):
        """Test creating ProviderConfig from model_config for Anthropic."""
        model_config = {
            "transcription_model": {
                "name": "claude-3-5-sonnet-20241022",
                "max_output_tokens": 4096,
                "temperature": 0.0,
            }
        }
        
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            config = ProviderConfig.from_config(model_config)
        
        assert config.provider == "anthropic"
        assert config.model == "claude-3-5-sonnet-20241022"
    
    @pytest.mark.unit
    def test_provider_config_from_config_google(self):
        """Test creating ProviderConfig from model_config for Google."""
        model_config = {
            "transcription_model": {
                "name": "gemini-2.0-flash",
                "max_output_tokens": 4096,
            }
        }
        
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            config = ProviderConfig.from_config(model_config)
        
        assert config.provider == "google"
    
    @pytest.mark.unit
    def test_provider_config_from_config_openrouter(self):
        """Test creating ProviderConfig from model_config for OpenRouter."""
        model_config = {
            "transcription_model": {
                "name": "anthropic/claude-3.5-sonnet",
                "max_output_tokens": 4096,
            }
        }
        
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
            config = ProviderConfig.from_config(model_config)
        
        assert config.provider == "openrouter"
        assert config.base_url == "https://openrouter.ai/api/v1"
    
    @pytest.mark.unit
    def test_provider_config_with_explicit_provider(self):
        """Test ProviderConfig with explicit provider field."""
        model_config = {
            "transcription_model": {
                "name": "some-model",
                "provider": "anthropic",
                "max_output_tokens": 4096,
            }
        }
        
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            config = ProviderConfig.from_config(model_config)
        
        assert config.provider == "anthropic"
    
    @pytest.mark.unit
    def test_provider_config_with_reasoning_config(self):
        """Test ProviderConfig with reasoning configuration."""
        model_config = {
            "transcription_model": {
                "name": "o3-mini",
                "max_output_tokens": 16384,
                "reasoning": {
                    "effort": "high",
                    "max_tokens": 8192,
                }
            }
        }
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            config = ProviderConfig.from_config(model_config)
        
        assert "reasoning_config" in config.extra_params
        assert config.extra_params["reasoning_effort"] == "high"
    
    @pytest.mark.unit
    def test_provider_config_provider_override(self):
        """Test provider override in from_config."""
        model_config = {
            "transcription_model": {
                "name": "gpt-4o",
                "max_output_tokens": 4096,
            }
        }
        
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            config = ProviderConfig.from_config(
                model_config, 
                provider_override="anthropic"
            )
        
        assert config.provider == "anthropic"
    
    @pytest.mark.unit
    def test_detect_provider_static_method(self):
        """Test the static _detect_provider method."""
        assert ProviderConfig._detect_provider("gpt-4o") == "openai"
        assert ProviderConfig._detect_provider("claude-3.5-sonnet") == "anthropic"
        assert ProviderConfig._detect_provider("gemini-pro") == "google"
        assert ProviderConfig._detect_provider("unknown-model") == "openai"  # Fallback
    
    @pytest.mark.unit
    def test_get_api_key_static_method(self):
        """Test the static _get_api_key method."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'openai-key',
            'ANTHROPIC_API_KEY': 'anthropic-key',
            'GOOGLE_API_KEY': 'google-key',
            'OPENROUTER_API_KEY': 'openrouter-key',
        }):
            assert ProviderConfig._get_api_key("openai") == "openai-key"
            assert ProviderConfig._get_api_key("anthropic") == "anthropic-key"
            assert ProviderConfig._get_api_key("google") == "google-key"
            assert ProviderConfig._get_api_key("openrouter") == "openrouter-key"
    
    @pytest.mark.unit
    def test_get_api_key_missing(self):
        """Test _get_api_key with missing environment variable."""
        with patch.dict('os.environ', {}, clear=True):
            assert ProviderConfig._get_api_key("openai") is None


class TestNormalizeSchemaForAnthropic:
    """Test schema normalization for Anthropic."""
    
    @pytest.mark.unit
    def test_normalize_simple_schema(self):
        """Test normalizing a simple schema."""
        schema = {"type": "string", "description": "A test field"}
        result = _normalize_schema_for_anthropic(schema)
        
        assert result == schema
    
    @pytest.mark.unit
    def test_normalize_schema_with_type_array(self):
        """Test normalizing schema with type array (union type)."""
        schema = {
            "type": ["string", "null"],
            "description": "Optional field"
        }
        result = _normalize_schema_for_anthropic(schema)
        
        assert "anyOf" in result
        assert len(result["anyOf"]) == 2
        assert {"type": "string", "description": "Optional field"} in result["anyOf"]
        assert {"type": "null"} in result["anyOf"]
    
    @pytest.mark.unit
    def test_normalize_nested_schema(self):
        """Test normalizing nested schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": ["integer", "null"]}
            }
        }
        result = _normalize_schema_for_anthropic(schema)
        
        assert result["type"] == "object"
        assert "anyOf" in result["properties"]["age"]
    
    @pytest.mark.unit
    def test_normalize_list_schema(self):
        """Test normalizing list of schemas."""
        schema = [
            {"type": "string"},
            {"type": ["integer", "null"]}
        ]
        result = _normalize_schema_for_anthropic(schema)
        
        assert isinstance(result, list)
        assert len(result) == 2
    
    @pytest.mark.unit
    def test_normalize_non_dict_non_list(self):
        """Test normalizing non-dict, non-list values."""
        assert _normalize_schema_for_anthropic("string") == "string"
        assert _normalize_schema_for_anthropic(123) == 123
        assert _normalize_schema_for_anthropic(True) is True


class TestComputeOpenRouterReasoningMaxTokens:
    """Test reasoning token budget computation."""
    
    @pytest.mark.unit
    def test_low_effort(self):
        """Test low effort reasoning budget."""
        result = _compute_openrouter_reasoning_max_tokens(10000, "low")
        assert result == 1024  # Minimum bound
    
    @pytest.mark.unit
    def test_medium_effort(self):
        """Test medium effort reasoning budget."""
        result = _compute_openrouter_reasoning_max_tokens(10000, "medium")
        assert result == 2500  # 10000 * 0.25
    
    @pytest.mark.unit
    def test_high_effort(self):
        """Test high effort reasoning budget."""
        result = _compute_openrouter_reasoning_max_tokens(10000, "high")
        assert result == 5000  # 10000 * 0.5
    
    @pytest.mark.unit
    def test_none_effort(self):
        """Test none effort disables reasoning."""
        result = _compute_openrouter_reasoning_max_tokens(10000, "none")
        assert result == 0
    
    @pytest.mark.unit
    def test_default_to_medium(self):
        """Test unknown effort defaults to medium."""
        result = _compute_openrouter_reasoning_max_tokens(10000, "unknown")
        assert result == 2500
    
    @pytest.mark.unit
    def test_minimum_bound(self):
        """Test minimum token bound of 1024."""
        result = _compute_openrouter_reasoning_max_tokens(100, "low")
        assert result == 1024
    
    @pytest.mark.unit
    def test_maximum_bound(self):
        """Test maximum token bound of 32768."""
        result = _compute_openrouter_reasoning_max_tokens(1000000, "high")
        assert result == 32768


class TestBuildOpenRouterReasoningPayload:
    """Test OpenRouter reasoning payload building."""
    
    @pytest.mark.unit
    def test_empty_config_returns_none(self):
        """Test empty reasoning config returns None."""
        result = _build_openrouter_reasoning_payload("gpt-4o", {}, 10000)
        assert result is None
    
    @pytest.mark.unit
    def test_none_config_returns_none(self):
        """Test None reasoning config returns None."""
        result = _build_openrouter_reasoning_payload("gpt-4o", None, 10000)
        assert result is None
    
    @pytest.mark.unit
    def test_with_effort(self):
        """Test payload with effort level."""
        result = _build_openrouter_reasoning_payload(
            "gpt-4o",
            {"effort": "high"},
            10000
        )
        assert result is not None
        assert result["effort"] == "high"
    
    @pytest.mark.unit
    def test_with_explicit_max_tokens(self):
        """Test payload with explicit max_tokens."""
        result = _build_openrouter_reasoning_payload(
            "gpt-4o",
            {"max_tokens": 5000},
            10000
        )
        assert result is not None
        assert result["max_tokens"] == 5000
    
    @pytest.mark.unit
    def test_with_exclude_flag(self):
        """Test payload with exclude flag."""
        result = _build_openrouter_reasoning_payload(
            "gpt-4o",
            {"exclude": True},
            10000
        )
        assert result is not None
        assert result["exclude"] is True
    
    @pytest.mark.unit
    def test_with_enabled_flag(self):
        """Test payload with enabled flag."""
        result = _build_openrouter_reasoning_payload(
            "gpt-4o",
            {"enabled": False},
            10000
        )
        assert result is not None
        assert result["enabled"] is False
    
    @pytest.mark.unit
    def test_anthropic_model_converts_effort_to_max_tokens(self):
        """Test Anthropic model converts effort to max_tokens."""
        result = _build_openrouter_reasoning_payload(
            "anthropic/claude-3.5-sonnet",
            {"effort": "high"},
            10000
        )
        assert result is not None
        assert "effort" not in result
        assert "max_tokens" in result
    
    @pytest.mark.unit
    def test_deepseek_model_converts_effort_to_enabled(self):
        """Test DeepSeek model converts effort to enabled flag."""
        result = _build_openrouter_reasoning_payload(
            "deepseek/deepseek-chat",
            {"effort": "high"},
            10000
        )
        assert result is not None
        assert "effort" not in result
        assert result["enabled"] is True
    
    @pytest.mark.unit
    def test_deepseek_none_effort_disables(self):
        """Test DeepSeek with none effort sets enabled=False."""
        result = _build_openrouter_reasoning_payload(
            "deepseek/deepseek-chat",
            {"effort": "none"},
            10000
        )
        assert result is not None
        assert result["enabled"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
