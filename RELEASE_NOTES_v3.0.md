# ChronoMiner v3.0 Release Notes

## Release Information

**Version:** 3.0  
**Release Date:** November 2025  
**Status:** Production Ready

## Overview

ChronoMiner v3.0 introduces a major architectural evolution with **multi-provider LLM support** powered by LangChain. This release enables seamless switching between OpenAI, Anthropic Claude, Google Gemini, and OpenRouter providers while maintaining full backward compatibility with existing workflows. The update also includes comprehensive model capability detection and support for the latest 2025 models across all providers.

## Major Features

### 1. Multi-Provider LLM Support via LangChain

ChronoMiner now supports four major LLM providers through a unified LangChain-based abstraction:

**Supported Providers:**
- **OpenAI** - GPT-4o, GPT-4.1, o1, o3, o4-mini, GPT-5, GPT-5.1
- **Anthropic** - Claude 3.5 Sonnet, Claude 4 Opus, Claude 4.5
- **Google Gemini** - Gemini 2.0 Flash, Gemini 2.5, Gemini 3 Pro
- **OpenRouter** - Unified access to 100+ models

**Key Benefits:**
- Write once, use with any provider
- Runtime provider switching without code changes
- Unified tool calling via `bind_tools()`
- Consistent structured output handling
- Automatic capability detection and parameter guarding

**Configuration:**
```yaml
# config/model_config.yaml
transcription_model:
  name: "claude-sonnet-4-5-20251101"  # or gpt-4o, gemini-2.5-pro, etc.
  temperature: 0.0
  max_output_tokens: 4096
```

### 2. Comprehensive Model Capability System

A new capability detection system automatically handles model-specific features and limitations:

**Capability Guarding:**
- Automatic parameter filtering for reasoning models (o1, o3, GPT-5)
- Disables unsupported parameters (temperature, top_p) for thinking models
- Structured output validation per provider
- Context window validation

**Provider Detection:**
- Automatic provider detection from model name
- Supports `claude-*`, `gemini-*`, `gpt-*`, `o1-*`, `o3-*`, `o4-*` prefixes
- OpenRouter models via `openrouter/` prefix or path notation

### 3. Latest 2025 Model Support

#### OpenAI Models Added
| Model | Context | Features |
|-------|---------|----------|
| `gpt-5.1`, `gpt-5.1-instant`, `gpt-5.1-thinking` | 256K | Adaptive thinking, Responses API |
| `gpt-5`, `gpt-5-mini`, `gpt-5-nano` | 256K | Reasoning models |
| `o4-mini` | 200K | Tool-use optimized reasoning |
| `o3-pro` | 200K | Deep reasoning variant |

#### Anthropic Claude Models Added
| Model | Context | Features |
|-------|---------|----------|
| `claude-opus-4-5-20251101` | 200K | Most intelligent |
| `claude-opus-4-1-20250805` | 200K | Opus 4.1 |
| `claude-opus-4-20250514` | 200K | Opus 4 |
| `claude-sonnet-4-5-*` variants | 200K | Most aligned |
| `claude-sonnet-4-20250514` | 200K | Sonnet 4 |
| `claude-haiku-4-5-*` variants | 200K | Fast inference |

#### Google Gemini Models Added
| Model | Context | Features |
|-------|---------|----------|
| `gemini-3-pro-preview` | 2M | Thinking model |
| `gemini-2.5-pro` | 1M | Thinking model |
| `gemini-2.5-flash` | 1M | Thinking model |
| `gemini-2.5-flash-lite` | 1M | Non-thinking |

### 4. LangChain Integration Architecture

**New Module:** `modules/llm/langchain_provider.py`

Core classes:
- **`ProviderConfig`** - Configuration dataclass for provider settings
- **`LangChainLLM`** - Unified wrapper with capability guarding
- **`LLMProvider`** - Factory for creating/caching LLM instances

**Features:**
- Lazy initialization for performance
- Instance caching by provider:model key
- Automatic retry configuration from `concurrency_config.yaml`
- Token usage tracking integration

## Technical Improvements

### Code Architecture

**New Modules:**
- `modules/llm/langchain_provider.py` - Multi-provider LLM abstraction (612 lines)

**Enhanced Modules:**
- `modules/llm/model_capabilities.py` - Comprehensive capability definitions (+736 lines)
- `modules/llm/structured_outputs.py` - Multi-provider structured output support
- `modules/llm/openai_utils.py` - Refactored for provider abstraction
- `modules/llm/batching.py` - OpenAI batch support improvements

**Configuration Updates:**
- `config/model_config.yaml` - Multi-provider model examples
- `config/paths_config.yaml` - Provider configuration options

### Dependency Updates

**New Dependencies:**
| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | 1.1.0 | Multi-provider LLM framework |
| `langchain-core` | 1.1.0 | Core LangChain components |
| `langchain-openai` | 1.1.0 | OpenAI provider integration |
| `langchain-anthropic` | 1.2.0 | Anthropic Claude provider |
| `langchain-google-genai` | 3.2.0 | Google Gemini provider |
| `anthropic` | 0.75.0 | Anthropic SDK |
| `google-ai-generativelanguage` | 0.9.0 | Google AI SDK |
| `langgraph` | 1.0.4 | LangChain graph support |
| `langsmith` | 0.4.49 | LangChain tracing |

**Updated Dependencies:**
| Package | Old Version | New Version |
|---------|-------------|-------------|
| `openai` | 2.1.0 | 2.8.1 |
| `numpy` | 2.3.3 | 2.3.5 |
| `pydantic` | 2.11.9 | 2.12.5 |
| `pydantic_core` | 2.33.2 | 2.41.5 |
| `tiktoken` | 0.11.0 | 0.12.0 |
| `regex` | 2025.9.18 | 2025.11.3 |

**Removed Dependencies:**
| Package | Reason |
|---------|--------|
| `pypdf` | Not used in codebase |

### API Key Management

Multi-provider API key support via environment variables:

```bash
# OpenAI (required for batch processing)
export OPENAI_API_KEY="sk-..."

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GOOGLE_API_KEY="AIza..."

# OpenRouter (unified access)
export OPENROUTER_API_KEY="sk-or-..."
```

## Usage Examples

### Switching Providers

Simply change the model name in your configuration:

```yaml
# OpenAI
transcription_model:
  name: "gpt-4o"

# Anthropic Claude
transcription_model:
  name: "claude-sonnet-4-5-20251101"

# Google Gemini
transcription_model:
  name: "gemini-2.5-pro"

# OpenRouter
transcription_model:
  name: "openrouter/anthropic/claude-3-5-sonnet"
```

### Programmatic Provider Selection

```python
from modules.llm.langchain_provider import LLMProvider

# Get LLM by model name (auto-detects provider)
llm = LLMProvider.get_llm(model="claude-sonnet-4-5-20251101")

# Or specify provider explicitly
llm = LLMProvider.get_llm(model="gpt-4o", provider="openai")

# Use with structured output
result = await llm.ainvoke_with_structured_output(
    messages=[{"role": "user", "content": "Extract data..."}],
    json_schema=schema
)
```

### Checking Available Providers

```python
from modules.llm.langchain_provider import list_available_providers, get_default_provider

# List providers with configured API keys
providers = list_available_providers()
# Returns: ["openai", "anthropic", "google"]

# Get default provider
default = get_default_provider()
# Returns: "openai" (first available)
```

## Migration Guide

### From v2.x to v3.0

**No Breaking Changes** - Existing OpenAI workflows continue to work unchanged.

**New Capabilities to Adopt:**

1. **Multi-provider support:** Set alternative provider API keys
2. **Model updates:** Use latest 2025 models in configuration
3. **Capability guarding:** Automatic parameter filtering (no code changes needed)

**Configuration Updates:**

```yaml
# Old (still works)
transcription_model:
  name: "gpt-4o"

# New options
transcription_model:
  name: "claude-sonnet-4-5-20251101"  # Anthropic
  # or
  name: "gemini-2.5-pro"  # Google
```

**Batch Processing Note:**
- Batch processing remains **OpenAI-only** (API limitation)
- Other providers use synchronous/async processing

## Known Limitations

**Provider-Specific:**
- Batch API only available for OpenAI
- Structured output methods vary by provider (handled automatically)
- Some reasoning models don't support temperature/top_p controls

**API Requirements:**
- Each provider requires its own API key
- Rate limits vary by provider and plan
- Token pricing differs across providers

## Future Enhancements

**Planned Improvements:**
- Anthropic batch API support (when available)
- Provider fallback chains
- Cost optimization routing
- LangGraph workflow integration
- Enhanced multi-modal support (images, audio)

## System Requirements

**Updated from v2.x:**
- Python 3.12+
- At least one provider API key configured
- Required packages in `requirements.txt`

**Recommended:**
- Multiple provider API keys for redundancy
- OpenAI API key for batch processing features

## Documentation

**Updated Documentation:**
- `README.md` - Multi-provider setup and usage
- Supported models tables for all providers
- API key configuration for each provider
- Model capability explanations

**New Documentation:**
- `LANGCHAIN_RESEARCH.md` - LangChain integration reference

## Verification

All changes verified with import tests:

```
All imports successful
GPT-5.1 caps: family=gpt-5.1, reasoning=True
Claude Opus 4.5 caps: family=claude-opus-4.5, provider=anthropic
Gemini 3 Pro caps: family=gemini-3-pro, reasoning=True
LangChain version: 1.1.0
```

## Credits and Acknowledgments

**Development:**
- LangChain multi-provider integration
- Comprehensive model capability system
- Latest 2025 model support
- Documentation updates

**Dependencies:**
- LangChain team for unified LLM abstraction
- OpenAI, Anthropic, Google for API access

## Upgrade Instructions

**For Existing Users:**

1. Pull the latest version from the repository
2. Update dependencies: `pip install -r requirements.txt`
3. Configure additional provider API keys (optional)
4. Update model configuration to use preferred provider
5. Enjoy multi-provider LLM support

**Dependency Installation:**
```bash
pip install -r requirements.txt
```

No data migration or schema updates required.

## Conclusion

ChronoMiner v3.0 represents a significant architectural evolution, enabling true multi-provider LLM support without sacrificing the reliability and ease of use of previous versions. Whether you prefer OpenAI's ecosystem, Anthropic's Claude models, or Google's Gemini, ChronoMiner now provides seamless access to all major providers through a unified interface.

The LangChain-based architecture ensures future-proof extensibility as new providers and models emerge, while the comprehensive capability detection system handles the complexity of model-specific features automatically.

Thank you for using ChronoMiner!
