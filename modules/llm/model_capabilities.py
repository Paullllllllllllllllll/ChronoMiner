"""
Model capabilities detection and gating.

This module provides capability detection for different LLM models and providers.
It is used to determine which API parameters are supported by each model.

LangChain Integration:
======================
LangChain does NOT automatically handle capability guarding (e.g., it will pass
temperature to reasoning models that don't support it, causing API errors).

This module provides the capability detection logic that is used by:
1. langchain_provider.py - to set `disabled_params` for unsupported parameters
2. openai_utils.py - to determine which features to enable (structured outputs, etc.)

The capabilities detected here are used to:
- Filter out unsupported sampler controls (temperature, top_p) for reasoning models
- Disable response_format for models that don't support structured outputs
- Enable reasoning_effort for reasoning models
- Set appropriate max_context_tokens for different models

This is NOT deprecated - LangChain needs our capability detection to work correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ImageDetail = Literal["auto", "high", "low"]
ApiPref = Literal["responses", "chat_completions", "either", "langchain"]
ProviderType = Literal["openai", "anthropic", "google", "openrouter", "unknown"]


@dataclass(frozen=True, slots=True)
class Capabilities:
    """
    Registry of model capabilities to gate API payload features.
    
    Supports multiple providers:
    - OpenAI (GPT-4o, GPT-4.1, o1, o3, GPT-5)
    - Anthropic (Claude 3.5 Sonnet, Claude 3 Opus, etc.)
    - Google (Gemini 2.0 Flash, Gemini 1.5 Pro, etc.)
    - OpenRouter (unified access to multiple providers)
    
    Key capability flags:
    - supports_sampler_controls: If False, temperature/top_p are disabled via LangChain's disabled_params
    - supports_structured_outputs: If False, response_format is disabled
    - supports_reasoning_effort: If True, reasoning_effort parameter is enabled
    - is_reasoning_model: Indicates o1, o3, or gpt-5 family models
    """

    model: str
    family: str
    provider: ProviderType = "openai"

    supports_responses_api: bool = True
    supports_chat_completions: bool = True
    api_preference: ApiPref = "langchain"

    is_reasoning_model: bool = False
    supports_reasoning_effort: bool = False
    supports_developer_messages: bool = True

    supports_image_input: bool = False
    supports_image_detail: bool = False
    default_ocr_detail: ImageDetail = "high"

    supports_structured_outputs: bool = True
    supports_function_calling: bool = True

    supports_sampler_controls: bool = True
    
    # Extended context window support (for Claude, Gemini)
    max_context_tokens: int = 128000


def _norm(name: str) -> str:
    return name.strip().lower()


def detect_provider(model_name: str) -> ProviderType:
    """
    Detect LLM provider from model name.
    
    This is the canonical provider detection function. All other modules
    should use this function or delegate to it.
    
    Args:
        model_name: The model identifier string.
        
    Returns:
        ProviderType literal: "openai", "anthropic", "google", "openrouter", or "unknown"
    """
    m = _norm(model_name)
    
    # OpenRouter models typically have openrouter/ prefix or contain /
    if m.startswith("openrouter/") or "/" in m:
        return "openrouter"
    
    # Anthropic models
    if m.startswith("claude") or "anthropic" in m:
        return "anthropic"
    
    # Google models
    if m.startswith("gemini") or "google" in m:
        return "google"
    
    # OpenAI models (default)
    if any(m.startswith(prefix) for prefix in ["gpt", "o1", "o3", "o4", "text-"]):
        return "openai"
    
    return "unknown"


# Backward compatibility alias
_detect_provider = detect_provider


def detect_capabilities(model_name: str) -> Capabilities:
    m = _norm(model_name)

    # ========================================================================
    # OpenAI GPT-5.1 family (November 2025) - reasoning with adaptive thinking
    # Models: gpt-5.1, gpt-5.1-instant, gpt-5.1-thinking
    # ========================================================================
    if m.startswith("gpt-5.1"):
        return Capabilities(
            model=model_name,
            family="gpt-5.1",
            provider="openai",
            supports_responses_api=True,
            supports_chat_completions=False,
            api_preference="responses",
            is_reasoning_model=True,
            supports_reasoning_effort=True,  # Supports "none", "low", "medium", "high"
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=False,  # No temperature/top_p
            max_context_tokens=256000,
        )

    # ========================================================================
    # OpenAI GPT-5 family (August 2025) - reasoning, multimodal
    # Models: gpt-5, gpt-5-mini, gpt-5-nano
    # ========================================================================
    if m.startswith("gpt-5"):
        return Capabilities(
            model=model_name,
            family="gpt-5",
            provider="openai",
            supports_responses_api=True,
            supports_chat_completions=False,
            api_preference="responses",
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=False,
            max_context_tokens=256000,
        )

    # ========================================================================
    # OpenAI o-series reasoning models (2025)
    # ========================================================================
    
    # o4-mini (2025) - latest reasoning model, optimized for tool use
    if m.startswith("o4-mini") or m.startswith("o4"):
        return Capabilities(
            model=model_name,
            family="o4-mini",
            provider="openai",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="responses",
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=False,
            max_context_tokens=200000,
        )
    
    # o3-pro (reasoning, enhanced tool usage)
    if m.startswith("o3-pro"):
        return Capabilities(
            model=model_name,
            family="o3-pro",
            provider="openai",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="responses",
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=False,
            max_context_tokens=200000,
        )
    
    # o3 (reasoning, vision, avoid sampler controls)
    if m == "o3" or (m.startswith("o3-") and not m.startswith("o3-mini") and not m.startswith("o3-pro")):
        return Capabilities(
            model=model_name,
            family="o3",
            provider="openai",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="responses",
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=False,
            supports_function_calling=True,
            supports_sampler_controls=False,
            max_context_tokens=200000,
        )

    # o3-mini (reasoning, no vision)
    if m.startswith("o3-mini"):
        return Capabilities(
            model=model_name,
            family="o3-mini",
            provider="openai",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="responses",
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_developer_messages=True,
            supports_image_input=False,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=False,
            max_context_tokens=200000,
        )

    # o1 (full reasoning; no sampler controls; allow Responses)
    if m == "o1" or m.startswith("o1-20") or (m.startswith("o1") and not m.startswith("o1-mini")):
        return Capabilities(
            model=model_name,
            family="o1",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="either",
            is_reasoning_model=True,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=False,
            supports_function_calling=True,
            supports_sampler_controls=False,
        )

    # o1-mini (small reasoning; prefer Chat Completions; no vision)
    if m.startswith("o1-mini"):
        return Capabilities(
            model=model_name,
            family="o1-mini",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="chat_completions",
            is_reasoning_model=True,
            supports_reasoning_effort=False,
            supports_developer_messages=False,
            supports_image_input=False,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=False,
            supports_function_calling=False,
            supports_sampler_controls=False,
        )

    # GPT-4o family (multimodal; structured outputs; sampler controls)
    if m.startswith("gpt-4o"):
        return Capabilities(
            model=model_name,
            family="gpt-4o",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="responses",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
        )

    # GPT-4.1 family (multimodal; structured outputs; sampler controls)
    if m.startswith("gpt-4.1"):
        return Capabilities(
            model=model_name,
            family="gpt-4.1",
            provider="openai",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
        )

    # ========================================================================
    # Anthropic Claude models (2025)
    # Model naming: claude-{tier}-{version}-{date}
    # Tiers: opus (largest), sonnet (balanced), haiku (fastest)
    # ========================================================================
    
    # Claude Opus 4.5 (November 2025) - most intelligent model
    # API ID: claude-opus-4-5-20251101
    if m.startswith("claude-opus-4-5") or m.startswith("claude-opus-4.5"):
        return Capabilities(
            model=model_name,
            family="claude-opus-4.5",
            provider="anthropic",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=200000,
        )
    
    # Claude Opus 4.1 (August 2025) - claude-opus-4-1-20250805
    if m.startswith("claude-opus-4-1") or m.startswith("claude-opus-4.1"):
        return Capabilities(
            model=model_name,
            family="claude-opus-4.1",
            provider="anthropic",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=200000,
        )
    
    # Claude Opus 4 (May 2025) - claude-opus-4-20250514
    if m.startswith("claude-opus-4"):
        return Capabilities(
            model=model_name,
            family="claude-opus-4",
            provider="anthropic",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=200000,
        )
    
    # Claude Sonnet 4.5 (October 2025) - most aligned frontier model
    if m.startswith("claude-sonnet-4-5") or m.startswith("claude-sonnet-4.5"):
        return Capabilities(
            model=model_name,
            family="claude-sonnet-4.5",
            provider="anthropic",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=200000,
        )
    
    # Claude Sonnet 4 (May 2025) - claude-sonnet-4-20250514
    if m.startswith("claude-sonnet-4"):
        return Capabilities(
            model=model_name,
            family="claude-sonnet-4",
            provider="anthropic",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=200000,
        )
    
    # Claude Haiku 4.5 (November 2025) - fast, cost-effective
    if m.startswith("claude-haiku-4-5") or m.startswith("claude-haiku-4.5"):
        return Capabilities(
            model=model_name,
            family="claude-haiku-4.5",
            provider="anthropic",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=200000,
        )
    
    # Claude 3.7 Sonnet (February 2025) - claude-3-7-sonnet-20250219
    if m.startswith("claude-3-7-sonnet") or m.startswith("claude-3.7-sonnet"):
        return Capabilities(
            model=model_name,
            family="claude-3.7-sonnet",
            provider="anthropic",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=200000,
        )
    
    # Claude 3.5 Sonnet (October 2024) - claude-3-5-sonnet-20241022
    if m.startswith("claude-3-5-sonnet") or m.startswith("claude-3.5-sonnet"):
        return Capabilities(
            model=model_name,
            family="claude-3.5-sonnet",
            provider="anthropic",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=200000,
        )
    
    # Claude 3.5 Haiku (October 2024) - claude-3-5-haiku-20241022
    if m.startswith("claude-3-5-haiku") or m.startswith("claude-3.5-haiku"):
        return Capabilities(
            model=model_name,
            family="claude-3.5-haiku",
            provider="anthropic",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=200000,
        )
    
    # Claude 3 Opus (legacy, deprecated June 2025)
    if m.startswith("claude-3-opus"):
        return Capabilities(
            model=model_name,
            family="claude-3-opus",
            provider="anthropic",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=200000,
        )
    
    # Claude 3 Sonnet (legacy, retired July 2025)
    if m.startswith("claude-3-sonnet"):
        return Capabilities(
            model=model_name,
            family="claude-3-sonnet",
            provider="anthropic",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=200000,
        )
    
    # Claude 3 Haiku (legacy)
    if m.startswith("claude-3-haiku"):
        return Capabilities(
            model=model_name,
            family="claude-3-haiku",
            provider="anthropic",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=200000,
        )
    
    # Generic Claude fallback (for any new Claude models)
    if m.startswith("claude"):
        return Capabilities(
            model=model_name,
            family="claude",
            provider="anthropic",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=200000,
        )

    # ========================================================================
    # Google Gemini models (2025)
    # Model naming: gemini-{version}-{variant}
    # Variants: pro (powerful), flash (fast), flash-lite (ultra fast)
    # ========================================================================
    
    # Gemini 3 Pro (2025) - most powerful, multimodal understanding
    # Models: gemini-3-pro-preview, gemini-3-pro-image-preview
    if m.startswith("gemini-3-pro") or m.startswith("gemini-3.0-pro"):
        return Capabilities(
            model=model_name,
            family="gemini-3-pro",
            provider="google",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=True,  # Thinking/reasoning model
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=1048576,  # 1M tokens
        )
    
    # Gemini 2.5 Pro (2025) - state-of-the-art thinking model
    # Models: gemini-2.5-pro, gemini-2.5-pro-preview-tts
    if m.startswith("gemini-2.5-pro") or m.startswith("gemini-2-5-pro"):
        return Capabilities(
            model=model_name,
            family="gemini-2.5-pro",
            provider="google",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=True,  # Thinking/reasoning model
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=1048576,  # 1M tokens
        )
    
    # Gemini 2.5 Flash (2025) - best price-performance, thinking
    # Models: gemini-2.5-flash, gemini-2.5-flash-preview-*, gemini-2.5-flash-image
    if m.startswith("gemini-2.5-flash") or m.startswith("gemini-2-5-flash"):
        return Capabilities(
            model=model_name,
            family="gemini-2.5-flash",
            provider="google",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=True,  # Thinking/reasoning model
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=1048576,  # 1M tokens
        )
    
    # Gemini 2.5 Flash-Lite (2025) - ultra fast, cost optimized
    if m.startswith("gemini-2.5-flash-lite") or m.startswith("gemini-2-5-flash-lite"):
        return Capabilities(
            model=model_name,
            family="gemini-2.5-flash-lite",
            provider="google",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=1048576,  # 1M tokens
        )
    
    # Gemini 2.0 Flash (previous gen)
    if m.startswith("gemini-2.0-flash") or m.startswith("gemini-2-flash") or m.startswith("gemini-2.0"):
        return Capabilities(
            model=model_name,
            family="gemini-2.0-flash",
            provider="google",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=1000000,  # 1M tokens
        )
    
    # Gemini 1.5 Pro (legacy)
    if m.startswith("gemini-1.5-pro") or m.startswith("gemini-1-5-pro"):
        return Capabilities(
            model=model_name,
            family="gemini-1.5-pro",
            provider="google",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=2000000,  # 2M tokens
        )
    
    # Gemini 1.5 Flash (legacy)
    if m.startswith("gemini-1.5-flash") or m.startswith("gemini-1-5-flash"):
        return Capabilities(
            model=model_name,
            family="gemini-1.5-flash",
            provider="google",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=1000000,  # 1M tokens
        )
    
    # Generic Gemini fallback (for any new Gemini models)
    if m.startswith("gemini"):
        return Capabilities(
            model=model_name,
            family="gemini",
            provider="google",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=1000000,
        )

    # ========================================================================
    # OpenRouter models (detect underlying model)
    # ========================================================================
    
    if m.startswith("openrouter/") or "/" in m:
        # Extract underlying model name for capability detection
        underlying = m.split("/")[-1] if "/" in m else m
        
        # OpenRouter Anthropic models
        if "claude" in underlying:
            return Capabilities(
                model=model_name,
                family="openrouter-claude",
                provider="openrouter",
                supports_responses_api=False,
                supports_chat_completions=True,
                api_preference="langchain",
                is_reasoning_model=False,
                supports_reasoning_effort=False,
                supports_developer_messages=True,
                supports_image_input=True,
                supports_image_detail=False,
                default_ocr_detail="high",
                supports_structured_outputs=True,
                supports_function_calling=True,
                supports_sampler_controls=True,
                max_context_tokens=200000,
            )
        
        # OpenRouter Google models
        if "gemini" in underlying:
            return Capabilities(
                model=model_name,
                family="openrouter-gemini",
                provider="openrouter",
                supports_responses_api=False,
                supports_chat_completions=True,
                api_preference="langchain",
                is_reasoning_model=False,
                supports_reasoning_effort=False,
                supports_developer_messages=True,
                supports_image_input=True,
                supports_image_detail=False,
                default_ocr_detail="high",
                supports_structured_outputs=True,
                supports_function_calling=True,
                supports_sampler_controls=True,
                max_context_tokens=1000000,
            )
        
        # OpenRouter Llama models
        if "llama" in underlying:
            return Capabilities(
                model=model_name,
                family="openrouter-llama",
                provider="openrouter",
                supports_responses_api=False,
                supports_chat_completions=True,
                api_preference="langchain",
                is_reasoning_model=False,
                supports_reasoning_effort=False,
                supports_developer_messages=True,
                supports_image_input=False,
                supports_image_detail=False,
                default_ocr_detail="high",
                supports_structured_outputs=True,
                supports_function_calling=True,
                supports_sampler_controls=True,
                max_context_tokens=128000,
            )
        
        # OpenRouter Mistral models
        if "mistral" in underlying:
            return Capabilities(
                model=model_name,
                family="openrouter-mistral",
                provider="openrouter",
                supports_responses_api=False,
                supports_chat_completions=True,
                api_preference="langchain",
                is_reasoning_model=False,
                supports_reasoning_effort=False,
                supports_developer_messages=True,
                supports_image_input=False,
                supports_image_detail=False,
                default_ocr_detail="high",
                supports_structured_outputs=True,
                supports_function_calling=True,
                supports_sampler_controls=True,
                max_context_tokens=32000,
            )
        
        # Generic OpenRouter fallback
        return Capabilities(
            model=model_name,
            family="openrouter",
            provider="openrouter",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=False,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
            max_context_tokens=128000,
        )

    # Fallback conservative text-only
    return Capabilities(
        model=model_name,
        family="unknown",
        provider=_detect_provider(model_name),
        supports_responses_api=True,
        supports_chat_completions=True,
        api_preference="langchain",
        is_reasoning_model=False,
        supports_reasoning_effort=False,
        supports_developer_messages=True,
        supports_image_input=False,
        supports_image_detail=False,
        default_ocr_detail="high",
        supports_structured_outputs=True,
        supports_function_calling=True,
        supports_sampler_controls=True,
    )
