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


# ---------------------------------------------------------------------------
# Provider-level capability defaults.  Each model entry in the registry below
# only needs to declare the fields that *differ* from its provider default.
# ---------------------------------------------------------------------------

_OPENAI_REASONING_BASE: dict = dict(
    provider="openai",
    supports_responses_api=True,
    supports_chat_completions=True,
    api_preference="responses",
    is_reasoning_model=True,
    supports_reasoning_effort=True,
    supports_developer_messages=True,
    supports_image_input=True,
    supports_image_detail=True,
    supports_structured_outputs=True,
    supports_function_calling=True,
    supports_sampler_controls=False,
    max_context_tokens=200000,
)

_OPENAI_STANDARD_BASE: dict = dict(
    provider="openai",
    supports_responses_api=True,
    supports_chat_completions=True,
    api_preference="responses",
    is_reasoning_model=False,
    supports_reasoning_effort=False,
    supports_developer_messages=True,
    supports_image_input=True,
    supports_image_detail=True,
    supports_structured_outputs=True,
    supports_function_calling=True,
    supports_sampler_controls=True,
)

_ANTHROPIC_BASE: dict = dict(
    provider="anthropic",
    supports_responses_api=False,
    supports_chat_completions=True,
    api_preference="langchain",
    is_reasoning_model=False,
    supports_reasoning_effort=False,
    supports_developer_messages=True,
    supports_image_input=True,
    supports_image_detail=False,
    supports_structured_outputs=True,
    supports_function_calling=True,
    supports_sampler_controls=True,
    max_context_tokens=200000,
)

_GOOGLE_BASE: dict = dict(
    provider="google",
    supports_responses_api=False,
    supports_chat_completions=True,
    api_preference="langchain",
    is_reasoning_model=False,
    supports_reasoning_effort=False,
    supports_developer_messages=True,
    supports_image_input=True,
    supports_image_detail=False,
    supports_structured_outputs=True,
    supports_function_calling=True,
    supports_sampler_controls=True,
    max_context_tokens=1000000,
)

_OPENROUTER_BASE: dict = dict(
    provider="openrouter",
    supports_responses_api=False,
    supports_chat_completions=True,
    api_preference="langchain",
    is_reasoning_model=False,
    supports_reasoning_effort=False,
    supports_developer_messages=True,
    supports_image_input=True,
    supports_image_detail=False,
    supports_structured_outputs=True,
    supports_function_calling=True,
    supports_sampler_controls=True,
    max_context_tokens=128000,
)

# ---------------------------------------------------------------------------
# Static model registry.  Each entry is (prefixes, family, base, overrides).
# Order matters: more specific prefixes MUST come before less specific ones.
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: list[tuple[tuple[str, ...], str, dict, dict]] = [
    # --- OpenAI GPT-5.2 family ---
    (("gpt-5.2",), "gpt-5.2", _OPENAI_REASONING_BASE, dict(
        supports_chat_completions=False, max_context_tokens=400000,
    )),
    # --- OpenAI GPT-5.1 family ---
    (("gpt-5.1",), "gpt-5.1", _OPENAI_REASONING_BASE, dict(
        supports_chat_completions=False, max_context_tokens=400000,
    )),
    # --- OpenAI GPT-5 family ---
    (("gpt-5",), "gpt-5", _OPENAI_REASONING_BASE, dict(
        supports_chat_completions=False, max_context_tokens=400000,
    )),
    # --- OpenAI o-series reasoning models ---
    (("o4-mini", "o4"), "o4-mini", _OPENAI_REASONING_BASE, {}),
    (("o3-pro",), "o3-pro", _OPENAI_REASONING_BASE, {}),
    (("o3-mini",), "o3-mini", _OPENAI_REASONING_BASE, dict(
        supports_image_input=False, supports_image_detail=False,
    )),
    # o3 (not o3-mini, not o3-pro) — uses a custom match function below
    (("o1-mini",), "o1-mini", _OPENAI_REASONING_BASE, dict(
        supports_responses_api=False, api_preference="chat_completions",
        supports_reasoning_effort=False, supports_developer_messages=False,
        supports_image_input=False, supports_image_detail=False,
        supports_structured_outputs=False, supports_function_calling=False,
    )),
    # o1 (not o1-mini) — uses a custom match function below
    # --- OpenAI GPT-4o / GPT-4.1 ---
    (("gpt-4o",), "gpt-4o", _OPENAI_STANDARD_BASE, {}),
    (("gpt-4.1",), "gpt-4.1", _OPENAI_STANDARD_BASE, dict(
        api_preference="langchain",
    )),
    # --- Anthropic Claude models (most-specific first) ---
    (("claude-opus-4-6", "claude-opus-4.6"), "claude-opus-4.6", _ANTHROPIC_BASE, {}),
    (("claude-opus-4-5", "claude-opus-4.5"), "claude-opus-4.5", _ANTHROPIC_BASE, {}),
    (("claude-opus-4-1", "claude-opus-4.1"), "claude-opus-4.1", _ANTHROPIC_BASE, {}),
    (("claude-opus-4",), "claude-opus-4", _ANTHROPIC_BASE, {}),
    (("claude-sonnet-4-6", "claude-sonnet-4.6"), "claude-sonnet-4.6", _ANTHROPIC_BASE, {}),
    (("claude-sonnet-4-5", "claude-sonnet-4.5"), "claude-sonnet-4.5", _ANTHROPIC_BASE, {}),
    (("claude-sonnet-4",), "claude-sonnet-4", _ANTHROPIC_BASE, {}),
    (("claude-haiku-4-5", "claude-haiku-4.5"), "claude-haiku-4.5", _ANTHROPIC_BASE, {}),
    (("claude-3-7-sonnet", "claude-3.7-sonnet"), "claude-3.7-sonnet", _ANTHROPIC_BASE, {}),
    (("claude-3-5-sonnet", "claude-3.5-sonnet"), "claude-3.5-sonnet", _ANTHROPIC_BASE, {}),
    (("claude-3-5-haiku", "claude-3.5-haiku"), "claude-3.5-haiku", _ANTHROPIC_BASE, {}),
    (("claude-3-opus",), "claude-3-opus", _ANTHROPIC_BASE, {}),
    (("claude-3-sonnet",), "claude-3-sonnet", _ANTHROPIC_BASE, {}),
    (("claude-3-haiku",), "claude-3-haiku", _ANTHROPIC_BASE, {}),
    (("claude",), "claude", _ANTHROPIC_BASE, {}),
    # --- Google Gemini models (most-specific first) ---
    (("gemini-3-pro", "gemini-3.0-pro"), "gemini-3-pro", _GOOGLE_BASE, dict(
        is_reasoning_model=True, max_context_tokens=1048576,
    )),
    (("gemini-3-flash-preview", "gemini-3.0-flash-preview"), "gemini-3-flash-preview", _GOOGLE_BASE, dict(
        is_reasoning_model=True, max_context_tokens=1048576,
    )),
    (("gemini-2.5-pro", "gemini-2-5-pro"), "gemini-2.5-pro", _GOOGLE_BASE, dict(
        is_reasoning_model=True, max_context_tokens=1048576,
    )),
    (("gemini-2.5-flash-lite", "gemini-2-5-flash-lite"), "gemini-2.5-flash-lite", _GOOGLE_BASE, dict(
        max_context_tokens=1048576,
    )),
    (("gemini-2.5-flash", "gemini-2-5-flash"), "gemini-2.5-flash", _GOOGLE_BASE, dict(
        is_reasoning_model=True, max_context_tokens=1048576,
    )),
    (("gemini-2.0-flash", "gemini-2-flash", "gemini-2.0"), "gemini-2.0-flash", _GOOGLE_BASE, {}),
    (("gemini-1.5-pro", "gemini-1-5-pro"), "gemini-1.5-pro", _GOOGLE_BASE, dict(
        max_context_tokens=2000000,
    )),
    (("gemini-1.5-flash", "gemini-1-5-flash"), "gemini-1.5-flash", _GOOGLE_BASE, {}),
    (("gemini",), "gemini", _GOOGLE_BASE, {}),
]


def _build_caps(model_name: str, family: str, base: dict, overrides: dict) -> Capabilities:
    """Merge *base* defaults with *overrides* and return a Capabilities instance."""
    merged = {**base, **overrides}
    merged["model"] = model_name
    merged["family"] = family
    return Capabilities(**merged)


def detect_capabilities(model_name: str) -> Capabilities:
    m = _norm(model_name)

    # --- Static registry lookup (covers OpenAI, Anthropic, Google) ----------
    for prefixes, family, base, overrides in _MODEL_REGISTRY:
        if any(m.startswith(p) for p in prefixes):
            return _build_caps(model_name, family, base, overrides)

    # --- o3 (not o3-mini, not o3-pro) — requires negative-prefix logic ------
    if m == "o3" or (m.startswith("o3-") and not m.startswith("o3-mini") and not m.startswith("o3-pro")):
        return _build_caps(model_name, "o3", _OPENAI_REASONING_BASE, dict(
            supports_structured_outputs=False,
        ))

    # --- o1 (not o1-mini) — requires negative-prefix logic ------------------
    if m == "o1" or m.startswith("o1-20") or (m.startswith("o1") and not m.startswith("o1-mini")):
        return _build_caps(model_name, "o1", _OPENAI_REASONING_BASE, dict(
            api_preference="either",
            supports_reasoning_effort=False,
            supports_structured_outputs=False,
        ))

    # --- OpenRouter models (dynamic matching on underlying model) -----------
    if m.startswith("openrouter/") or "/" in m:
        underlying = m.split("/")[-1] if "/" in m else m

        # DeepSeek
        if "deepseek" in m:
            is_r1 = "deepseek-r1" in m
            is_terminus = "terminus" in m
            return _build_caps(model_name, "openrouter-deepseek", _OPENROUTER_BASE, dict(
                is_reasoning_model=is_r1 or is_terminus,
                supports_reasoning_effort=True,
                supports_sampler_controls=not is_r1,
            ))

        # GPT-OSS-120b via OpenRouter (DeepInfra fp4)
        if "gpt-oss" in m:
            return _build_caps(model_name, "openrouter-gpt-oss", _OPENROUTER_BASE, dict(
                is_reasoning_model=True,
                supports_reasoning_effort=True,
                supports_sampler_controls=False,
                supports_structured_outputs=True,
                supports_function_calling=True,
                max_context_tokens=131072,
            ))

        # GPT-5 via OpenRouter
        if "gpt-5" in m:
            return _build_caps(model_name, "openrouter-gpt5", _OPENROUTER_BASE, dict(
                is_reasoning_model=True,
                supports_reasoning_effort=True,
                supports_image_detail=True,
                supports_sampler_controls=False,
                max_context_tokens=256000,
            ))

        # o-series via OpenRouter
        if any(x in m for x in ("/o1", "/o3", "/o4", "openai/o1", "openai/o3", "openai/o4")):
            return _build_caps(model_name, "openrouter-o-series", _OPENROUTER_BASE, dict(
                is_reasoning_model=True,
                supports_reasoning_effort=True,
                supports_image_input="mini" not in m,
                supports_image_detail=True,
                supports_sampler_controls=False,
                max_context_tokens=200000,
            ))

        # Claude via OpenRouter
        if "claude" in underlying or "anthropic/" in m:
            return _build_caps(model_name, "openrouter-claude", _OPENROUTER_BASE, dict(
                is_reasoning_model=True,
                supports_reasoning_effort=True,
                max_context_tokens=200000,
            ))

        # Gemini via OpenRouter
        if "gemini" in underlying or "google/" in m:
            is_thinking = any(x in m for x in ("gemini-2.5", "gemini-3", "gemini-2-5", "gemini-3-"))
            return _build_caps(model_name, "openrouter-gemini", _OPENROUTER_BASE, dict(
                is_reasoning_model=is_thinking,
                supports_reasoning_effort=True,
                max_context_tokens=1000000,
            ))

        # Llama via OpenRouter
        if "llama" in underlying or "meta/" in m:
            return _build_caps(model_name, "openrouter-llama", _OPENROUTER_BASE, dict(
                supports_image_input="vision" in m or "llama-3.2" in m,
            ))

        # Mistral via OpenRouter
        if "mistral" in underlying or "mixtral" in m:
            return _build_caps(model_name, "openrouter-mistral", _OPENROUTER_BASE, dict(
                supports_image_input="pixtral" in m,
            ))

        # Generic OpenRouter fallback
        return _build_caps(model_name, "openrouter", _OPENROUTER_BASE, {})

    # --- Fallback: conservative text-only -----------------------------------
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
