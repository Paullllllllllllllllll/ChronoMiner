"""Capability lookup: provider detection and capability resolution.

This is the runtime entry point for capability queries. Pure data lives
in :mod:`modules.config.capabilities.registry`; LangChain ``disabled_params``
computation lives in :mod:`modules.config.capabilities.params`.
"""

from __future__ import annotations

from modules.config.capabilities.registry import (
    Capabilities,
    ProviderType,
    _CUSTOM_BASE,
    _MODEL_REGISTRY,
    _OPENAI_REASONING_BASE,
    _OPENROUTER_BASE,
    _norm,
)


def detect_provider(model_name: str) -> ProviderType:
    """
    Detect LLM provider from model name.

    This is the canonical provider detection function. All other modules
    should use this function or delegate to it.

    Args:
        model_name: The model identifier string.

    Returns:
        ProviderType literal: ``"openai"``, ``"anthropic"``, ``"google"``,
        ``"openrouter"``, or ``"unknown"``.
    """
    m = _norm(model_name)

    # OpenRouter models typically have openrouter/ prefix or contain /
    if m.startswith("openrouter/") or "/" in m:
        return "openrouter"

    # Anthropic models
    if m.startswith("claude") or "anthropic" in m:
        return "anthropic"

    # Google models (Gemini + Gemma via Gemini API)
    if m.startswith("gemini") or m.startswith("gemma") or "google" in m:
        return "google"

    # OpenAI models (default)
    if any(m.startswith(prefix) for prefix in ["gpt", "o1", "o3", "o4", "text-"]):
        return "openai"

    return "unknown"


def _build_caps(
    model_name: str, family: str, base: dict, overrides: dict
) -> Capabilities:
    """Merge *base* defaults with *overrides* and return a Capabilities instance."""
    merged = {**base, **overrides}
    merged["model"] = model_name
    merged["family"] = family
    return Capabilities(**merged)


def detect_capabilities(
    model_name: str,
    provider: ProviderType | None = None,
) -> Capabilities:
    m = _norm(model_name)

    # --- Custom endpoints: use conservative defaults -------------------------
    if provider == "custom":
        return _build_caps(model_name, "custom", _CUSTOM_BASE, {})

    # --- Static registry lookup (covers OpenAI, Anthropic, Google) ----------
    for prefixes, family, base, overrides in _MODEL_REGISTRY:
        if any(m.startswith(p) for p in prefixes):
            return _build_caps(model_name, family, base, overrides)

    # --- o3 (not o3-mini, not o3-pro) — requires negative-prefix logic ------
    if m == "o3" or (
        m.startswith("o3-")
        and not m.startswith("o3-mini")
        and not m.startswith("o3-pro")
    ):
        return _build_caps(model_name, "o3", _OPENAI_REASONING_BASE, dict(
            supports_structured_outputs=False,
        ))

    # --- o1 (not o1-mini) — requires negative-prefix logic ------------------
    if m == "o1" or m.startswith("o1-20") or (
        m.startswith("o1") and not m.startswith("o1-mini")
    ):
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
                supports_prompt_caching=True,
                max_context_tokens=200000,
            ))

        # Gemma via OpenRouter (must come before Gemini — both match "google/")
        if "gemma" in m:
            return _build_caps(model_name, "openrouter-gemma", _OPENROUTER_BASE, dict(
                is_reasoning_model=True,
                supports_reasoning_effort=True,
                supports_image_input=True,
                supports_structured_outputs=True,
                max_context_tokens=256000,
            ))

        # Gemini via OpenRouter
        if "gemini" in underlying or "google/" in m:
            is_thinking = any(
                x in m for x in (
                    "gemini-2.5", "gemini-3", "gemini-2-5", "gemini-3-"
                )
            )
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

        # Qwen via OpenRouter
        if "qwen" in underlying or "qwen" in m:
            return _build_caps(model_name, "openrouter-qwen", _OPENROUTER_BASE, dict(
                is_reasoning_model=True,
                supports_reasoning_effort=True,
                supports_image_input=True,
                supports_structured_outputs=True,
                max_context_tokens=131072,
            ))

        # Generic OpenRouter fallback
        return _build_caps(model_name, "openrouter", _OPENROUTER_BASE, {})

    # --- Fallback: conservative text-only -----------------------------------
    return Capabilities(
        model=model_name,
        family="unknown",
        provider=detect_provider(model_name),
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
