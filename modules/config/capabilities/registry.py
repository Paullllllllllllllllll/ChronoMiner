"""Capability data: dataclass definition, provider bases, and model registry.

Pure-data module. No runtime dependencies beyond ``typing`` and
``dataclasses``. Every runtime lookup happens in
:mod:`modules.config.capabilities.detection`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ImageDetail = Literal["auto", "high", "low"]
ApiPref = Literal["responses", "chat_completions", "either", "langchain"]
ProviderType = Literal[
    "openai", "anthropic", "google", "openrouter", "custom", "unknown"
]


@dataclass(frozen=True, slots=True)
class Capabilities:
    """
    Registry of model capabilities to gate API payload features.

    Supports multiple providers:
    - OpenAI (GPT-4o, GPT-4.1, o1, o3, GPT-5, GPT-5.3, GPT-5.4)
    - Anthropic (Claude 3.5 Sonnet, Claude 3 Opus, etc.)
    - Google (Gemini 2.0 Flash, Gemini 1.5 Pro, etc.)
    - OpenRouter (unified access to multiple providers)

    Key capability flags:
    - supports_sampler_controls: If False, temperature/top_p are disabled via
      LangChain's disabled_params.
    - supports_structured_outputs: If False, response_format is disabled.
    - supports_reasoning_effort: If True, reasoning_effort parameter is enabled.
    - is_reasoning_model: Indicates o1, o3, gpt-5, or similar reasoning families.
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

    # Prompt caching support (explicit cache_control breakpoints)
    supports_prompt_caching: bool = False

    # Extended context window support (for Claude, Gemini)
    max_context_tokens: int = 128000

    # Hard per-model output-token ceiling enforced by the provider API.
    # ``None`` means unknown: no clamping is applied at request-build time.
    max_output_tokens: int | None = None


def _norm(name: str) -> str:
    """Normalize a model name for prefix matching."""
    return name.strip().lower()


# ---------------------------------------------------------------------------
# Provider-level capability defaults. Each model entry in the registry below
# only declares fields that *differ* from its provider default.
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
    supports_prompt_caching=True,
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

_CUSTOM_BASE: dict = dict(
    provider="custom",
    supports_responses_api=False,
    supports_chat_completions=True,
    api_preference="langchain",
    is_reasoning_model=False,
    supports_reasoning_effort=False,
    supports_developer_messages=True,
    supports_image_input=True,
    supports_image_detail=False,
    supports_structured_outputs=True,
    supports_function_calling=False,
    supports_sampler_controls=True,
    max_context_tokens=128000,
)


# ---------------------------------------------------------------------------
# Static model registry. Each entry is (prefixes, family, base, overrides).
# Order matters: more specific prefixes MUST come before less specific ones.
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: list[tuple[tuple[str, ...], str, dict, dict]] = [
    # --- OpenAI GPT-5.4 family ---
    (
        ("gpt-5.4-pro",),
        "gpt-5.4-pro",
        _OPENAI_REASONING_BASE,
        dict(
            supports_chat_completions=False,
            max_context_tokens=1050000,
            supports_structured_outputs=False,
        ),
    ),
    (
        ("gpt-5.4-mini",),
        "gpt-5.4-mini",
        _OPENAI_REASONING_BASE,
        dict(
            max_context_tokens=400000,
        ),
    ),
    (
        ("gpt-5.4-nano",),
        "gpt-5.4-nano",
        _OPENAI_REASONING_BASE,
        dict(
            max_context_tokens=400000,
        ),
    ),
    (
        ("gpt-5.4",),
        "gpt-5.4",
        _OPENAI_REASONING_BASE,
        dict(
            supports_chat_completions=False,
            max_context_tokens=1050000,
        ),
    ),
    # --- OpenAI GPT-5.3 family ---
    (
        ("gpt-5.3-chat",),
        "gpt-5.3-chat",
        _OPENAI_STANDARD_BASE,
        dict(
            max_context_tokens=128000,
        ),
    ),
    (
        ("gpt-5.3-codex",),
        "gpt-5.3-codex",
        _OPENAI_REASONING_BASE,
        dict(
            supports_chat_completions=False,
            max_context_tokens=400000,
        ),
    ),
    (
        ("gpt-5.3",),
        "gpt-5.3",
        _OPENAI_REASONING_BASE,
        dict(
            supports_chat_completions=False,
            max_context_tokens=400000,
        ),
    ),
    # --- OpenAI GPT-5.2 family ---
    (
        ("gpt-5.2",),
        "gpt-5.2",
        _OPENAI_REASONING_BASE,
        dict(
            supports_chat_completions=False,
            max_context_tokens=400000,
        ),
    ),
    # --- OpenAI GPT-5.1 family ---
    (
        ("gpt-5.1",),
        "gpt-5.1",
        _OPENAI_REASONING_BASE,
        dict(
            supports_chat_completions=False,
            max_context_tokens=400000,
        ),
    ),
    # --- OpenAI GPT-5 family ---
    (
        ("gpt-5",),
        "gpt-5",
        _OPENAI_REASONING_BASE,
        dict(
            supports_chat_completions=False,
            max_context_tokens=400000,
        ),
    ),
    # --- OpenAI o-series reasoning models ---
    (("o4-mini",), "o4-mini", _OPENAI_REASONING_BASE, {}),
    (("o4",), "o4", _OPENAI_REASONING_BASE, {}),
    (("o3-pro",), "o3-pro", _OPENAI_REASONING_BASE, {}),
    (
        ("o3-mini",),
        "o3-mini",
        _OPENAI_REASONING_BASE,
        dict(
            supports_image_input=False,
            supports_image_detail=False,
        ),
    ),
    # o3 (not o3-mini, not o3-pro) — uses a custom match function below
    (
        ("o1-mini",),
        "o1-mini",
        _OPENAI_REASONING_BASE,
        dict(
            supports_responses_api=False,
            api_preference="chat_completions",
            supports_reasoning_effort=False,
            supports_developer_messages=False,
            supports_image_input=False,
            supports_image_detail=False,
            supports_structured_outputs=False,
            supports_function_calling=False,
        ),
    ),
    # o1 (not o1-mini) — uses a custom match function below
    # --- OpenAI GPT-4o / GPT-4.1 ---
    (("gpt-4o",), "gpt-4o", _OPENAI_STANDARD_BASE, {}),
    (
        ("gpt-4.1-mini",),
        "gpt-4.1-mini",
        _OPENAI_STANDARD_BASE,
        dict(
            max_context_tokens=1050000,
        ),
    ),
    (
        ("gpt-4.1-nano",),
        "gpt-4.1-nano",
        _OPENAI_STANDARD_BASE,
        dict(
            max_context_tokens=1050000,
        ),
    ),
    (
        ("gpt-4.1",),
        "gpt-4.1",
        _OPENAI_STANDARD_BASE,
        dict(
            api_preference="langchain",
            max_context_tokens=1050000,
        ),
    ),
    # --- Anthropic Claude models (most-specific first) ---
    (
        ("claude-opus-4-7", "claude-opus-4.7"),
        "claude-opus-4.7",
        _ANTHROPIC_BASE,
        dict(
            max_context_tokens=1000000,
        ),
    ),
    (("claude-opus-4-6", "claude-opus-4.6"), "claude-opus-4.6", _ANTHROPIC_BASE, {}),
    (
        ("claude-opus-4-5", "claude-opus-4.5"),
        "claude-opus-4.5",
        _ANTHROPIC_BASE,
        dict(
            max_output_tokens=64000,
        ),
    ),
    (
        ("claude-opus-4-1", "claude-opus-4.1"),
        "claude-opus-4.1",
        _ANTHROPIC_BASE,
        dict(
            max_output_tokens=32000,
        ),
    ),
    (
        ("claude-opus-4",),
        "claude-opus-4",
        _ANTHROPIC_BASE,
        dict(
            max_output_tokens=32000,
        ),
    ),
    (
        ("claude-sonnet-4-6", "claude-sonnet-4.6"),
        "claude-sonnet-4.6",
        _ANTHROPIC_BASE,
        dict(
            max_output_tokens=64000,
        ),
    ),
    (
        ("claude-sonnet-4-5", "claude-sonnet-4.5"),
        "claude-sonnet-4.5",
        _ANTHROPIC_BASE,
        dict(
            max_output_tokens=64000,
        ),
    ),
    (
        ("claude-sonnet-4",),
        "claude-sonnet-4",
        _ANTHROPIC_BASE,
        dict(
            max_output_tokens=64000,
        ),
    ),
    (
        ("claude-haiku-4-5", "claude-haiku-4.5"),
        "claude-haiku-4.5",
        _ANTHROPIC_BASE,
        dict(
            max_output_tokens=64000,
        ),
    ),
    (
        ("claude-3-7-sonnet", "claude-3.7-sonnet"),
        "claude-3.7-sonnet",
        _ANTHROPIC_BASE,
        dict(
            max_output_tokens=64000,
        ),
    ),
    (
        ("claude-3-5-sonnet", "claude-3.5-sonnet"),
        "claude-3.5-sonnet",
        _ANTHROPIC_BASE,
        dict(
            max_output_tokens=8192,
        ),
    ),
    (
        ("claude-3-5-haiku", "claude-3.5-haiku"),
        "claude-3.5-haiku",
        _ANTHROPIC_BASE,
        dict(
            max_output_tokens=8192,
        ),
    ),
    (("claude-3-opus",), "claude-3-opus", _ANTHROPIC_BASE, {}),
    (("claude-3-sonnet",), "claude-3-sonnet", _ANTHROPIC_BASE, {}),
    (("claude-3-haiku",), "claude-3-haiku", _ANTHROPIC_BASE, {}),
    (("claude",), "claude", _ANTHROPIC_BASE, {}),
    # --- Google Gemma models (via Gemini API) ---
    (
        ("gemma-4-31b-it",),
        "gemma-4-31b",
        _GOOGLE_BASE,
        dict(
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            max_context_tokens=262144,
        ),
    ),
    (
        ("gemma-4-26b-a4b-it",),
        "gemma-4-26b-moe",
        _GOOGLE_BASE,
        dict(
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            max_context_tokens=262144,
        ),
    ),
    (
        ("gemma",),
        "gemma",
        _GOOGLE_BASE,
        dict(
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            max_context_tokens=262144,
        ),
    ),
    # --- Google Gemini models (most-specific first) ---
    (
        ("gemini-3.1-pro-preview", "gemini-3.1-pro"),
        "gemini-3.1-pro",
        _GOOGLE_BASE,
        dict(
            is_reasoning_model=True,
            max_context_tokens=1048576,
        ),
    ),
    (
        ("gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite"),
        "gemini-3.1-flash-lite",
        _GOOGLE_BASE,
        dict(
            max_context_tokens=1048576,
        ),
    ),
    (
        ("gemini-3-pro", "gemini-3.0-pro"),
        "gemini-3-pro",
        _GOOGLE_BASE,
        dict(
            is_reasoning_model=True,
            max_context_tokens=1048576,
        ),
    ),
    (
        ("gemini-3-flash-preview", "gemini-3.0-flash-preview", "gemini-3-flash"),
        "gemini-3-flash",
        _GOOGLE_BASE,
        dict(
            is_reasoning_model=True,
            max_context_tokens=1048576,
        ),
    ),
    (
        ("gemini-2.5-pro", "gemini-2-5-pro"),
        "gemini-2.5-pro",
        _GOOGLE_BASE,
        dict(
            is_reasoning_model=True,
            max_context_tokens=1048576,
        ),
    ),
    (
        ("gemini-2.5-flash-lite", "gemini-2-5-flash-lite"),
        "gemini-2.5-flash-lite",
        _GOOGLE_BASE,
        dict(
            max_context_tokens=1048576,
        ),
    ),
    (
        ("gemini-2.5-flash", "gemini-2-5-flash"),
        "gemini-2.5-flash",
        _GOOGLE_BASE,
        dict(
            is_reasoning_model=True,
            max_context_tokens=1048576,
        ),
    ),
    (
        ("gemini-2.0-flash", "gemini-2-flash", "gemini-2.0"),
        "gemini-2.0-flash",
        _GOOGLE_BASE,
        {},
    ),
    (
        ("gemini-1.5-pro", "gemini-1-5-pro"),
        "gemini-1.5-pro",
        _GOOGLE_BASE,
        dict(
            max_context_tokens=2000000,
        ),
    ),
    (("gemini-1.5-flash", "gemini-1-5-flash"), "gemini-1.5-flash", _GOOGLE_BASE, {}),
    (("gemini",), "gemini", _GOOGLE_BASE, {}),
]
