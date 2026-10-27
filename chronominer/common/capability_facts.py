"""The settled vendor facts that :mod:`.capability_table` applies to its base.

The base is the family table the tools share. Two lists change it, each record
naming its source (the lab's page, read on the date the record states):

- :data:`STRUCTURE` adds, drops or re-matches rules, so that every id below
  resolves to a rule of its own. A new rule copies the rule that resolves
  *like*, or starts empty.
- :data:`FACTS` sets one field to one value in the rule that resolves each id.

Structure records apply first, in order, then the facts. An id names a model
as a user would; the record changes the whole rule that resolves it. A later
change in the vendors' documentation is a one-line edit here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "FACTS",
    "STRUCTURE",
    "AddRule",
    "DropRule",
    "Fact",
    "Rematch",
]


@dataclass(frozen=True)
class AddRule:
    """Insert a rule *family* before the rule that resolves *before*.

    The new rule copies the rule that resolves *like* (fields, provider,
    predicate), or starts with no fields; the given matchers replace the
    copied ones.
    """

    family: str
    before: str
    source: str
    like: str | None = None
    prefixes: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()
    exact: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class DropRule:
    """Remove the rule that resolves *id*."""

    id: str
    source: str


@dataclass(frozen=True)
class Rematch:
    """Replace the matchers of the rule that resolves *id*; None keeps one."""

    id: str
    source: str
    prefixes: tuple[str, ...] | None = None
    exact: tuple[str, ...] | None = None
    exclude: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Fact:
    """*field* takes *value* in the rule that resolves each id of *ids*."""

    ids: tuple[str, ...]
    field: str
    value: Any
    source: str


_OA = "developers.openai.com/api/docs/"
_AN = "platform.claude.com/docs/en/"
_GG = "ai.google.dev/gemini-api/docs/"
_OR = "openrouter.ai/api/v1/models, entry "
_ZAI = "docs.z.ai/guides/vlm/"

_NONE_TO_MAX = ("none", "low", "medium", "high", "xhigh", "max")
_LOW_TO_MAX = ("low", "medium", "high", "xhigh", "max")
_NONE_TO_XHIGH = ("none", "low", "medium", "high", "xhigh")
_NONE_TO_HIGH = ("none", "low", "medium", "high")
_MINIMAL_TO_HIGH = ("minimal", "low", "medium", "high")
_LOW_TO_XHIGH = ("low", "medium", "high", "xhigh")
_PRO = ("medium", "high", "xhigh")
_LOW_TO_HIGH = ("low", "medium", "high")

_GPT6 = ("gpt-6-luna", "gpt-6-sol", "gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-terra")
_GPT56 = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6")
_CODEX = ("gpt-5.2-codex", "gpt-5.3-codex")
_OLD_CHAT = ("gpt-5-chat-latest", "gpt-5.1-chat-latest", "gpt-5.2-chat-latest")
_CHAT = (*_OLD_CHAT, "gpt-5.3-chat-latest")
_NO_SAMPLERS_CLAUDE = (
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-fable-5",
    "claude-sonnet-5",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
)
_GLM_DIRECT = (
    "glm-5.3-flash",
    "glm-5.3-flashx",
    "glm-5v-turbo",
    "glm-4.6v",
    "glm-4.5v",
)
_GLM_ROUTED = (
    "z-ai/glm-5.3-flash",
    "z-ai/glm-5v-turbo",
    "z-ai/glm-4.6v",
    "z-ai/glm-4.5v",
)

STRUCTURE: tuple[AddRule | DropRule | Rematch, ...] = (
    # OpenAI
    DropRule("gpt-5.3", _OA + "pricing: no gpt-5.3 row, no model page"),
    Rematch(
        "gpt-5",
        _OA + "models: gpt-5.3 and gpt-5.3-chat are no OpenAI ids",
        prefixes=("gpt-5-",),
        exact=("gpt-5",),
    ),
    AddRule(
        "gpt-6.1-terra",
        "gpt-6.1-sol",
        _OA + "models/gpt-6.1-terra: text and image input, structured outputs",
        like="gpt-6.1-sol",
        prefixes=("gpt-6.1-terra",),
    ),
    AddRule(
        "gpt-5.2-pro",
        "gpt-5.2",
        _OA + "models/gpt-5.2-pro: Structured outputs / Not supported",
        like="gpt-5.2",
        prefixes=("gpt-5.2-pro",),
    ),
    AddRule(
        "gpt-5-mini",
        "gpt-5",
        _OA + "guides/prompt-caching: extended retention lists gpt-5, not -mini",
        like="gpt-5",
        prefixes=("gpt-5-mini",),
    ),
    AddRule(
        "gpt-5-nano",
        "gpt-5",
        _OA + "guides/prompt-caching: extended retention lists gpt-5, not -nano",
        like="gpt-5",
        prefixes=("gpt-5-nano",),
    ),
    AddRule(
        "o4-mini-deep-research",
        "o4-mini",
        _OA + "models/o4-mini-deep-research: Structured outputs / Not supported",
        like="o4-mini",
        prefixes=("o4-mini-deep-research",),
    ),
    AddRule(
        "o3-deep-research",
        "o3",
        _OA + "models/o3-deep-research: Structured outputs / Not supported",
        like="o3",
        prefixes=("o3-deep-research",),
    ),
    AddRule(
        "gpt-4.1-mini",
        "gpt-4.1",
        _OA + "guides/prompt-caching: extended retention lists gpt-4.1 only",
        like="gpt-4.1",
        prefixes=("gpt-4.1-mini",),
    ),
    AddRule(
        "gpt-4.1-nano",
        "gpt-4.1",
        _OA + "guides/prompt-caching: extended retention lists gpt-4.1 only",
        like="gpt-4.1",
        prefixes=("gpt-4.1-nano",),
    ),
    AddRule(
        "gpt-4.5",
        "claude-opus-5-5",
        _OA + "models/gpt-4.5-preview: 128,000 context / 16,384 max output",
        like="gpt-4o",
        prefixes=("gpt-4.5",),
    ),
    AddRule(
        "gpt-4-turbo",
        "claude-opus-5-5",
        _OA + "models/gpt-4-turbo: Image input; Structured outputs / Not supported",
        like="gpt-4o",
        prefixes=("gpt-4-turbo",),
    ),
    AddRule(
        "gpt-4",
        "claude-opus-5-5",
        _OA + "models/gpt-4: 8,192 context / 8,192 max output",
        like="gpt-4o",
        prefixes=("gpt-4",),
    ),
    # Google
    DropRule("gemma-3-27b-it", _GG + "gemma: the API serves only two Gemma 4 ids"),
    AddRule(
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        _GG + "models/gemini-3.8-flash: Thinking / Supported (low, medium, high)",
        like="gemini-3.7-flash",
        prefixes=("gemini-3.8-flash", "gemini-3-8-flash"),
    ),
    Rematch(
        "gemini-3.1-pro-preview",
        _GG + "models/gemini-3.1-pro-preview: Input token limit 1,048,576",
        prefixes=("gemini-3.1-pro", "gemini-3-1-pro"),
    ),
    DropRule(
        "gemini-3.1-flash-lite-preview",
        _GG + "models/gemini-3.1-flash-lite-preview: Thinking / Supported",
    ),
    # Z.ai GLM, called directly
    *(
        AddRule(
            family, "deepseek/deepseek-r1", _ZAI + family + ".md", prefixes=(family,)
        )
        for family in ("glm-5.3-flashx", "glm-5v-turbo", "glm-4.6v", "glm-4.5v")
    ),
    AddRule(
        "glm-5.3-flash",
        "deepseek/deepseek-r1",
        _ZAI + "glm-5.3-flash.md (glm-5.3-flash:batch is not documented)",
        prefixes=("glm-5.3-flash-",),
        exact=("glm-5.3-flash",),
    ),
    # OpenRouter
    *(
        AddRule(
            family,
            "deepseek/deepseek-r1",
            _OR + "z-ai/" + family,
            like="vendor/model",
            contains=(family,),
            exclude=(":batch", "flashx"),
        )
        for family in ("glm-5.3-flash", "glm-5v-turbo", "glm-4.6v", "glm-4.5v")
    ),
    DropRule("openai/gpt-5.3-chat", _OR + "list: no openai/gpt-5.3-chat entry"),
    Rematch(
        "openai/gpt-5",
        _OR + "list: no openai/gpt-5.3-chat entry",
        exclude=("gpt-5.3-chat",),
    ),
    AddRule(
        "openrouter-gpt6",
        "openai/gpt-5",
        _OR + "openai/gpt-6-luna: reasoning, reasoning_effort; no temperature",
        like="vendor/model",
        contains=("/gpt-6",),
    ),
    Rematch(
        "openai/o3-mini",
        _OR + "openai/o4-mini: input_modalities image, text, file",
        exclude=("/o4-mini",),
    ),
    AddRule(
        "openrouter-gpt-4o",
        "vendor/model",
        _OR + "openai/gpt-4o: frequency_penalty, presence_penalty",
        like="vendor/model",
        contains=("/gpt-4o",),
    ),
    AddRule(
        "openrouter-claude-no-samplers",
        "anthropic/claude-opus-5.5",
        _OR + "anthropic/claude-opus-5.5: parameters without temperature, top_p",
        like="anthropic/claude-opus-5.5",
        contains=(
            "claude-opus-5.5",
            "claude-opus-5-5",
            "claude-haiku-5.5",
            "claude-haiku-5-5",
        ),
    ),
    AddRule(
        "openrouter-qwen-vl",
        "qwen/qwen3-vl-235b-a22b-instruct",
        _OR + "qwen/qwen3-vl-235b-a22b-instruct: ctx 262144, no reasoning",
        like="qwen/qwen3-vl-235b-a22b-instruct",
        contains=("qwen3-vl-235b-a22b-instruct",),
    ),
    AddRule(
        "openrouter-qwen3.5-flash",
        "qwen/qwen3.5-flash-02-23",
        _OR + "qwen/qwen3.5-flash-02-23: ctx 1000000, max out 65536",
        like="qwen/qwen3.5-flash-02-23",
        contains=("qwen3.5-flash",),
    ),
)
"""Rules added, dropped or re-matched, in the order they apply."""

FACTS: tuple[Fact, ...] = (
    # OpenAI limits
    Fact(
        _CHAT, "max_context_tokens", 128000, _OA + "models/gpt-5.3-chat-latest: 128,000"
    ),
    Fact(
        ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"),
        "max_context_tokens",
        1047576,
        _OA + "models/gpt-4.1: 1,047,576 context window",
    ),
    # OpenAI chat-latest models are not reasoning models
    Fact(_OLD_CHAT, "is_reasoning_model", False, _OA + "models/gpt-5.1-chat-latest"),
    Fact(
        _OLD_CHAT, "supports_reasoning_effort", False, _OA + "models/gpt-5-chat-latest"
    ),
    Fact(
        ("gpt-5.3-chat-latest",),
        "supports_temperature",
        False,
        _OA + "models/gpt-5.3-chat-latest: no sampling stated (undocumented: omit)",
    ),
    Fact(
        ("gpt-5.3-chat-latest",),
        "supports_top_p",
        False,
        _OA + "models/gpt-5.3-chat-latest: no sampling stated (undocumented: omit)",
    ),
    # OpenAI codex models
    Fact(_CODEX, "supports_vision", True, _OA + "models/gpt-5.3-codex: Text, Image"),
    Fact(
        _CODEX,
        "supports_image_detail",
        True,
        _OA + "guides/images-vision: detail defaults to auto (original undocumented)",
    ),
    Fact(_CODEX, "is_reasoning_model", True, _OA + "models/gpt-5.3-codex"),
    Fact(_CODEX, "supports_reasoning_effort", True, _OA + "models/gpt-5.3-codex"),
    Fact(
        _CODEX,
        "reasoning_efforts",
        _LOW_TO_XHIGH,
        _OA + "models/gpt-5.3-codex: low, medium, high, and xhigh",
    ),
    Fact(_CODEX, "supports_temperature", False, _OA + "guides/latest-model"),
    Fact(_CODEX, "supports_top_p", False, _OA + "guides/latest-model"),
    # OpenAI structured outputs
    Fact(
        ("gpt-5.2-pro", "gpt-5.4-pro"),
        "supports_structured_output",
        False,
        _OA + "models/gpt-5.4-pro: Structured outputs / Not supported",
    ),
    Fact(
        ("gpt-5.2-pro", "gpt-5.4-pro"),
        "supports_json_mode",
        False,
        _OA + "models/gpt-5.2-pro: Structured outputs / Not supported",
    ),
    Fact(
        ("o1", "o1-pro", "o3"),
        "supports_structured_output",
        True,
        _OA + "models/o1, o1-pro, o3: Structured outputs / Supported",
    ),
    Fact(
        ("o4-mini-deep-research",),
        "supports_structured_output",
        False,
        _OA + "models/o4-mini-deep-research: Structured outputs / Not supported",
    ),
    # OpenAI verbosity on the GPT-5 and GPT-6 reasoning models
    Fact(
        (
            *_GPT6,
            *_GPT56,
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.2",
            "gpt-5.1",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
        ),
        "supports_text_verbosity",
        True,
        _OA + "reference/responses/create: verbosity low, medium, high",
    ),
    # OpenAI image detail
    Fact(
        ("gpt-5.4-mini", "gpt-5.4-nano"),
        "supports_original_image_detail",
        True,
        _OA + "guides/images-vision: gpt-5.4-mini, -nano: original, 10,000 patches",
    ),
    # OpenAI reasoning efforts
    Fact(
        ("gpt-6-astra", "gpt-6.1-sol"),
        "reasoning_efforts",
        _LOW_TO_MAX,
        _OA + "guides/reasoning: no none on Astra; no none or minimal on 6.1 Sol",
    ),
    Fact(
        ("gpt-6-luna", "gpt-6-sol", *_GPT56),
        "reasoning_efforts",
        _NONE_TO_MAX,
        _OA + "models/gpt-6-luna: none, low, medium (default), high, xhigh, max",
    ),
    Fact(
        ("gpt-6.1-terra",),
        "reasoning_efforts",
        _NONE_TO_MAX,
        _OA + "models/gpt-6.1-terra: none, low, medium (default), high, xhigh, max",
    ),
    Fact(
        ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.2"),
        "reasoning_efforts",
        _NONE_TO_XHIGH,
        _OA + "models/gpt-5.5: none, low, medium (default), high and xhigh",
    ),
    Fact(
        ("gpt-5", "gpt-5-mini", "gpt-5-nano"),
        "reasoning_efforts",
        _MINIMAL_TO_HIGH,
        _OA + "models/gpt-5: minimal, low, medium, and high",
    ),
    Fact(
        ("gpt-5.1",),
        "reasoning_efforts",
        _NONE_TO_HIGH,
        _OA + "models/gpt-5.1: none (default), low, medium, and high",
    ),
    Fact(
        ("gpt-5.5-pro", "gpt-5.4-pro", "gpt-5.2-pro"),
        "reasoning_efforts",
        _PRO,
        _OA + "models/gpt-5.5-pro: medium, high (default) and xhigh",
    ),
    # OpenAI Flex processing
    Fact(
        (
            *_GPT6,
            *_GPT56,
            "gpt-5.5",
            "gpt-5.5-pro",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.4-pro",
            "gpt-5.2",
            "gpt-5.1",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "o3",
            "o4-mini",
        ),
        "supports_flex_tier",
        True,
        _OA + "guides/flex-processing: models with a Flex price on the pricing page",
    ),
    # OpenAI prompt caching
    Fact(
        (*_GPT6, *_GPT56),
        "prompt_cache_parameter",
        "prompt_cache_options",
        _OA + "guides/prompt-caching: GPT-5.6 and later use prompt_cache_options.ttl",
    ),
    Fact(
        (
            "gpt-5.5",
            "gpt-5.5-pro",
            "gpt-5.4",
            "gpt-5.2",
            "gpt-5.1",
            "gpt-5.1-chat-latest",
            "gpt-5",
            "gpt-4.1",
        ),
        "prompt_cache_parameter",
        "prompt_cache_retention",
        _OA + "guides/prompt-caching: extended retention is supported by ...",
    ),
    # OpenAI GPT-4 family
    Fact(
        ("gpt-4-turbo", "gpt-4"),
        "supports_structured_output",
        False,
        _OA + "models/gpt-4-turbo, gpt-4: Structured outputs / Not supported",
    ),
    Fact(
        ("gpt-4-turbo", "gpt-4"),
        "supports_json_mode",
        False,
        _OA + "models/gpt-4-turbo: JSON mode not stated (undocumented: off)",
    ),
    Fact(("gpt-4-turbo",), "max_output_tokens", 4096, _OA + "models/gpt-4-turbo"),
    Fact(("gpt-4",), "supports_vision", False, _OA + "models/gpt-4: no image input"),
    Fact(("gpt-4",), "supports_image_detail", False, _OA + "models/gpt-4"),
    Fact(("gpt-4",), "max_context_tokens", 8192, _OA + "models/gpt-4: 8,192 context"),
    Fact(("gpt-4",), "max_output_tokens", 8192, _OA + "models/gpt-4: 8,192 output"),
    # Anthropic
    Fact(
        _NO_SAMPLERS_CLAUDE,
        "supports_temperature",
        False,
        _AN + "build-with-claude/thinking: temperature with thinking returns 400",
    ),
    Fact(
        (
            "claude-3-7-sonnet",
            "claude-3-5-sonnet",
            "claude-sonnet-4",
            "claude-opus-4",
            "claude-opus-4-1",
        ),
        "supports_structured_output",
        False,
        _AN + "build-with-claude/structured-outputs: not in the supported models",
    ),
    # Google
    Fact(
        ("gemini-2.5-flash",),
        "is_reasoning_model",
        True,
        _GG + "models/gemini-2.5-flash: Thinking / Supported",
    ),
    Fact(
        ("gemini-2.5-flash",),
        "supports_reasoning_effort",
        True,
        _GG + "models/gemini-2.5-flash: Thinking / Supported",
    ),
    Fact(
        ("gemini-2.5-flash-lite",),
        "is_reasoning_model",
        True,
        _GG + "models/gemini-2.5-flash-lite: Thinking / Supported",
    ),
    Fact(
        ("gemini-2.5-pro",),
        "max_context_tokens",
        1048576,
        _GG + "models/gemini-2.5-pro: Input token limit 1,048,576",
    ),
    Fact(
        ("gemini-3.1-pro",),
        "max_context_tokens",
        1048576,
        _GG + "models/gemini-3.1-pro-preview: Input token limit 1,048,576",
    ),
    Fact(
        ("gemini-3.1-pro", "gemini-3.7-flash", "gemini-3.8-flash"),
        "reasoning_efforts",
        _LOW_TO_HIGH,
        _GG + "gemini-3: minimal not supported on 3.1 Pro; 3.8 Flash: low-high",
    ),
    Fact(
        ("gemini-3.1-flash-lite",),
        "reasoning_efforts",
        _MINIMAL_TO_HIGH,
        "docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/"
        "3-1-flash-lite: thinking levels minimal, low, medium, or high",
    ),
    Fact(
        ("gemini-3-preview",),
        "max_context_tokens",
        1000000,
        _GG + "models: no gemini-3-preview id (undocumented: 1,000,000)",
    ),
    Fact(
        ("gemini-3.5-flash",),
        "supports_temperature",
        False,
        _GG + "models/gemini-3.5-flash: routed to gemini-3.6-flash",
    ),
    Fact(
        ("gemini-3.5-flash",),
        "supports_top_p",
        False,
        _GG + "models/gemini-3.5-flash: routed to gemini-3.6-flash",
    ),
    *(
        Fact(
            ("gemini-3-pro-image", "gemini-3.1-flash-image"),
            field,
            value,
            _GG + "models/gemini-3-pro-image: Structured outputs / Not supported",
        )
        for field, value in (
            ("supports_structured_output", False),
            ("supports_json_mode", False),
            ("is_reasoning_model", True),
        )
    ),
    Fact(
        ("gemini-3.1-flash-image",),
        "max_context_tokens",
        131072,
        _GG + "models/gemini-3.1-flash-image: Input token limit 131,072",
    ),
    *(
        Fact(
            ("gemma-4-31b-it", "gemma-4-26b-a4b-it"),
            field,
            value,
            _GG + "gemma/docs/core/gemma_on_gemini_api (undocumented: off)",
        )
        for field, value in (
            ("supports_structured_output", False),
            ("supports_json_mode", False),
            ("supports_media_resolution", False),
        )
    ),
    Fact(
        ("gemma-4-31b-it", "gemma-4-26b-a4b-it"),
        "reasoning_efforts",
        ("minimal", "high"),
        _GG + "gemma_on_gemini_api: thinking level high (on) or minimal (off)",
    ),
    # Z.ai GLM, called directly and through OpenRouter
    *(
        Fact((*_GLM_DIRECT, *_GLM_ROUTED), field, True, _ZAI + "glm-5.3-flash.md")
        for field in ("supports_vision", "supports_json_mode", "is_reasoning_model")
    ),
    Fact(
        ("glm-5.3-flash", "glm-5.3-flashx", "z-ai/glm-5.3-flash"),
        "supports_reasoning_effort",
        True,
        _ZAI + "glm-5.3-flash.md: reasoning_effort: max recommended",
    ),
    Fact(
        ("glm-5.3-flash", "glm-5.3-flashx", "z-ai/glm-5.3-flash"),
        "max_context_tokens",
        1048576,
        _ZAI + "glm-5.3-flash.md: Context Length 1M",
    ),
    Fact(
        ("glm-5.3-flash", "glm-5.3-flashx", "z-ai/glm-5.3-flash"),
        "max_output_tokens",
        131072,
        _ZAI + "glm-5.3-flash.md: Maximum Output Tokens 128K",
    ),
    Fact(
        ("glm-5v-turbo", "z-ai/glm-5v-turbo"),
        "max_context_tokens",
        202752,
        _ZAI + "glm-5v-turbo.md: Context Length 200K",
    ),
    Fact(
        ("glm-5v-turbo", "z-ai/glm-5v-turbo"),
        "max_output_tokens",
        131072,
        _ZAI + "glm-5v-turbo.md: Maximum Output Tokens 128K",
    ),
    Fact(
        ("glm-4.6v", "z-ai/glm-4.6v"),
        "max_context_tokens",
        131072,
        _ZAI + "glm-4.6v.md: Context Length 128K",
    ),
    Fact(
        ("glm-4.6v", "z-ai/glm-4.6v"),
        "max_output_tokens",
        32768,
        _OR + "z-ai/glm-4.6v: max_out=32768",
    ),
    Fact(
        ("glm-4.5v", "z-ai/glm-4.5v"),
        "max_context_tokens",
        65536,
        _OR + "z-ai/glm-4.5v: ctx=65536",
    ),
    Fact(
        ("glm-4.5v", "z-ai/glm-4.5v"),
        "max_output_tokens",
        16384,
        _ZAI + "glm-4.5v.md: Maximum Output Tokens 16K",
    ),
    Fact(
        ("z-ai/glm-5v-turbo", "z-ai/glm-4.6v", "z-ai/glm-4.5v"),
        "supports_structured_output",
        False,
        _OR + "z-ai/glm-4.6v: response_format, no structured_outputs",
    ),
    *(
        Fact(
            ("z-ai/glm-5.3-flash", "z-ai/glm-4.6v", "z-ai/glm-4.5v"),
            field,
            True,
            _OR + "z-ai/glm-4.6v: frequency_penalty, presence_penalty",
        )
        for field in ("supports_frequency_penalty", "supports_presence_penalty")
    ),
    # OpenRouter
    Fact(
        ("deepseek/deepseek-r1",),
        "supports_temperature",
        True,
        _OR + "deepseek/deepseek-r1: temperature, top_p",
    ),
    Fact(
        ("deepseek/deepseek-r1",),
        "supports_top_p",
        True,
        _OR + "deepseek/deepseek-r1: temperature, top_p",
    ),
    Fact(
        ("openai/gpt-5.4",),
        "supports_original_image_detail",
        False,
        _OR + "list: no image detail levels (undocumented: high)",
    ),
    *(
        Fact(
            ("openai/gpt-6-luna",),
            field,
            value,
            _OR + "openai/gpt-6-luna: reasoning, reasoning_effort; no temperature",
        )
        for field, value in (
            ("is_reasoning_model", True),
            ("supports_reasoning_effort", True),
            ("supports_temperature", False),
            ("supports_top_p", False),
        )
    ),
    *(
        Fact(
            ("openai/gpt-4o",),
            field,
            True,
            _OR + "openai/gpt-4o: frequency_penalty, presence_penalty",
        )
        for field in ("supports_frequency_penalty", "supports_presence_penalty")
    ),
    *(
        Fact(
            ("anthropic/claude-opus-5.5",),
            field,
            False,
            _OR + "anthropic/claude-opus-5.5: parameters without temperature, top_p",
        )
        for field in ("supports_temperature", "supports_top_p")
    ),
    *(
        Fact(
            ("qwen/qwen3-vl-235b-a22b-instruct",),
            field,
            value,
            _OR + "qwen/qwen3-vl-235b-a22b-instruct: ctx 262144, max out 32768",
        )
        for field, value in (
            ("is_reasoning_model", False),
            ("supports_reasoning_effort", False),
            ("max_context_tokens", 262144),
            ("max_output_tokens", 32768),
        )
    ),
    *(
        Fact(
            ("qwen/qwen3.5-flash-02-23",),
            field,
            value,
            _OR + "qwen/qwen3.5-flash-02-23 (endpoints): ctx 1000000, max out 65536",
        )
        for field, value in (
            ("max_context_tokens", 1000000),
            ("max_output_tokens", 65536),
        )
    ),
)
"""The settled values, each with its source."""
