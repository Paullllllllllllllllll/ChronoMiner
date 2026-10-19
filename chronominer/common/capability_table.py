"""The model capability table the tools share: a base and the settled facts.

:data:`BASE_FAMILIES` is the shared family table. :data:`FAMILIES` is that base
with the records of :mod:`.capability_facts` applied: structural records
add, drop or re-match rules, and each fact sets one field of the rule that
resolves a named id to the value the lab's documentation states (read on
13.10.2026). Ids no rule matches, and ids no lab documents (``gpt-5.3``,
``gpt-5.3-chat``, ``glm-flash-latest``), get the conservative profile of the
engine.

Rules are matched in order against the lowercased id after a ``provider:`` or
``models/`` prefix is removed; the first match wins, so specific prefixes
precede general ones. ``vendor/model`` ids resolve through the OpenRouter
rules. :func:`build_table` adds a tool's :class:`TableExtension`: rules
matched before the shared ones and extra keys for the ``extras`` of named
families (ChronoTranscriber's audio flags use it). :data:`AUDIO_FAMILIES`
lists the families that take audio input and :data:`SPEECH_TO_TEXT_FAMILIES`
those served only by OpenAI's transcription endpoint.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from .capabilities import (
    OPENROUTER,
    CapabilityTable,
    FamilyRule,
    ProviderRule,
    detect_provider,
    split_model_id,
)
from .capability_facts import FACTS, STRUCTURE, AddRule, DropRule, Fact, Rematch

__all__ = [
    "AUDIO_FAMILIES",
    "BASE_FAMILIES",
    "FAMILIES",
    "ORIGINAL_MAX_SIDE",
    "OUTPUT_LIMIT_FAMILIES",
    "PROVIDERS",
    "SPEECH_TO_TEXT_FAMILIES",
    "TABLE",
    "TableExtension",
    "apply_facts",
    "build_table",
    "rule_index",
]

ORIGINAL_MAX_SIDE = "image_original_max_side_px"
"""The ``Capabilities.extras`` key of the default long-side limit in pixels of
OpenAI pages sent at ``original`` detail."""

_Fields = Mapping[str, Any]

_NO_SAMPLING: _Fields = {
    "supports_temperature": False,
    "supports_top_p": False,
    "supports_frequency_penalty": False,
    "supports_presence_penalty": False,
}
_REASONING: _Fields = {"is_reasoning_model": True, "supports_reasoning_effort": True}
_TEXT_ONLY: _Fields = {"supports_vision": False, "supports_image_detail": False}
_NO_STRUCTURE: _Fields = {
    "supports_structured_output": False,
    "supports_json_mode": False,
}

_OPENAI_REASONING: _Fields = {
    "provider_name": "openai",
    "supports_vision": True,
    "supports_image_detail": True,
    "default_image_detail": "high",
    "supports_structured_output": True,
    "supports_json_mode": True,
    **_REASONING,
    **_NO_SAMPLING,
    "max_context_tokens": 200000,
    "max_output_tokens": 100000,
}
_OPENAI_STANDARD: _Fields = {
    "provider_name": "openai",
    "supports_vision": True,
    "supports_image_detail": True,
    "default_image_detail": "high",
    "supports_structured_output": True,
    "supports_json_mode": True,
    "max_context_tokens": 128000,
    "max_output_tokens": 16384,
}
_OPENAI_AUDIO: _Fields = {
    "provider_name": "openai",
    "default_image_detail": "high",
    "supports_top_p": False,
    "max_context_tokens": 16000,
    "max_output_tokens": 2000,
}
_ANTHROPIC: _Fields = {
    "provider_name": "anthropic",
    "supports_vision": True,
    "supports_structured_output": True,
    "supports_json_mode": True,
    "max_context_tokens": 200000,
    "max_output_tokens": 8192,
}
_GOOGLE: _Fields = {
    "provider_name": "google",
    "supports_vision": True,
    "supports_media_resolution": True,
    "supports_structured_output": True,
    "supports_json_mode": True,
    "max_context_tokens": 1000000,
    "max_output_tokens": 8192,
}
_GEMINI = _GOOGLE
_OPENROUTER: _Fields = {
    "provider_name": "openrouter",
    "supports_vision": True,
    "supports_structured_output": True,
    "supports_json_mode": True,
    "max_context_tokens": 128000,
    "max_output_tokens": 4096,
}

_LARGE_GPT: _Fields = {"max_context_tokens": 1050000, "max_output_tokens": 128000}
_GPT5: _Fields = {"max_context_tokens": 400000, "max_output_tokens": 128000}
_GPT5_CHAT: _Fields = {**_GPT5, "max_output_tokens": 16384}
_ORIGINAL: _Fields = {"supports_original_image_detail": True}
_PATCH_CAP: _Fields = {**_ORIGINAL, "image_original_patch_cap_30k": True}
_ADAPTIVE: _Fields = {
    **_REASONING,
    "uses_adaptive_thinking": True,
    "supports_top_p": False,
    "max_context_tokens": 1000000,
    "max_output_tokens": 128000,
}
_CLAUDE_45: _Fields = {**_REASONING, "supports_top_p": False}
_GEMINI_THINKING: _Fields = {**_REASONING, "max_output_tokens": 65536}
_GEMMA: _Fields = {**_REASONING, "max_context_tokens": 262144}


def _rule(
    family: str, base: _Fields, prefixes: tuple[str, ...], **fields: Any
) -> FamilyRule:
    """A rule matching ids that start with one of *prefixes*."""
    return FamilyRule(family, {**base, **fields}, prefixes)


def _rules(base: _Fields, *families: str, **fields: Any) -> tuple[FamilyRule, ...]:
    """One rule per family name, matched as a prefix."""
    return tuple(_rule(family, base, (family,), **fields) for family in families)


def _dotted(family: str, base: _Fields, **fields: Any) -> FamilyRule:
    """A rule that also matches the dashed spelling (``4-5`` for ``4.5``)."""
    spellings = (family.replace(".", "-"), family)
    return _rule(family, base, tuple(dict.fromkeys(spellings)), **fields)


def _contains(*parts: str) -> Callable[[str], bool]:
    return lambda name: any(part in name for part in parts)


def _tail(part: str, vendor: str) -> Callable[[str], bool]:
    """Match *part* in the model segment of an id or *vendor* anywhere."""
    return lambda name: part in name.rsplit("/", 1)[-1] or vendor in name


def _routed(
    family: str, fields: _Fields, match: Callable[[str], bool] | None = None
) -> FamilyRule:
    """A rule for OpenRouter ``vendor/model`` ids that *match* selects."""

    def predicate(name: str) -> bool:
        return "/" in name and (match is None or match(name))

    return FamilyRule(
        family,
        {**_OPENROUTER, **fields},
        provider=OPENROUTER,
        predicate=predicate,
    )


def _openai_rules() -> tuple[FamilyRule, ...]:
    return (
        *_rules(
            _OPENAI_REASONING,
            "gpt-6.1-sol",
            "gpt-6-astra",
            "gpt-6-sol",
            **_LARGE_GPT,
            **_PATCH_CAP,
        ),
        *_rules(
            _OPENAI_REASONING,
            "gpt-6-luna",
            **_LARGE_GPT,
            **_PATCH_CAP,
            extras={ORIGINAL_MAX_SIDE: 3000},
        ),
        *_rules(
            _OPENAI_REASONING,
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.6",
            **_LARGE_GPT,
            **_PATCH_CAP,
        ),
        *_rules(_OPENAI_REASONING, "gpt-5.5-pro", **_LARGE_GPT),
        *_rules(_OPENAI_REASONING, "gpt-5.5", "gpt-5.4-pro", **_LARGE_GPT, **_ORIGINAL),
        *_rules(_OPENAI_REASONING, "gpt-5.4-mini", "gpt-5.4-nano", **_GPT5),
        *_rules(_OPENAI_REASONING, "gpt-5.4", **_LARGE_GPT, **_ORIGINAL),
        *_rules(_OPENAI_STANDARD, "gpt-5.3-codex", **_GPT5, **_TEXT_ONLY),
        *_rules(_OPENAI_STANDARD, "gpt-5.3-chat-latest", **_GPT5_CHAT),
        *_rules(_OPENAI_STANDARD, "gpt-5.3", **_GPT5),
        *_rules(_OPENAI_STANDARD, "gpt-5.2-codex", **_GPT5, **_TEXT_ONLY),
        *_rules(
            _OPENAI_REASONING,
            "gpt-5.2-chat-latest",
            "gpt-5.1-chat-latest",
            "gpt-5-chat-latest",
            **_GPT5_CHAT,
        ),
        *_rules(_OPENAI_REASONING, "gpt-5.2", "gpt-5.1", "gpt-5", **_GPT5),
        _rule("o4-mini", _OPENAI_REASONING, ("o4-mini", "o4")),
        *_rules(_OPENAI_REASONING, "o3-pro"),
        *_rules(_OPENAI_REASONING, "o3-mini", **_TEXT_ONLY),
        *_rules(_OPENAI_REASONING, "o1-pro", supports_structured_output=False),
        *_rules(
            _OPENAI_REASONING,
            "o1-mini",
            supports_reasoning_effort=False,
            **_TEXT_ONLY,
            **_NO_STRUCTURE,
            max_context_tokens=128000,
            max_output_tokens=65536,
        ),
        FamilyRule(
            "o3",
            {**_OPENAI_REASONING, "supports_structured_output": False},
            ("o3-",),
            exact=("o3",),
        ),
        _rule(
            "o1",
            _OPENAI_REASONING,
            ("o1",),
            supports_reasoning_effort=False,
            supports_structured_output=False,
        ),
        _rule(
            "openai-audio",
            _OPENAI_AUDIO,
            ("gpt-transcribe", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"),
        ),
        _rule("openai-audio", _OPENAI_AUDIO, ("whisper-1",)),
        *_rules(_OPENAI_STANDARD, "gpt-4o"),
        *_rules(
            _OPENAI_STANDARD,
            "gpt-4.1",
            max_context_tokens=1000000,
            max_output_tokens=32768,
        ),
    )


def _anthropic_rules() -> tuple[FamilyRule, ...]:
    high_res = {**_ADAPTIVE, "image_high_res_tier": True}
    return (
        _dotted("claude-opus-5.5", _ANTHROPIC, **high_res),
        *_rules(_ANTHROPIC, "claude-opus-5", "claude-fable-5", **high_res),
        _dotted("claude-opus-4.8", _ANTHROPIC, **high_res),
        _dotted("claude-sonnet-5.5", _ANTHROPIC, **high_res),
        *_rules(_ANTHROPIC, "claude-sonnet-5", **high_res),
        _dotted("claude-haiku-5.5", _ANTHROPIC, **high_res),
        _dotted("claude-opus-4.7", _ANTHROPIC, **high_res),
        _dotted("claude-opus-4.6", _ANTHROPIC, **_ADAPTIVE),
        _dotted("claude-sonnet-4.6", _ANTHROPIC, **_ADAPTIVE),
        _dotted("claude-opus-4.5", _ANTHROPIC, **_CLAUDE_45, max_output_tokens=64000),
        _dotted("claude-sonnet-4.5", _ANTHROPIC, **_CLAUDE_45, max_output_tokens=64000),
        _dotted("claude-haiku-4.5", _ANTHROPIC, **_CLAUDE_45, max_output_tokens=64000),
        _dotted("claude-opus-4.1", _ANTHROPIC, **_REASONING, max_output_tokens=32000),
        *_rules(_ANTHROPIC, "claude-sonnet-4", **_REASONING, max_output_tokens=64000),
        *_rules(_ANTHROPIC, "claude-opus-4", **_REASONING, max_output_tokens=32000),
        _dotted("claude-3.7-sonnet", _ANTHROPIC),
        _dotted("claude-3.5-sonnet", _ANTHROPIC),
        _dotted("claude-3.5-haiku", _ANTHROPIC, **_NO_STRUCTURE),
        *_rules(
            _ANTHROPIC,
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            **_NO_STRUCTURE,
            max_output_tokens=4096,
        ),
        *_rules(_ANTHROPIC, "claude"),
    )


def _google_rules() -> tuple[FamilyRule, ...]:
    return (
        _rule("gemma-4-31b", _GOOGLE, ("gemma-4-31b-it",), **_GEMMA),
        _rule("gemma-4-26b-moe", _GOOGLE, ("gemma-4-26b-a4b-it",), **_GEMMA),
        *_rules(_GOOGLE, "gemma", **_GEMMA),
        _dotted("gemini-3.7-flash", _GEMINI, **_GEMINI_THINKING, **_NO_SAMPLING),
        _dotted("gemini-3.6-flash", _GEMINI, **_GEMINI_THINKING, **_NO_SAMPLING),
        _dotted(
            "gemini-3.5-flash", _GEMINI, **_GEMINI_THINKING, max_context_tokens=1048576
        ),
        _rule(
            "gemini-3.1-pro",
            _GEMINI,
            ("gemini-3.1-pro-preview", "gemini-3-1-pro-preview"),
            **_GEMINI_THINKING,
        ),
        _rule(
            "gemini-3.1-flash-lite",
            _GEMINI,
            ("gemini-3.1-flash-lite-preview", "gemini-3-1-flash-lite-preview"),
            max_output_tokens=65536,
        ),
        _dotted(
            "gemini-3.1-flash-lite",
            _GEMINI,
            **_GEMINI_THINKING,
            max_context_tokens=1048576,
        ),
        _rule(
            "gemini-3.1-flash-image",
            _GEMINI,
            ("gemini-3.1-flash-image", "gemini-3-1-flash-image"),
            max_context_tokens=128000,
            max_output_tokens=32768,
        ),
        _rule(
            "gemini-3-pro-image",
            _GEMINI,
            ("gemini-3-pro-image",),
            max_context_tokens=65536,
            max_output_tokens=32768,
        ),
        _rule(
            "gemini-3-flash",
            _GEMINI,
            ("gemini-3-flash", "gemini-3.0-flash"),
            **_GEMINI_THINKING,
            max_context_tokens=1048576,
        ),
        _rule(
            "gemini-3-pro",
            _GEMINI,
            ("gemini-3-pro", "gemini-3.0-pro"),
            **_GEMINI_THINKING,
            max_context_tokens=2000000,
        ),
        _rule(
            "gemini-3-preview",
            _GEMINI,
            ("gemini-3-preview", "gemini-3.0-preview"),
            **_GEMINI_THINKING,
            max_context_tokens=1048576,
        ),
        _rule("gemini-3", _GEMINI, ("gemini-3", "gemini-3.0"), **_GEMINI_THINKING),
        _dotted(
            "gemini-2.5-pro", _GEMINI, **_GEMINI_THINKING, max_context_tokens=2000000
        ),
        _dotted(
            "gemini-2.5-flash-lite",
            _GEMINI,
            max_context_tokens=1048576,
            max_output_tokens=65536,
        ),
        _dotted("gemini-2.5-flash", _GEMINI, max_output_tokens=65536),
        _rule(
            "gemini-2.0-flash",
            _GEMINI,
            ("gemini-2.0-flash", "gemini-2-flash", "gemini-2.0"),
        ),
        _dotted("gemini-1.5-pro", _GEMINI, max_context_tokens=2000000),
        _dotted("gemini-1.5-flash", _GEMINI),
        *_rules(_GEMINI, "gemini"),
    )


_ROUTED_DETAIL: _Fields = {
    "supports_image_detail": True,
    "default_image_detail": "high",
}
_ROUTED_GPT5: _Fields = {**_ROUTED_DETAIL, **_REASONING, **_NO_SAMPLING}
_ROUTED_O_SERIES = _contains("/o1", "/o3", "/o4")
_ROUTED_GEMINI: _Fields = {
    "supports_reasoning_effort": True,
    "supports_media_resolution": True,
    "max_context_tokens": 1000000,
    "max_output_tokens": 8192,
}
_GEMINI_TAIL = _tail("gemini", "google/")
_LLAMA_TAIL = _tail("llama", "meta/")
_PENALTIES: _Fields = {
    "supports_frequency_penalty": True,
    "supports_presence_penalty": True,
}


def _mistral(name: str) -> bool:
    return "mistral" in name.rsplit("/", 1)[-1] or "mixtral" in name


def _openrouter_rules() -> tuple[FamilyRule, ...]:
    return (
        _routed(
            "openrouter-deepseek",
            {**_REASONING, **_NO_SAMPLING},
            _contains("deepseek-r1"),
        ),
        _routed(
            "openrouter-deepseek",
            _REASONING,
            lambda name: "deepseek" in name and "terminus" in name,
        ),
        _routed(
            "openrouter-deepseek",
            {"supports_reasoning_effort": True},
            _contains("deepseek"),
        ),
        _routed(
            "openrouter-gpt-oss",
            {**_ROUTED_DETAIL, **_REASONING, **_PENALTIES},
            _contains("gpt-oss"),
        ),
        _routed("openrouter-gpt5", _ROUTED_DETAIL, _contains("gpt-5.3")),
        _routed("openrouter-gpt5", _ROUTED_GPT5, _contains("gpt-5.5-pro")),
        _routed(
            "openrouter-gpt5",
            {**_ROUTED_GPT5, **_ORIGINAL},
            _contains("gpt-5.4", "gpt-5.5", "gpt-5.6"),
        ),
        _routed("openrouter-gpt5", _ROUTED_GPT5, _contains("gpt-5")),
        _routed(
            "openrouter-o-series",
            {**_ROUTED_GPT5, "supports_vision": False},
            lambda name: _ROUTED_O_SERIES(name) and "mini" in name,
        ),
        _routed("openrouter-o-series", _ROUTED_GPT5, _ROUTED_O_SERIES),
        _routed(
            "openrouter-claude",
            {**_REASONING, "max_context_tokens": 200000},
            _tail("claude", "anthropic/"),
        ),
        _routed(
            "openrouter-gemini",
            {**_ROUTED_GEMINI, "is_reasoning_model": True},
            lambda name: (
                _GEMINI_TAIL(name)
                and _contains("gemini-2.5", "gemini-3", "gemini-2-5")(name)
            ),
        ),
        _routed("openrouter-gemini", _ROUTED_GEMINI, _GEMINI_TAIL),
        _routed(
            "openrouter-llama",
            _PENALTIES,
            lambda name: _LLAMA_TAIL(name) and _contains("vision", "llama-3.2")(name),
        ),
        _routed(
            "openrouter-llama", {**_PENALTIES, "supports_vision": False}, _LLAMA_TAIL
        ),
        _routed(
            "openrouter-mistral",
            {},
            lambda name: _mistral(name) and "pixtral" in name,
        ),
        _routed("openrouter-mistral", {"supports_vision": False}, _mistral),
        _routed(
            "openrouter-qwen",
            {**_REASONING, "max_context_tokens": 131072},
            _contains("qwen"),
        ),
        _routed("openrouter", {}),
    )


BASE_FAMILIES: tuple[FamilyRule, ...] = (
    *_openai_rules(),
    *_anthropic_rules(),
    *_google_rules(),
    *_openrouter_rules(),
)
"""The rules ChronoTranscriber and ChronoMiner shared, without tool extras."""

PROVIDERS: tuple[ProviderRule, ...] = (
    ProviderRule("anthropic", prefixes=("claude",), contains=("anthropic",)),
    ProviderRule("google", prefixes=("gemini", "gemma"), contains=("google",)),
    # whisper-1 is an OpenAI model (developers.openai.com/api/docs/models/whisper-1).
    ProviderRule(
        "openai", prefixes=("gpt", "o1", "o3", "o4", "chatgpt", "text-", "whisper")
    ),
    ProviderRule(
        "openrouter", contains=("llama", "mistral", "mixtral", "qwen", "deepseek")
    ),
)
"""Provider detection for ids without a ``provider:`` prefix or a vendor."""


def rule_index(rules: tuple[FamilyRule, ...], model: str) -> int:
    """Return the position of the rule of *rules* that resolves *model*.

    Raises:
        ValueError: No rule resolves *model*.
    """
    table = CapabilityTable(families=rules, providers=PROVIDERS)
    explicit, name = split_model_id(model, table)
    provider = explicit or detect_provider(model, table)
    for index, rule in enumerate(rules):
        if rule.matches(name, provider):
            return index
    raise ValueError(f"No capability rule resolves {model!r}")


def _matchers(record: AddRule | Rematch) -> dict[str, Any]:
    names = ("prefixes", "contains", "exact", "exclude")
    return {
        name: getattr(record, name)
        for name in names
        if getattr(record, name, None) is not None
    }


def _structure(
    rules: list[FamilyRule], record: AddRule | DropRule | Rematch
) -> list[FamilyRule]:
    if isinstance(record, DropRule):
        del rules[rule_index(tuple(rules), record.id)]
    elif isinstance(record, Rematch):
        index = rule_index(tuple(rules), record.id)
        rules[index] = replace(rules[index], **_matchers(record))
    else:
        if record.like is None:
            new = FamilyRule(record.family, {}, **_matchers(record))
        else:
            like = rules[rule_index(tuple(rules), record.like)]
            new = replace(like, family=record.family, **_matchers(record))
        rules.insert(rule_index(tuple(rules), record.before), new)
    return rules


def _fact(rules: list[FamilyRule], record: Fact) -> list[FamilyRule]:
    for model in record.ids:
        index = rule_index(tuple(rules), model)
        rule = rules[index]
        rules[index] = replace(rule, fields={**rule.fields, record.field: record.value})
    return rules


def apply_facts(
    base: Iterable[FamilyRule],
    structure: Iterable[AddRule | DropRule | Rematch] = STRUCTURE,
    facts: Iterable[Fact] = FACTS,
) -> tuple[FamilyRule, ...]:
    """Return *base* with the structural records and then the facts applied.

    Raises:
        ValueError: A record names an id no rule resolves, or a field the
            capability record lacks.
    """
    rules = list(base)
    for record in structure:
        rules = _structure(rules, record)
    for fact in facts:
        rules = _fact(rules, fact)
    return tuple(rules)


FAMILIES: tuple[FamilyRule, ...] = apply_facts(BASE_FAMILIES)
"""The shared family rules, in match order."""


def _documented_limits(
    rules: tuple[FamilyRule, ...], facts: Iterable[Fact] = FACTS
) -> frozenset[str]:
    return frozenset(
        rules[rule_index(rules, model)].family
        for fact in facts
        if fact.field == "max_output_tokens"
        for model in fact.ids
    )


OUTPUT_LIMIT_FAMILIES: frozenset[str] = _documented_limits(FAMILIES)
"""Families whose output limit a sourced fact sets; for a routed provider
(OpenRouter) only these limits are reliable enough to send and to clamp to."""

AUDIO_FAMILIES: frozenset[str] = frozenset(
    {
        "openai-audio",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.1-pro",
        "gemini-3.1-flash-lite",
        "gemini-3-flash",
        "gemini-3-pro",
        "gemini-3-preview",
        "gemini-3",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini",
    }
)
"""Families whose models take audio input (the Gemini pages list audio input;
the Gemini image models do not take it)."""

SPEECH_TO_TEXT_FAMILIES: frozenset[str] = frozenset({"openai-audio"})
"""Families served only by OpenAI's transcription endpoint."""


@dataclass(frozen=True)
class TableExtension:
    """A tool's additions to the shared table.

    *rules* are matched before the shared rules. *family_extras* maps a
    family name to keys merged into the ``extras`` of every shared rule of
    that family.
    """

    rules: tuple[FamilyRule, ...] = ()
    family_extras: Mapping[str, Mapping[str, Any]] = field(
        default_factory=lambda: MappingProxyType({})
    )


def _with_extras(rule: FamilyRule, extras: Mapping[str, Any] | None) -> FamilyRule:
    if not extras:
        return rule
    merged = {**dict(rule.fields.get("extras") or {}), **extras}
    return replace(rule, fields={**rule.fields, "extras": merged})


def build_table(extension: TableExtension | None = None) -> CapabilityTable:
    """Return the shared table with *extension* applied.

    Raises:
        ValueError: *extension* names a family no shared rule has.
    """
    if extension is None:
        return TABLE
    known = {rule.family for rule in FAMILIES}
    unknown = sorted(set(extension.family_extras) - known)
    if unknown:
        raise ValueError(f"Unknown capability families: {', '.join(unknown)}")
    families = tuple(
        _with_extras(rule, extension.family_extras.get(rule.family))
        for rule in FAMILIES
    )
    return CapabilityTable(families=(*extension.rules, *families), providers=PROVIDERS)


TABLE = CapabilityTable(families=FAMILIES, providers=PROVIDERS)
"""The shared table without a tool's extension."""
