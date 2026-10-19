"""The shared capability table holds the settled vendor facts.

Each row of :data:`FACTS` names a model id and the fields the capability-facts
report settles for it (the labs' documentation, read on 13.10.2026). A row's
value must come from the base table or from a record of
``common/capability_facts.py``, which names the source. Fields a row does not
name are not checked here.
"""

from __future__ import annotations

from typing import Any

import pytest

from chronominer.common import capability_facts
from chronominer.common.capabilities import (
    CapabilityTable,
    FamilyRule,
    detect_provider,
    resolve_capabilities,
)
from chronominer.common.capability_facts import AddRule, DropRule, Rematch
from chronominer.common.capability_table import (
    AUDIO_FAMILIES,
    BASE_FAMILIES,
    FAMILIES,
    ORIGINAL_MAX_SIDE,
    PROVIDERS,
    SPEECH_TO_TEXT_FAMILIES,
    TABLE,
    TableExtension,
    apply_facts,
    build_table,
    rule_index,
)

BASE = CapabilityTable(families=BASE_FAMILIES, providers=PROVIDERS)

_NO_REASONING = {
    "is_reasoning_model": False,
    "supports_reasoning_effort": False,
    "supports_text_verbosity": False,
}
_NO_SAMPLING = {"supports_temperature": False, "supports_top_p": False}
_NO_PENALTIES = {
    "supports_frequency_penalty": False,
    "supports_presence_penalty": False,
}
_PENALTIES = {"supports_frequency_penalty": True, "supports_presence_penalty": True}
_NO_STRUCTURE = {"supports_structured_output": False, "supports_json_mode": False}
_LARGE = {"max_context_tokens": 1050000, "max_output_tokens": 128000}
_PATCH_CAP = {
    "supports_original_image_detail": True,
    "image_original_patch_cap_30k": True,
}
_CHAT_LATEST = {**_NO_REASONING, **_NO_SAMPLING, "max_context_tokens": 128000}
_CODEX = {
    "supports_vision": True,
    "supports_image_detail": True,
    "supports_original_image_detail": False,
    "is_reasoning_model": True,
    "supports_reasoning_effort": True,
    "reasoning_efforts": ("low", "medium", "high", "xhigh"),
    "supports_text_verbosity": False,
    **_NO_SAMPLING,
}
_NONE_TO_MAX = ("none", "low", "medium", "high", "xhigh", "max")
_LOW_TO_MAX = ("low", "medium", "high", "xhigh", "max")
_NONE_TO_XHIGH = ("none", "low", "medium", "high", "xhigh")
_MINIMAL_TO_HIGH = ("minimal", "low", "medium", "high")
_PRO = ("medium", "high", "xhigh")
_ADAPTIVE = {"uses_adaptive_thinking": True, **_NO_SAMPLING}
_TRANSCRIBE = {
    "provider_name": "openai",
    "family": "openai-audio",
    "supports_vision": False,
    "supports_structured_output": False,
    "max_context_tokens": 16000,
    "max_output_tokens": 2000,
}
_GEMMA = {
    "provider_name": "google",
    "supports_vision": True,
    "supports_media_resolution": False,
    **_NO_STRUCTURE,
    "is_reasoning_model": True,
    "supports_reasoning_effort": True,
    "reasoning_efforts": ("minimal", "high"),
    "max_context_tokens": 262144,
    "max_output_tokens": 8192,
}
_IMAGE_MODEL = {
    "is_reasoning_model": True,
    "supports_reasoning_effort": False,
    **_NO_STRUCTURE,
    "max_output_tokens": 32768,
}
_GLM = {"is_reasoning_model": True, "supports_vision": True}
_UNKNOWN = {"family": "unknown", "supports_vision": False}

FACTS: tuple[tuple[str, dict[str, Any]], ...] = (
    # OpenAI model pages: context and output limits.
    *((m, {"max_context_tokens": 400000}) for m in ("gpt-5", "gpt-5.1")),
    *((m, {"max_context_tokens": 400000}) for m in ("gpt-5-mini", "gpt-5-nano")),
    ("gpt-4.1-nano", {"max_output_tokens": 32768, "max_context_tokens": 1047576}),
    ("gpt-4.1", {"max_context_tokens": 1047576}),
    ("gpt-4.1-mini", {"max_context_tokens": 1047576}),
    *((m, _LARGE) for m in ("gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol", "gpt-6-astra")),
    *((m, _LARGE) for m in ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol")),
    ("gpt-5.6", {"family": "gpt-5.6", **_LARGE}),
    # chat-latest: non-reasoning ChatGPT snapshots, 128,000 context.
    ("gpt-5-chat-latest", {**_CHAT_LATEST, "max_output_tokens": 16384}),
    ("gpt-5.1-chat-latest", _CHAT_LATEST),
    ("gpt-5.2-chat-latest", _CHAT_LATEST),
    ("gpt-5.3-chat-latest", _CHAT_LATEST),
    # Ids OpenAI does not document.
    ("gpt-5.3", _UNKNOWN),
    ("gpt-5.3-chat", _UNKNOWN),
    # Codex: image input, low to xhigh effort, no samplers.
    ("gpt-5.2-codex", _CODEX),
    ("gpt-5.3-codex", _CODEX),
    # Pro models: no verbosity, medium to xhigh effort.
    ("gpt-5.2-pro", {**_NO_STRUCTURE, "reasoning_efforts": _PRO}),
    ("gpt-5.4-pro", {"supports_structured_output": False, "reasoning_efforts": _PRO}),
    ("gpt-5.5-pro", {"supports_text_verbosity": False, "reasoning_efforts": _PRO}),
    # Verbosity on the GPT-5 and GPT-6 reasoning ids.
    *(
        (m, {"supports_text_verbosity": True, **_NO_SAMPLING})
        for m in (
            "gpt-5",
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.5",
            "gpt-5.6-luna",
            "gpt-6-luna",
            "gpt-6-sol",
            "gpt-6-astra",
            "gpt-6.1-sol",
        )
    ),
    # Image detail.
    ("gpt-5.4-mini", {"supports_original_image_detail": True}),
    ("gpt-5.4-nano", {"supports_original_image_detail": True}),
    *((m, _PATCH_CAP) for m in ("gpt-5.6-sol", "gpt-6-astra", "gpt-6.1-sol")),
    ("gpt-6-luna", {**_PATCH_CAP, "extras": {ORIGINAL_MAX_SIDE: 3000}}),
    ("gpt-5.6-luna", {"default_image_detail": "high"}),
    # Reasoning efforts per model page and the reasoning guide.
    ("gpt-6-astra", {"reasoning_efforts": _LOW_TO_MAX}),
    ("gpt-6.1-sol", {"reasoning_efforts": _LOW_TO_MAX}),
    *(
        (m, {"reasoning_efforts": _NONE_TO_MAX})
        for m in ("gpt-6-luna", "gpt-6-sol", "gpt-5.6-terra")
    ),
    *(
        (m, {"reasoning_efforts": _NONE_TO_XHIGH})
        for m in ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.2")
    ),
    *(
        (m, {"reasoning_efforts": _MINIMAL_TO_HIGH})
        for m in ("gpt-5", "gpt-5-mini", "gpt-5-nano")
    ),
    ("gpt-5.1", {"reasoning_efforts": ("none", "low", "medium", "high")}),
    # o-series.
    ("o1", {"supports_structured_output": True, "supports_reasoning_effort": False}),
    ("o1-mini", {"supports_reasoning_effort": False}),
    ("o1-pro", {"supports_structured_output": True}),
    ("o3", {"supports_structured_output": True}),
    ("o3-2025-04-16", {"family": "o3", "supports_structured_output": True}),
    ("o3-deep-research", {"supports_structured_output": False}),
    ("o4-mini-deep-research", {"supports_structured_output": False}),
    # GPT-4 family on the Responses API: no penalties.
    *(
        (m, _NO_PENALTIES)
        for m in ("gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.5", "gpt-4-turbo")
    ),
    (
        "gpt-4.5-preview",
        {
            "supports_vision": True,
            "supports_structured_output": True,
            "max_output_tokens": 16384,
        },
    ),
    (
        "gpt-4-turbo",
        {"supports_vision": True, **_NO_STRUCTURE, "max_output_tokens": 4096},
    ),
    (
        "gpt-4",
        {
            "supports_vision": False,
            "supports_structured_output": False,
            "max_context_tokens": 8192,
            "max_output_tokens": 8192,
        },
    ),
    # Speech-to-text models.
    ("gpt-4o-transcribe", _TRANSCRIBE),
    ("gpt-4o-mini-transcribe", _TRANSCRIBE),
    ("gpt-transcribe", _TRANSCRIBE),
    ("whisper-1", _TRANSCRIBE),
    # Flex processing: the models with a Flex price.
    *(
        (m, {"supports_flex_tier": True})
        for m in (
            "gpt-6-astra",
            "gpt-6.1-sol",
            "gpt-6-luna",
            "gpt-6-sol",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
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
        )
    ),
    *(
        (m, {"supports_flex_tier": False})
        for m in (
            "gpt-5.2-pro",
            "gpt-5.3-codex",
            "gpt-5.3-chat-latest",
            "gpt-5-chat-latest",
            "gpt-4o",
            "gpt-4.1",
            "o1",
            "o3-pro",
            "o3-mini",
        )
    ),
    # Prompt caching: prompt_cache_options from GPT-5.6, retention on the list.
    *(
        (m, {"prompt_cache_parameter": "prompt_cache_options"})
        for m in ("gpt-6-luna", "gpt-6-astra", "gpt-6.1-sol", "gpt-5.6-terra")
    ),
    *(
        (m, {"prompt_cache_parameter": "prompt_cache_retention"})
        for m in (
            "gpt-5.5",
            "gpt-5.5-pro",
            "gpt-5.4",
            "gpt-5.2",
            "gpt-5.1",
            "gpt-5.1-chat-latest",
            "gpt-5",
            "gpt-4.1",
        )
    ),
    *(
        (m, {"prompt_cache_parameter": None})
        for m in ("gpt-5.4-mini", "gpt-5.4-pro", "gpt-5-mini", "gpt-4o", "o3")
    ),
    # OpenRouter's model list.
    ("deepseek/deepseek-r1", {"supports_temperature": True, "supports_top_p": True}),
    ("deepseek/deepseek-chat-v3.1", {"supports_reasoning_effort": True}),
    ("deepseek/deepseek-v3.2", {"supports_reasoning_effort": True}),
    ("openai/o4-mini", {"supports_vision": True}),
    ("openai/o3-mini", {"supports_vision": False}),
    ("openai/gpt-5.4", {"supports_original_image_detail": False}),
    ("openai/gpt-5.6-luna", {"supports_original_image_detail": False}),
    ("openai/gpt-4o", {**_PENALTIES, "supports_image_detail": False}),
    ("openai/gpt-4.1", _NO_PENALTIES),
    (
        "openai/gpt-6-luna",
        {
            "is_reasoning_model": True,
            "supports_temperature": False,
            **_NO_PENALTIES,
            "supports_image_detail": False,
        },
    ),
    ("openai/gpt-5.3-chat", {"family": "openrouter"}),
    ("anthropic/claude-opus-5.5", _NO_SAMPLING),
    ("anthropic/claude-haiku-5.5", _NO_SAMPLING),
    (
        "qwen/qwen3-vl-235b-a22b-instruct",
        {"is_reasoning_model": False, "max_context_tokens": 262144},
    ),
    *(
        (
            m,
            {
                "family": "openrouter-qwen3.5-flash",
                "supports_vision": True,
                "is_reasoning_model": True,
                "supports_structured_output": True,
                "max_context_tokens": 1000000,
                "max_output_tokens": 65536,
            },
        )
        for m in ("qwen/qwen3.5-flash-02-23", "qwen/qwen3.5-flash-20260224")
    ),
    ("qwen/qwen-2.5-72b-instruct", {"family": "openrouter-qwen"}),
    (
        "z-ai/glm-4.6v",
        {
            **_GLM,
            **_PENALTIES,
            "supports_structured_output": False,
            "max_context_tokens": 131072,
            "max_output_tokens": 32768,
        },
    ),
    (
        "z-ai/glm-4.5v",
        {
            **_GLM,
            **_PENALTIES,
            "supports_structured_output": False,
            "max_context_tokens": 65536,
            "max_output_tokens": 16384,
        },
    ),
    (
        "z-ai/glm-5v-turbo",
        {
            **_GLM,
            "supports_structured_output": False,
            "max_context_tokens": 202752,
            "max_output_tokens": 131072,
        },
    ),
    (
        "z-ai/glm-5.3-flash",
        {
            **_GLM,
            **_PENALTIES,
            "supports_reasoning_effort": True,
            "max_context_tokens": 1048576,
            "max_output_tokens": 131072,
        },
    ),
    # Anthropic.
    *(
        (m, {"max_context_tokens": 200000})
        for m in (
            "claude-sonnet-4-5",
            "claude-sonnet-4.5",
            "claude-sonnet-4-5-20250929",
        )
    ),
    ("claude-opus-4-6", _ADAPTIVE),
    ("claude-sonnet-4-6", _ADAPTIVE),
    *(
        (m, {**_ADAPTIVE, "image_high_res_tier": True})
        for m in (
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-fable-5",
            "claude-fable-5-1",
            "claude-sonnet-5",
        )
    ),
    *(
        (
            m,
            {
                **_ADAPTIVE,
                "image_high_res_tier": True,
                "max_context_tokens": 1000000,
                "max_output_tokens": 128000,
            },
        )
        for m in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
    ),
    *(
        (m, _NO_STRUCTURE)
        for m in (
            "claude-3-5-haiku-20241022",
            "claude-3-haiku",
            "claude-3-opus",
            "claude-3-sonnet",
        )
    ),
    *(
        (m, {"supports_structured_output": False})
        for m in (
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-opus-4-1-20250805",
        )
    ),
    ("claude-opus-4-1", {"max_output_tokens": 32000}),
    # Google.
    (
        "gemini-2.5-flash",
        {"is_reasoning_model": True, "supports_reasoning_effort": True},
    ),
    ("gemini-2-5-flash", {"family": "gemini-2.5-flash"}),
    *(
        (
            m,
            {
                "family": "gemini-3.1-pro",
                "max_context_tokens": 1048576,
                "reasoning_efforts": ("low", "medium", "high"),
            },
        )
        for m in ("gemini-3.1-pro-preview", "gemini-3.1-pro", "gemini-3-1-pro")
    ),
    (
        "gemini-3.1-flash-lite-preview",
        {"is_reasoning_model": True, "max_context_tokens": 1048576},
    ),
    (
        "gemini-3.1-flash-lite",
        {
            "family": "gemini-3.1-flash-lite",
            "reasoning_efforts": _MINIMAL_TO_HIGH,
            "supports_structured_output": True,
            "max_context_tokens": 1048576,
            "max_output_tokens": 65536,
        },
    ),
    *(
        (m, {**_IMAGE_MODEL, "max_context_tokens": 65536})
        for m in ("gemini-3-pro-image", "gemini-3-pro-image-preview")
    ),
    ("gemini-3.1-flash-image", {**_IMAGE_MODEL, "max_context_tokens": 131072}),
    ("gemini-3-preview", {"max_context_tokens": 1000000}),
    ("gemini-2.5-pro", {"max_context_tokens": 1048576}),
    ("gemini-2.5-flash-lite", {"is_reasoning_model": True}),
    *(
        (m, {"reasoning_efforts": ("low", "medium", "high"), **_NO_SAMPLING})
        for m in ("gemini-3.7-flash", "gemini-3.8-flash")
    ),
    *((m, _NO_SAMPLING) for m in ("gemini-3.5-flash", "gemini-3.6-flash")),
    ("gemma-4-31b-it", _GEMMA),
    ("gemma-4-26b-a4b-it", _GEMMA),
    ("gemma-3-27b-it", {"provider_name": "google", **_UNKNOWN}),
    ("models/gemini-2.5-flash", {"provider_name": "google"}),
    # Z.ai GLM ids called directly.
    (
        "glm-5.3-flash",
        {
            **_GLM,
            "supports_reasoning_effort": True,
            "supports_temperature": True,
            "supports_json_mode": True,
            "supports_structured_output": False,
            **_NO_PENALTIES,
            "max_context_tokens": 1048576,
            "max_output_tokens": 131072,
        },
    ),
    (
        "glm-5.3-flashx",
        {**_GLM, "supports_reasoning_effort": True, "supports_temperature": True},
    ),
    ("glm-5.3-flash:batch", _UNKNOWN),
    ("glm-flash-latest", _UNKNOWN),
    (
        "glm-5v-turbo",
        {**_GLM, "max_context_tokens": 202752, "max_output_tokens": 131072},
    ),
    (
        "glm-4.6v",
        {"max_context_tokens": 131072, "max_output_tokens": 32768, **_NO_PENALTIES},
    ),
    ("glm-4.5v", {"max_context_tokens": 65536, "max_output_tokens": 16384}),
)


@pytest.mark.parametrize(
    ("model", "expected"), FACTS, ids=[f"{m}-{i}" for i, (m, _) in enumerate(FACTS)]
)
def test_the_table_holds_the_settled_facts(
    model: str, expected: dict[str, Any]
) -> None:
    caps = resolve_capabilities(model, TABLE)
    actual = {name: getattr(caps, name) for name in expected}
    if "extras" in expected:
        actual["extras"] = dict(caps.extras)

    assert actual == expected


def _family(model: str) -> str:
    return FAMILIES[rule_index(FAMILIES, model)].family


_RESHAPED = {
    record.family
    for record in capability_facts.STRUCTURE
    if isinstance(record, AddRule)
}
_RESHAPED |= {
    _family(record.id)
    for record in capability_facts.STRUCTURE
    if isinstance(record, Rematch)
}
_FACT_TARGETS = {
    (_family(model), record.field)
    for record in capability_facts.FACTS
    for model in record.ids
}
_DROPPED = {
    record.id for record in capability_facts.STRUCTURE if isinstance(record, DropRule)
}


@pytest.mark.parametrize(
    ("model", "expected"), FACTS, ids=[f"{m}-{i}" for i, (m, _) in enumerate(FACTS)]
)
def test_every_fact_comes_from_the_base_or_a_record(
    model: str, expected: dict[str, Any]
) -> None:
    """A settled value the base lacks is set by a record of capability_facts."""
    base = resolve_capabilities(model, BASE)
    family = resolve_capabilities(model, TABLE).family
    for name, value in expected.items():
        if name in ("family", "extras") or getattr(base, name) == value:
            continue
        traced = (
            (family, name) in _FACT_TARGETS
            or family in _RESHAPED
            or model in _DROPPED
            or family == "unknown"
        )
        assert traced, f"{model}: {name} has no source record"


def test_every_record_names_its_source() -> None:
    records = (*capability_facts.STRUCTURE, *capability_facts.FACTS)
    assert all(record.source.strip() for record in records)


@pytest.mark.parametrize("index", range(len(capability_facts.FACTS)))
def test_every_fact_record_changes_the_base(index: int) -> None:
    """A fact that the base already holds would hide which records matter."""
    record = capability_facts.FACTS[index]
    before = apply_facts(BASE_FAMILIES, facts=capability_facts.FACTS[:index])
    after = apply_facts(BASE_FAMILIES, facts=capability_facts.FACTS[: index + 1])

    assert before != after, record


@pytest.mark.parametrize(
    ("prefixed", "plain", "provider"),
    [
        ("openai:gpt-5.6-luna", "gpt-5.6-luna", "openai"),
        ("models/gemini-2.5-flash", "gemini-2.5-flash", "google"),
        ("google:models/gemini-3.6-flash", "gemini-3.6-flash", "google"),
        ("anthropic:claude-opus-5-5", "claude-opus-5-5", "anthropic"),
    ],
)
def test_prefixed_ids_resolve_like_plain_ids(
    prefixed: str, plain: str, provider: str
) -> None:
    caps = resolve_capabilities(prefixed, TABLE)

    assert detect_provider(prefixed, TABLE) == provider
    assert caps.family == resolve_capabilities(plain, TABLE).family
    assert caps.model_name == prefixed


def test_every_family_level_is_a_known_effort() -> None:
    known = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
    for rule in FAMILIES:
        assert set(rule.fields.get("reasoning_efforts", ())) <= known, rule.family


def test_audio_families_exist() -> None:
    families = {rule.family for rule in FAMILIES}

    assert families >= AUDIO_FAMILIES >= SPEECH_TO_TEXT_FAMILIES
    assert "gemini-3-pro-image" not in AUDIO_FAMILIES
    assert "gemini-3.1-flash-image" not in AUDIO_FAMILIES


def test_the_table_carries_no_tool_extras() -> None:
    assert resolve_capabilities("whisper-1", TABLE).extras == {}
    assert resolve_capabilities("gemini-3.6-flash", TABLE).extras == {}


def test_an_extension_adds_family_extras_and_rules_first() -> None:
    table = build_table(
        TableExtension(
            rules=(FamilyRule("local", {"supports_vision": True}, ("gpt-6-luna-x",)),),
            family_extras={
                "gpt-6-luna": {"flag": True},
                "openai-audio": {"audio": True},
            },
        )
    )

    luna = resolve_capabilities("gpt-6-luna", table)
    assert dict(luna.extras) == {ORIGINAL_MAX_SIDE: 3000, "flag": True}
    assert resolve_capabilities("whisper-1", table).extra("audio") is True
    assert resolve_capabilities("gpt-6-luna-x", table).family == "local"
    assert resolve_capabilities("gpt-6-luna", TABLE).extra("flag") is None


def test_an_extension_naming_an_unknown_family_fails() -> None:
    with pytest.raises(ValueError, match="no-such-family"):
        build_table(TableExtension(family_extras={"no-such-family": {"x": 1}}))


def test_no_extension_returns_the_shared_table() -> None:
    assert build_table() is TABLE
