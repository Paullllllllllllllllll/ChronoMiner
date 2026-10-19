"""Model capability resolution from family rules.

A ``CapabilityTable`` holds provider detection rules, family rules matched in
order (first match wins), and overrides for the conservative profile used for
unknown models. The engine holds no model data; the tools share the table of
:mod:`.capability_table`.

Model ids are matched lowercased after removing a ``provider:`` prefix and
Google's ``models/`` prefix, unless the table keeps prefixes. Ids of the form
``vendor/model`` belong to OpenRouter.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any

logger = logging.getLogger(__name__)

OPENROUTER = "openrouter"
GOOGLE = "google"
CUSTOM = "custom"
UNKNOWN = "unknown"

_BUILTIN_PROVIDERS = frozenset({OPENROUTER, GOOGLE, CUSTOM})
_GOOGLE_PREFIX = "models/"

_warned_models: set[tuple[str, str]] = set()


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What one model accepts; the defaults form the conservative profile.

    ``reasoning_efforts`` lists the reasoning efforts (OpenAI) or thinking
    levels (Gemini) the vendor documents for the model, lowest first; an
    empty tuple sends a requested effort unchanged. ``supports_flex_tier``
    marks the models OpenAI lists for Flex processing.
    ``prompt_cache_parameter`` names the OpenAI parameter that sets the
    prompt-cache lifetime (``prompt_cache_options`` or
    ``prompt_cache_retention``), None where none is documented. ``extras``
    holds tool-specific flags that have no field here.
    """

    provider_name: str
    model_name: str
    family: str = UNKNOWN

    supports_vision: bool = False
    supports_image_detail: bool = False
    supports_original_image_detail: bool = False
    default_image_detail: str = "auto"
    supports_media_resolution: bool = False
    default_media_resolution: str = "high"
    image_original_patch_cap_30k: bool = False
    image_high_res_tier: bool = False

    supports_structured_output: bool = False
    supports_json_mode: bool = False

    is_reasoning_model: bool = False
    supports_reasoning_effort: bool = False
    reasoning_efforts: tuple[str, ...] = ()
    uses_adaptive_thinking: bool = False
    supports_text_verbosity: bool = False

    supports_temperature: bool = True
    supports_top_p: bool = True
    supports_frequency_penalty: bool = False
    supports_presence_penalty: bool = False

    supports_flex_tier: bool = False
    prompt_cache_parameter: str | None = None

    max_context_tokens: int = 128000
    max_output_tokens: int = 4096

    extras: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def extra(self, name: str, default: Any = None) -> Any:
        """Return the tool-specific flag *name*, or *default*."""
        return self.extras.get(name, default)


_RULE_FIELDS = frozenset(f.name for f in fields(Capabilities)) - {
    "model_name",
    "family",
}


class CapabilityError(ValueError):
    """Raised when a selected model cannot do what the run needs."""


def ensure_image_support(capabilities: Capabilities) -> None:
    """Raise ``CapabilityError`` unless the model accepts image input."""
    if not capabilities.supports_vision:
        raise CapabilityError(
            f"Model '{capabilities.model_name}' does not support image inputs. "
            "Choose an image-capable model or use a text-only flow."
        )


@dataclass(frozen=True)
class ProviderRule:
    """Assign *provider* to ids that start with a prefix or contain a part."""

    provider: str
    prefixes: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()

    def matches(self, name: str) -> bool:
        """Whether the normalized id *name* belongs to this provider."""
        return name.startswith(self.prefixes) or any(p in name for p in self.contains)


@dataclass(frozen=True)
class FamilyRule:
    """Capability fields for the models one rule matches.

    A rule matches an id equal to an entry of ``exact``, starting with one of
    ``prefixes`` or containing one of ``contains``, unless it contains one of
    ``exclude``. ``predicate``, when set, must also hold. A rule without any
    matcher matches every id. ``provider`` limits the rule to one provider;
    ``warn`` logs once per model that it fell back to this profile.
    """

    family: str
    fields: Mapping[str, Any]
    prefixes: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()
    exact: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    provider: str | None = None
    predicate: Callable[[str], bool] | None = None
    warn: bool = False

    def __post_init__(self) -> None:
        unknown = sorted(set(self.fields) - _RULE_FIELDS)
        if unknown:
            raise ValueError(
                f"Unknown capability fields in family {self.family!r}: "
                f"{', '.join(unknown)}"
            )

    def matches(self, name: str, provider: str) -> bool:
        """Whether this rule applies to the normalized id *name*."""
        if self.provider is not None and self.provider != provider:
            return False
        if any(part in name for part in self.exclude):
            return False
        if self.exact or self.prefixes or self.contains:
            hit = (
                name in self.exact
                or name.startswith(self.prefixes)
                or any(part in name for part in self.contains)
            )
            if not hit:
                return False
        return self.predicate is None or self.predicate(name)

    def build(self, model: str, provider: str) -> Capabilities:
        """Return the capability record of *model* under this rule."""
        return _make(model, provider, self.family, self.fields)


@dataclass(frozen=True)
class CapabilityTable:
    """One tool's capability data.

    ``unknown`` overrides fields of the conservative profile;
    ``default_provider`` is returned for ids no provider rule matches.
    With ``strip_prefixes`` False, ids keep their ``provider:`` and
    ``models/`` prefixes and match as given.
    """

    families: tuple[FamilyRule, ...]
    providers: tuple[ProviderRule, ...] = ()
    unknown: Mapping[str, Any] = field(default_factory=dict)
    default_provider: str = UNKNOWN
    strip_prefixes: bool = True

    def __post_init__(self) -> None:
        unknown = sorted(set(self.unknown) - _RULE_FIELDS)
        if unknown:
            raise ValueError(f"Unknown capability fields: {', '.join(unknown)}")

    @property
    def provider_names(self) -> frozenset[str]:
        """Every provider name the table and the engine know."""
        names = {rule.provider for rule in self.providers}
        names.update(rule.provider for rule in self.families if rule.provider)
        return frozenset(names) | _BUILTIN_PROVIDERS


def _make(
    model: str, provider: str, family: str, values: Mapping[str, Any]
) -> Capabilities:
    merged: dict[str, Any] = {"provider_name": provider, **values}
    merged["extras"] = MappingProxyType(dict(values.get("extras") or {}))
    return Capabilities(model_name=model, family=family, **merged)


def split_model_id(model: str, table: CapabilityTable) -> tuple[str | None, str]:
    """Return the provider named in *model* (or None) and the normalized id."""
    name = model.strip().lower()
    explicit: str | None = None
    if not table.strip_prefixes:
        return explicit, name
    prefix, separator, rest = name.partition(":")
    if separator and prefix in table.provider_names:
        explicit, name = prefix, rest.strip()
    if name.startswith(_GOOGLE_PREFIX):
        explicit = explicit or GOOGLE
        name = name.removeprefix(_GOOGLE_PREFIX)
    return explicit, name


def detect_provider(model: str, table: CapabilityTable) -> str:
    """Return the provider of *model*.

    Order: a ``provider:`` prefix, Google's ``models/`` prefix, a
    ``vendor/model`` id (OpenRouter), the table's provider rules, then
    ``table.default_provider``.
    """
    explicit, name = split_model_id(model, table)
    if explicit is not None:
        return explicit
    if "/" in name:
        return OPENROUTER
    for rule in table.providers:
        if rule.matches(name):
            return rule.provider
    return table.default_provider


def _warn_once(model: str, family: str, consequence: str) -> None:
    key = (model, family)
    if key in _warned_models:
        return
    _warned_models.add(key)
    logger.warning(
        "No exact capability profile for model %r; using %s defaults. %s",
        model,
        family,
        consequence,
    )


def resolve_capabilities(
    model: str, table: CapabilityTable, provider: str | None = None
) -> Capabilities:
    """Return the capabilities of *model* from *table*.

    *provider* overrides detection (for example ``custom`` for a named
    endpoint). Without a matching rule the conservative profile applies:
    no image input and no structured output.
    """
    explicit, name = split_model_id(model, table)
    if provider:
        resolved = provider.strip().lower()
    else:
        resolved = explicit or detect_provider(model, table)
    for rule in table.families:
        if rule.matches(name, resolved):
            if rule.warn:
                _warn_once(model, rule.family, "Capabilities are guessed.")
            return rule.build(model, resolved)
    _warn_once(
        model,
        "conservative unknown-model",
        "Image input and structured output are assumed unsupported.",
    )
    return _make(model, resolved, UNKNOWN, table.unknown)


__all__ = [
    "CUSTOM",
    "GOOGLE",
    "OPENROUTER",
    "UNKNOWN",
    "Capabilities",
    "CapabilityError",
    "CapabilityTable",
    "FamilyRule",
    "ProviderRule",
    "detect_provider",
    "ensure_image_support",
    "resolve_capabilities",
    "split_model_id",
]
