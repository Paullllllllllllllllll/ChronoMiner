"""Image presets: the recommended page-image values and request settings per family.

One :class:`FamilyPreset` per model family (openai, anthropic, google,
custom) holds the values the wizard recommends (:class:`Recommended`) and the
request section built from them: target DPI, payload format, JPEG quality,
grayscale, the detail, the resize profile and the size limits. The builders
:func:`openai_preset`, :func:`anthropic_preset`, :func:`google_preset` and
:func:`custom_preset` carry the values the tools share as defaults; a tool
passes its own where it differs. :class:`ImagePresets` adds the tool's
:class:`DetailRules` and resampling filter.

:func:`section_settings` applies a run's image options to a section, and
:func:`page_detail` resolves the detail a model's pages are sized and sent
with, as the tool's :class:`DetailRules` say.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

from .capabilities import Capabilities

__all__ = [
    "DETAIL_KEYS",
    "FAMILIES",
    "MAX_IMAGE_BYTES",
    "SECTIONS",
    "DetailCheck",
    "DetailRules",
    "FamilyPreset",
    "ImagePresets",
    "ImageValues",
    "ModelType",
    "Recommended",
    "anthropic_preset",
    "custom_preset",
    "detect_model_type",
    "google_preset",
    "openai_preset",
    "page_detail",
    "section_name",
    "section_settings",
    "with_resampling",
]

ModelType = Literal["openai", "google", "anthropic", "custom"]

FAMILIES: tuple[ModelType, ...] = ("openai", "anthropic", "google", "custom")
SECTIONS: Mapping[str, str] = MappingProxyType(
    {
        "openai": "api_image_processing",
        "anthropic": "anthropic_image_processing",
        "google": "google_image_processing",
        "custom": "custom_image_processing",
    }
)
"""The request section of each model family."""

DETAIL_KEYS: Mapping[str, tuple[str, tuple[str, ...]]] = MappingProxyType(
    {
        "api_image_processing": ("llm_detail", ("low", "high", "auto", "original")),
        "custom_image_processing": ("llm_detail", ("low", "high", "auto", "original")),
        "anthropic_image_processing": ("resize_profile", ("low", "high", "auto")),
        "google_image_processing": ("media_resolution", ("low", "high", "auto")),
    }
)
"""The key that carries the detail of each section and the values it takes."""

MAX_IMAGE_BYTES: Mapping[str, int] = MappingProxyType(
    {
        "openai": 50_000_000,
        "anthropic": 10_000_000,
        "google": 20_000_000,
        "custom": 0,
    }
)
"""The payload size limit of each family's pages; 0 sets none."""

_OPENAI_NAMES = ("gpt-", "o1", "o3", "o4", "chatgpt")
"""Name prefixes of OpenAI models given without a vendor."""

_NATIVE_FALLBACK_DPI = 300
_HIGH_TARGET_BOX = (768, 1536)


class ImageValues(Protocol):
    """A run's image options; None leaves the preset's value."""

    @property
    def dpi(self) -> Any: ...

    @property
    def detail(self) -> Any: ...

    @property
    def format(self) -> Any: ...

    @property
    def jpeg_quality(self) -> Any: ...

    @property
    def grayscale(self) -> Any: ...


@dataclass(frozen=True)
class Recommended:
    """The image values the wizard recommends for one model family."""

    dpi: int | str
    detail: str
    format: str
    jpeg_quality: int
    grayscale: bool


@dataclass(frozen=True)
class FamilyPreset:
    """One family's recommended values and the request section built from them."""

    recommended: Recommended
    section: Mapping[str, Any]


def _common(recommended: Recommended, family: str) -> dict[str, Any]:
    return {
        "target_dpi": recommended.dpi,
        "native_fallback_dpi": _NATIVE_FALLBACK_DPI,
        "payload_format": recommended.format,
        "max_image_bytes": MAX_IMAGE_BYTES[family],
        "grayscale_conversion": recommended.grayscale,
        "handle_transparency": True,
        "jpeg_quality": recommended.jpeg_quality,
    }


def _limits(low_max_side_px: int, high_target_box: tuple[int, int]) -> dict[str, Any]:
    return {
        "low_max_side_px": low_max_side_px,
        "high_target_box": list(high_target_box),
    }


def openai_preset(
    *,
    dpi: int | str = "native",
    detail: str = "original",
    image_format: str = "jpeg",
    jpeg_quality: int = 95,
    grayscale: bool = False,
) -> FamilyPreset:
    """Return the OpenAI preset; the detail is the section's ``llm_detail``."""
    recommended = Recommended(dpi, detail, image_format, jpeg_quality, grayscale)
    section = {
        **_common(recommended, "openai"),
        "llm_detail": detail,
        "resize_profile": "high",
        **_limits(512, _HIGH_TARGET_BOX),
    }
    return FamilyPreset(recommended, MappingProxyType(section))


def anthropic_preset(
    *,
    dpi: int | str = "native",
    detail: str = "auto",
    image_format: str = "jpeg",
    jpeg_quality: int = 95,
    grayscale: bool = False,
) -> FamilyPreset:
    """Return the Anthropic preset; the detail is the section's resize profile."""
    recommended = Recommended(dpi, detail, image_format, jpeg_quality, grayscale)
    section = {
        **_common(recommended, "anthropic"),
        "resize_profile": detail,
        **_limits(512, _HIGH_TARGET_BOX),
    }
    return FamilyPreset(recommended, MappingProxyType(section))


def google_preset(
    *,
    dpi: int | str = 300,
    detail: str = "high",
    image_format: str = "jpeg",
    jpeg_quality: int = 95,
    grayscale: bool = False,
) -> FamilyPreset:
    """Return the Google preset; the detail is the section's media resolution."""
    recommended = Recommended(dpi, detail, image_format, jpeg_quality, grayscale)
    section = {
        **_common(recommended, "google"),
        "media_resolution": detail,
        "resize_profile": "high",
        **_limits(512, _HIGH_TARGET_BOX),
    }
    return FamilyPreset(recommended, MappingProxyType(section))


def custom_preset(
    *,
    dpi: int | str = 150,
    detail: str = "high",
    image_format: str = "jpeg",
    jpeg_quality: int = 95,
    grayscale: bool = False,
) -> FamilyPreset:
    """Return the preset of custom endpoints and unknown families."""
    recommended = Recommended(dpi, detail, image_format, jpeg_quality, grayscale)
    section = {
        **_common(recommended, "custom"),
        "llm_detail": detail,
        "resize_profile": "low",
        **_limits(768, (512, 1024)),
    }
    return FamilyPreset(recommended, MappingProxyType(section))


@dataclass(frozen=True)
class DetailCheck:
    """The details a provider's requests keep, and what replaces the others.

    The check applies to the model types in *model_types* (None: every
    type) and, with *needs_detail_support*, only to models that take an
    image detail. It checks the section's ``llm_detail``, or with *source*
    ``request`` the detail of the request option that :func:`page_detail`
    receives. A model without image detail gets *without_detail* when it is
    set. Otherwise a detail in *kept* stays; ``original`` stays where
    *original* is set and the model takes it, else it becomes
    *unsupported_original* when that is set. Every other detail becomes
    *fallback*, or the model's default detail when *fallback* is None.
    """

    kept: frozenset[str]
    original: bool = False
    unsupported_original: str | None = None
    fallback: str | None = "high"
    without_detail: str | None = None
    needs_detail_support: bool = False
    model_types: frozenset[str] | None = None
    source: Literal["section", "request"] = "section"

    def applies(self, model_type: str, caps: Capabilities) -> bool:
        """Whether the check applies to *model_type* and the model's *caps*."""
        if self.model_types is not None and model_type not in self.model_types:
            return False
        return caps.supports_image_detail or not self.needs_detail_support

    def resolve(self, detail: str, caps: Capabilities) -> str:
        """Return the detail a request keeps for the requested *detail*."""
        if self.without_detail is not None and not caps.supports_image_detail:
            return self.without_detail
        if detail in self.kept:
            return detail
        if detail == "original":
            if self.original and caps.supports_original_image_detail:
                return detail
            if self.unsupported_original is not None:
                return self.unsupported_original
        return self.fallback if self.fallback is not None else caps.default_image_detail


@dataclass(frozen=True)
class DetailRules:
    """How a tool turns the requested image detail into the detail of its pages.

    With *filter_overrides* a run detail the section's detail key does not
    take is dropped; without it the value is written as given. With
    *openrouter_openai_detail* the settings of an OpenRouter model whose
    section has another detail key (Anthropic, Google) carry the OpenAI
    section's ``llm_detail``. *checks* holds the check of each
    provider and *other* the check of every provider not listed; a provider
    without a check keeps its section's detail.
    """

    filter_overrides: bool = True
    openrouter_openai_detail: bool = True
    checks: Mapping[str, DetailCheck] = field(
        default_factory=lambda: MappingProxyType({})
    )
    other: DetailCheck | None = None


@dataclass(frozen=True)
class ImagePresets:
    """A tool's image presets: one :class:`FamilyPreset` per family.

    *families* must hold every family of :data:`FAMILIES`; unknown families
    take the custom preset. *resampling* is the downscaling filter of page
    images, *render_strategy* and *max_pixels_per_page* the rendering
    options of a numeric PDF render.
    """

    families: Mapping[str, FamilyPreset]
    detail: DetailRules
    resampling: str = "lanczos"
    render_strategy: str = "direct"
    max_pixels_per_page: int = 24_000_000

    def recommended(self, family: str) -> Recommended:
        """Return the recommended values of *family* (custom when unknown)."""
        preset = self.families.get(family) or self.families["custom"]
        return preset.recommended

    def sections(self) -> dict[str, dict[str, Any]]:
        """Return a fresh copy of the request section of every family."""
        return {
            SECTIONS[family]: copy.deepcopy(dict(preset.section))
            for family, preset in self.families.items()
        }


def detect_model_type(provider: str, model_name: str | None = None) -> ModelType:
    """Return the model family whose image settings apply.

    Direct providers decide; OpenRouter and other providers are detected from
    the model name: Gemini, Claude and OpenAI models take their vendor's
    family, every other OpenRouter model (Qwen, Llama, Mistral) the custom
    preset and a model of an unknown provider OpenAI's.
    """
    provider = provider.lower()
    name = model_name.lower() if model_name else ""
    for family in FAMILIES:
        if provider == family:
            return family
    if "gemini" in name or "google/" in name:
        return "google"
    if "claude" in name or "anthropic/" in name:
        return "anthropic"
    if provider != "openrouter":
        return "openai"
    if name.startswith("openai/") or name.startswith(_OPENAI_NAMES):
        return "openai"
    return "custom"


def section_name(model_type: str) -> str:
    """Return the request section of *model_type* (OpenAI's when unknown)."""
    return SECTIONS.get(model_type, SECTIONS["openai"])


def _with_overrides(
    values: Mapping[str, Any],
    section: str,
    images: ImageValues | None,
    filtered: bool,
) -> dict[str, Any]:
    cfg: dict[str, Any] = copy.deepcopy(dict(values))
    if images is None:
        return cfg
    if images.dpi is not None:
        cfg["target_dpi"] = images.dpi
    if images.format is not None:
        cfg["payload_format"] = images.format
    if images.jpeg_quality is not None:
        cfg["jpeg_quality"] = images.jpeg_quality
    if images.grayscale is not None:
        cfg["grayscale_conversion"] = images.grayscale
    key, accepted = DETAIL_KEYS[section]
    if images.detail is not None and (not filtered or images.detail in accepted):
        cfg[key] = images.detail
    return cfg


def section_settings(
    table: Mapping[str, Mapping[str, Any]],
    section: str,
    images: ImageValues | None,
    rules: DetailRules,
    provider: str,
) -> dict[str, Any]:
    """Return *section* of *table* with the run's image options applied.

    *table* maps section names to their values (a tool's request table); a
    section it lacks starts empty. An OpenRouter model takes the OpenAI
    section's detail where *rules* say so, unless the detail key of its own
    section is ``llm_detail`` (the custom section of Qwen, Llama or Mistral
    models).
    """
    filtered = rules.filter_overrides
    cfg = _with_overrides(table.get(section, {}), section, images, filtered)
    own_detail = DETAIL_KEYS.get(section, ("", ()))[0] == "llm_detail"
    if rules.openrouter_openai_detail and provider == "openrouter" and not own_detail:
        openai = SECTIONS["openai"]
        api = _with_overrides(table.get(openai, {}), openai, images, filtered)
        cfg["llm_detail"] = api["llm_detail"]
    return cfg


def _section_detail(settings: Mapping[str, Any], model_type: str) -> str:
    if model_type == "google":
        return str(settings.get("media_resolution", "high") or "high")
    if model_type == "anthropic":
        return str(settings.get("resize_profile", "auto") or "auto")
    return str(settings.get("llm_detail", "high") or "high")


def page_detail(
    provider: str,
    model_type: str,
    settings: Mapping[str, Any],
    caps: Capabilities,
    rules: DetailRules,
    *,
    request: str | None = None,
) -> str:
    """Return the detail a model's pages are sized and sent with.

    Where a check of *rules* applies, it checks the ``llm_detail`` of
    *settings*, or *request* (the request option's detail; None or empty
    for none) when the check's source is the request; otherwise the detail
    of the section's own key stands.
    """
    check = rules.checks.get(provider, rules.other)
    if check is not None and check.applies(model_type, caps):
        if check.source == "request":
            requested = str(request or "")
        else:
            requested = str(settings.get("llm_detail", "high"))
        return check.resolve(requested.lower().strip(), caps)
    return _section_detail(settings, model_type).lower().strip()


def with_resampling(img_cfg: Mapping[str, Any], algorithm: str) -> dict[str, Any]:
    """Return a copy of *img_cfg* that downscales with *algorithm*."""
    return {**img_cfg, "resampling_algorithm": algorithm}
