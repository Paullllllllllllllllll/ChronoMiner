"""Page images: image settings and the payloads of LLM requests.

:func:`resolve_image_settings` builds the image settings of a model from the
recommended values of its provider family (:data:`IMAGE_PROCESSING`, built
from ``spec.IMAGE_PRESETS``), the run's ``ImageSpec`` and the model's
capabilities, with the detail rules of the presets, as ChronoMiner has always
sized its pages: per-provider sections with resize profiles,
``high_target_box``, ``low_max_side_px``, ``max_image_bytes`` and
``media_resolution``, the model caps of ``common.native`` and Lanczos
downscaling. The page sources of ``common.images`` render and encode the
pages; :class:`FolderPages` loads one image file as one page, and
:func:`stream_pages` turns their payloads into :class:`PageImage` chunks, a
page that failed to render included.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import fitz
import PIL

from chronominer.common import image_presets
from chronominer.common.images import (
    FolderPageSource,
    PageError,
    PagePayload,
    PageSource,
    resolve_detail,
    sha256_of_file,
    stream_payloads,
)
from chronominer.common.native import (
    image_settings_fingerprint,
    model_image_cap,
    validate_image_settings,
)
from chronominer.llm.capabilities import ORIGINAL_MAX_SIDE, detect_capabilities
from chronominer.llm.media import SUPPORTED_IMAGE_FORMATS
from chronominer.spec import IMAGE_PRESETS, ImageSpec

__all__ = [
    "IMAGE_PROCESSING",
    "MAX_PIXELS_PER_PAGE",
    "PAYLOAD_SHA256_KEY",
    "RENDER_STRATEGY",
    "RESAMPLING_ALGORITHM",
    "FolderPages",
    "PageImage",
    "detect_model_type",
    "encode_image_to_base64",
    "file_provenance",
    "image_section",
    "load_image",
    "page_image",
    "resolve_image_settings",
    "resume_images",
    "stream_pages",
    "with_resampling",
]

logger = logging.getLogger(__name__)

PAYLOAD_SHA256_KEY = "image_sha256"
"""The provenance key of the SHA-256 of an encoded page."""

RESAMPLING_ALGORITHM = IMAGE_PRESETS.resampling
"""The downscaling filter of page images.

The page sources add it to their own copy of the settings, so it stays out of
the image-settings record and its fingerprint.
"""

RENDER_STRATEGY = IMAGE_PRESETS.render_strategy
"""Numeric PDF renders go straight to the profile size, never above the DPI."""

MAX_PIXELS_PER_PAGE = IMAGE_PRESETS.max_pixels_per_page
"""The memory guard of a numeric PDF render, in pixels."""

_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "target_dpi": 300,
        "native_fallback_dpi": 300,
        "payload_format": "jpeg",
        "max_image_bytes": 0,
        "jpeg_quality": 95,
        "grayscale_conversion": True,
        "handle_transparency": True,
        "resize_profile": "auto",
        "low_max_side_px": 512,
        "high_target_box": [768, 1536],
    }
)
"""Values a section that leaves a key out takes."""

IMAGE_PROCESSING: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    IMAGE_PRESETS.sections()
)
"""Recommended image settings, one section per model family."""


def detect_model_type(provider: str, model_name: str | None = None) -> str:
    """Return the model family whose image settings apply.

    Direct providers decide; OpenRouter and other providers are detected from
    the model name, defaulting to OpenAI.
    """
    return image_presets.detect_model_type(provider, model_name)


def image_section(model_type: str) -> str:
    """Return the name of the settings section of *model_type*."""
    return image_presets.section_name(model_type)


def with_resampling(img_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of *img_cfg* that downscales with the page filter."""
    return image_presets.with_resampling(img_cfg, RESAMPLING_ALGORITHM)


def resume_images(images: ImageSpec, stored: Mapping[str, Any] | None) -> ImageSpec:
    """Return *images* with the unset options taken from a stored record.

    *stored* is the image-settings record of a document's stored pages.
    Its DPI, format, JPEG quality and grayscale fill the options the run
    leaves unset, so a changed preset does not change the pages of a
    document that resumes; the detail stays that of the model's family.
    Without a record *images* is returned unchanged.
    """
    if not stored:
        return images
    kept = ImageSpec(
        dpi=stored.get("target_dpi"),
        format=stored.get("payload_format"),
        jpeg_quality=stored.get("jpeg_quality"),
        grayscale=stored.get("grayscale_conversion"),
    )
    return images.over(kept)


def resolve_image_settings(
    provider: str | None,
    model_name: str | None,
    images: ImageSpec | None = None,
    original_max_side_px: int | None = None,
) -> tuple[dict[str, Any], str, str]:
    """Return the image settings, model family and section of a model.

    The settings are the section's recommended values with the run's values
    applied, the detail the model accepts (``resolved_detail`` and
    ``detail``), the model's image caps and the rendering options; the model
    name is lowercased. OpenRouter takes the OpenAI section's detail. OpenAI
    pages at ``original`` detail are limited on the long side to
    *original_max_side_px*, else to the model's default limit, if any.

    Raises:
        ValueError: A setting is invalid, or a native render has no bound.
    """
    provider = (provider or "openai").lower()
    model = (model_name or "").lower()
    model_type = detect_model_type(provider, model)
    section = image_section(model_type)
    rules = IMAGE_PRESETS.detail
    cfg = image_presets.section_settings(
        IMAGE_PROCESSING, section, images, rules, provider
    )
    validate_image_settings(cfg, section)
    result: dict[str, Any] = {**_DEFAULTS, **cfg}
    caps = detect_capabilities(model, provider)
    detail = image_presets.page_detail(provider, model_type, result, caps, rules)
    image_caps = (
        detect_capabilities(model.split("/")[-1])
        if provider == "openrouter" and model_type == "anthropic"
        else caps
    )
    result.update(
        model_name=model,
        provider=provider,
        resolved_detail=detail,
        detail=detail,
        image_original_patch_cap_30k=(
            caps.image_original_patch_cap_30k and provider == "openai"
        ),
        image_high_res_tier=image_caps.image_high_res_tier,
        max_pixels_per_page=MAX_PIXELS_PER_PAGE,
        render_strategy=RENDER_STRATEGY,
    )
    if provider == "openai" and detail == "original":
        side = original_max_side_px
        if side is None:
            side = caps.extra(ORIGINAL_MAX_SIDE)
        if side is not None:
            result["original_max_side_px"] = int(side)
    cap = model_image_cap(model_type, model, detail, result)
    result["cap_policy"] = cap.policy if cap else "profile-v1"
    if (
        result["target_dpi"] == "native"
        and not cap
        and result["resize_profile"] == "none"
    ):
        raise ValueError(f"Invalid resize_profile in {section}: native needs a bound")
    return result, model_type, section


@dataclass(frozen=True)
class PageImage:
    """One page as the extraction pipeline records it: one chunk.

    A page that could not be rendered or encoded carries ``render_error`` and
    no image; it is recorded as a failed chunk. *custom_id* names the chunk
    in ``_output.json``; :meth:`provenance` is the chunk unit's record.
    """

    index: int
    image_name: str
    source_file: str = ""
    page_index: int | None = None
    base64: str = ""
    mime_type: str = "image/jpeg"
    render_error: str | None = None
    image_provenance: Mapping[str, Any] = field(default_factory=dict)
    custom_id: str = ""

    @property
    def chunk_range(self) -> None:
        """Pages have no line range."""
        return None

    @property
    def text(self) -> None:
        """Pages carry no chunk text."""
        return None

    def provenance(self) -> dict[str, Any]:
        """Return the chunk unit's provenance: custom id and image record."""
        record: dict[str, Any] = {}
        if self.custom_id:
            record["custom_id"] = self.custom_id
        record["chunk_range"] = None
        if self.image_provenance:
            record["image_provenance"] = dict(self.image_provenance)
        return record


def _page_provenance(payload: PagePayload) -> dict[str, Any]:
    """Return the page record of *payload*, with ``effective_dpi`` always set."""
    record = dict(payload.provenance)
    record.setdefault("effective_dpi", None)
    return record


def page_image(result: PagePayload | PageError, source: PageSource) -> PageImage:
    """Return the page chunk of a payload or render error of *source*."""
    if isinstance(result, PagePayload):
        return PageImage(
            index=result.index,
            image_name=result.image_name,
            source_file=result.source_file,
            page_index=result.page_index,
            base64=result.base64,
            mime_type=result.mime_type,
            image_provenance=_page_provenance(result),
        )
    if isinstance(source, FolderPageSource):
        source_file, page_index = str(source.image_paths[result.index]), None
    else:
        source_file, page_index = str(source.source_path), result.index
    return PageImage(
        index=result.index,
        image_name=result.image_name,
        source_file=source_file,
        page_index=page_index,
        render_error=result.error,
        image_provenance={"render_error": result.error},
    )


async def stream_pages(
    source: PageSource, indices: Iterable[int] | None = None
) -> AsyncIterator[PageImage]:
    """Yield the pages of *indices* (all by default) in order, one at a time.

    The source is closed after the last page.

    Raises:
        RuntimeError: Too many pages failed to render (``check_failure_rate``).
    """
    with source:
        async for result in stream_payloads(source, indices):
            yield page_image(result, source)


class FolderPages(FolderPageSource):
    """Image files as pages, named by their file name.

    An image input is one file and one page; the run passes it as the only
    entry of *image_paths*. Images are downscaled with
    :data:`RESAMPLING_ALGORITHM`.
    """

    def __init__(
        self,
        folder_path: Path,
        img_cfg: dict[str, Any],
        model_type: str,
        *,
        image_paths: list[Path] | None = None,
        section: str = "",
    ) -> None:
        super().__init__(
            folder_path,
            with_resampling(img_cfg),
            model_type,
            image_paths=image_paths,
            section=section,
            payload_sha256_key=PAYLOAD_SHA256_KEY,
        )


def load_image(
    image_path: Path, index: int, img_cfg: dict[str, Any], model_type: str
) -> PageImage:
    """Load and encode one image file as the page at *index*."""
    pages = FolderPages(
        image_path.parent, img_cfg, model_type, image_paths=[image_path]
    )
    return replace(page_image(pages.build_payload(0), pages), index=index)


def file_provenance(
    path: Path, img_cfg: Mapping[str, Any], model_type: str
) -> dict[str, Any]:
    """Return the document-level image record of the input *path*.

    It holds what re-derives the pages sent: the source file and its hash,
    the library versions, the image settings with their fingerprint, the
    model and the detail.
    """
    settings = dict(img_cfg)
    return {
        "source_file": path.name,
        "source_sha256": sha256_of_file(path),
        "pymupdf_version": getattr(fitz, "__version__", None)
        or getattr(fitz, "version", ("unknown",))[0],
        "pillow_version": PIL.__version__,
        "image_config": settings,
        "model_type": model_type,
        "model_name": settings.get("model_name", ""),
        "detail": resolve_detail(settings, model_type),
        "cap_policy": settings.get("cap_policy", "profile-v1"),
        "image_settings_fingerprint": image_settings_fingerprint(
            {**settings, "model_type": model_type}
        ),
    }


def encode_image_to_base64(image_path: Path) -> tuple[str, str]:
    """Return the base64 text and MIME type of an image file, unchanged.

    Raises:
        ValueError: The file type is not a supported image format.
    """
    ext = image_path.suffix.lower()
    mime_type = SUPPORTED_IMAGE_FORMATS.get(ext)
    if not mime_type:
        raise ValueError(f"Unsupported image format: {ext}")
    return base64.b64encode(image_path.read_bytes()).decode("utf-8"), mime_type
