"""Streaming page producer for visual extraction.

Renders and preprocesses document pages one at a time, yielding compact
base64 JPEG payloads with provenance fingerprints. Replaces the former
load-all-pages design: peak memory is one full-resolution page plus the
payloads currently in flight, independent of document length.

Used by both the synchronous (queue producer-consumer) and batch
(materialize-then-submit) visual pipelines.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from modules.config.constants import SUPPORTED_PDF_EXTENSIONS
from modules.images.encoding import encode_bytes_to_base64
from modules.images.llm_preprocess import (
    ImageProcessor,
    detect_model_type,
    get_image_config_section_name,
)
from modules.images.pdf_utils import PDFProcessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PagePayload:
    """One preprocessed page, ready for an LLM vision call.

    Provenance fields fingerprint the exact JPEG bytes sent to the model so
    a run can be verified by re-rendering (source + config + versions are
    recorded at file level; see ``build_image_provenance``).
    """

    index: int  # 1-based page number within the source document
    base64: str
    mime_type: str
    detail: str | None
    sha256: str
    width: int
    height: int
    byte_size: int
    effective_dpi: int | None  # None for non-PDF image inputs

    def as_chunk(self) -> dict[str, Any]:
        """Return the dict shape expected by the processing strategies."""
        return {
            "base64": self.base64,
            "mime_type": self.mime_type,
            "detail": self.detail,
            "image_provenance": self.provenance(),
        }

    def provenance(self) -> dict[str, Any]:
        """Return the per-page provenance record."""
        return {
            "image_sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "byte_size": self.byte_size,
            "effective_dpi": self.effective_dpi,
        }


@dataclass(frozen=True)
class PageError:
    """A page that failed to render/preprocess.

    Yielded in place of a :class:`PagePayload` so the consumer can record
    the failure (the output is then marked partial and the page is
    re-queued on resume) instead of the page silently disappearing.
    """

    index: int
    error: str


def _payload_from_pil(
    img: Image.Image,
    index: int,
    processor: ImageProcessor,
    image_detail: str | None,
    effective_dpi: int | None,
) -> PagePayload:
    """Preprocess one PIL image fully in memory and build its payload."""
    jpeg_bytes = processor.process_pil(img)
    # Re-derive final dimensions from the transform chain output rather than
    # the raw render: a header-only probe of the JPEG bytes is cheap.
    import io

    with Image.open(io.BytesIO(jpeg_bytes)) as probe:
        width, height = probe.size
    return PagePayload(
        index=index,
        base64=encode_bytes_to_base64(jpeg_bytes, "image/jpeg"),
        mime_type="image/jpeg",
        detail=image_detail,
        sha256=hashlib.sha256(jpeg_bytes).hexdigest(),
        width=width,
        height=height,
        byte_size=len(jpeg_bytes),
        effective_dpi=effective_dpi,
    )


async def stream_page_payloads(
    file_path: Path,
    page_indices: list[int],
    image_config: dict[str, Any],
    provider: str,
    model_name: str,
    image_detail: str | None,
) -> AsyncIterator[PagePayload | PageError]:
    """Yield preprocessed page payloads one at a time.

    For PDFs, pages are rendered sequentially via PyMuPDF inside
    ``asyncio.to_thread`` (the document handle is only ever touched by one
    rendering call at a time). Each raw page is preprocessed, encoded, and
    released before the next page is rendered.

    Args:
        file_path: Source PDF or single image file.
        page_indices: 1-based page numbers to produce, in order. For
            single-image inputs only ``[1]`` is meaningful.
        image_config: Full image-processing config (top-level dict with
            ``target_dpi``, ``max_pixels_per_page``, and provider sections).
        provider: LLM provider name (drives the preprocessing profile).
        model_name: Model name (drives the preprocessing profile).
        image_detail: Detail level recorded on each payload.

    Yields:
        :class:`PagePayload` per successfully rendered page, or a
        :class:`PageError` for pages that fail to render/preprocess.
    """
    processor = ImageProcessor(
        provider=provider,
        model_name=model_name,
        image_config=image_config,
    )
    ext = file_path.suffix.lower()

    if ext in SUPPORTED_PDF_EXTENSIONS:
        target_dpi = resolve_target_dpi(image_config, provider, model_name)
        max_pixels = int(image_config.get("max_pixels_per_page", 0))
        render_strategy = str(
            image_config.get("render_strategy", "direct") or "direct"
        ).lower()
        # Detail actually driving the resize profile (may differ from the
        # detail recorded on the payload); needed to derive the direct DPI.
        resize_detail = processor._effective_detail()

        def _render_and_process(page_number: int) -> PagePayload:
            render_dpi = target_dpi
            if render_strategy == "direct":
                rect = pdf.doc[page_number - 1].rect  # type: ignore[index]
                render_dpi = ImageProcessor.compute_direct_render_dpi(
                    rect.width,
                    rect.height,
                    target_dpi,
                    resize_detail,
                    processor.img_cfg,
                    processor.model_type,
                )
            img, effective_dpi = pdf.render_page_with_dpi(
                page_number - 1, render_dpi, max_pixels=max_pixels
            )
            try:
                return _payload_from_pil(
                    img, page_number, processor, image_detail, effective_dpi
                )
            finally:
                img.close()

        with PDFProcessor(file_path) as pdf:
            for page_number in page_indices:
                try:
                    yield await asyncio.to_thread(_render_and_process, page_number)
                except Exception as e:
                    logger.error(
                        "Error rendering page %d from %s: %s",
                        page_number,
                        file_path.name,
                        e,
                    )
                    yield PageError(index=page_number, error=str(e))
    else:

        def _load_and_process() -> PagePayload:
            with Image.open(file_path) as raw:
                img = raw.convert("RGB")
            try:
                return _payload_from_pil(img, 1, processor, image_detail, None)
            finally:
                img.close()

        try:
            yield await asyncio.to_thread(_load_and_process)
        except Exception as e:
            logger.error("Error processing image %s: %s", file_path.name, e)
            yield PageError(index=1, error=str(e))


def resolve_image_section(
    image_config: dict[str, Any], provider: str, model_name: str
) -> dict[str, Any]:
    """Return the provider-specific section of the image config."""
    model_type = detect_model_type(provider, model_name)
    section_name = get_image_config_section_name(model_type)
    return image_config.get(section_name, {})


def resolve_target_dpi(
    image_config: dict[str, Any], provider: str, model_name: str
) -> int:
    """Resolve the render DPI for the active provider.

    Prefers a ``target_dpi`` in the provider-specific section, falls back to
    the top-level ``target_dpi``, then to 300. This mirrors ChronoTranscriber's
    ``resolve_image_settings`` and makes per-provider overrides (e.g. the
    custom endpoint's 150 DPI) actually take effect.
    """
    section = resolve_image_section(image_config, provider, model_name)
    if section.get("target_dpi") is not None:
        return int(section["target_dpi"])
    return int(image_config.get("target_dpi", 300))


def build_image_provenance(
    file_path: Path,
    image_config: dict[str, Any],
    provider: str,
    model_name: str,
    image_detail: str | None,
) -> dict[str, Any]:
    """Build the file-level provenance record for a visual extraction.

    Captures everything needed to deterministically re-derive the images
    sent to the model: source fingerprint, library versions, and the
    effective preprocessing parameters.
    """
    import fitz
    import PIL

    sha = hashlib.sha256()
    with file_path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            sha.update(block)

    section = resolve_image_section(image_config, provider, model_name)
    return {
        "source_file": file_path.name,
        "source_sha256": sha.hexdigest(),
        "pymupdf_version": getattr(fitz, "__version__", None)
        or getattr(fitz, "version", ("unknown",))[0],
        "pillow_version": PIL.__version__,
        "image_config": {
            "target_dpi": image_config.get("target_dpi"),
            "max_pixels_per_page": image_config.get("max_pixels_per_page"),
            "resize_profile": section.get("resize_profile"),
            "jpeg_quality": section.get("jpeg_quality"),
            "grayscale_conversion": section.get("grayscale_conversion"),
            "detail": image_detail,
        },
    }
