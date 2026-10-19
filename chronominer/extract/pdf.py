"""PDF files: page counts and the page images of LLM requests.

:class:`PdfPages` renders the pages of a PDF for an LLM request through
``common.images`` and names them ``page_NNNN``. A numeric render takes an
integer DPI, as ChronoMiner has always rendered: the direct strategy rounds
the profile's scale to a whole DPI, and the memory guard lowers it to the
largest whole DPI within the pixel budget. :class:`PDFProcessor` counts the
pages.
"""

from __future__ import annotations

import logging
import math
import types
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import fitz

from chronominer.common.images import (
    POINTS_PER_INCH,
    PagePayload,
    PdfPageSource,
    content_scale_factor,
)
from chronominer.extract.images import (
    PAYLOAD_SHA256_KEY,
    PageImage,
    page_image,
    with_resampling,
)

__all__ = [
    "PDF_PAGE_NAME",
    "PDFProcessor",
    "PdfPages",
    "render_pdf_page",
]

logger = logging.getLogger(__name__)

PDF_PAGE_NAME = "page_{number:04d}"
"""The name of a PDF page; ``number`` is 1-based."""


class PdfPages(PdfPageSource):
    """The pages of one PDF, named ``page_NNNN``.

    Rendered pages are downscaled with the filter of
    :func:`~chronominer.extract.images.with_resampling`; numeric renders use
    a whole DPI, which the page's provenance records.
    """

    name_template: ClassVar[str] = PDF_PAGE_NAME

    def __init__(
        self,
        pdf_path: Path,
        img_cfg: dict[str, Any],
        model_type: str,
        *,
        section: str = "",
    ) -> None:
        super().__init__(
            pdf_path,
            with_resampling(img_cfg),
            model_type,
            section=section,
            payload_sha256_key=PAYLOAD_SHA256_KEY,
        )
        self._render_dpi: dict[int, int] = {}

    def _render_zoom(self, page: Any, img_cfg: dict[str, Any]) -> float:
        """Return the zoom of a whole render DPI for a numeric target.

        ``supersample`` renders at ``target_dpi``; ``direct`` at the target
        DPI times the profile's scale, rounded and never above the target.
        The memory guard then lowers the DPI to fit ``max_pixels_per_page``.
        """
        if self.target_dpi == "native":
            raise ValueError("Native zoom is resolved per page inside the render lock")
        target = int(self.target_dpi)
        rect = page.rect
        pt_width, pt_height = float(rect.width), float(rect.height)
        dpi = target
        if self.render_strategy == "direct" and pt_width > 0 and pt_height > 0:
            zoom = target / POINTS_PER_INCH
            scale = content_scale_factor(
                (pt_width * zoom, pt_height * zoom), img_cfg, self.model_type
            )
            dpi = max(1, round(target * min(scale, 1.0)))
        if self.max_pixels > 0:
            pixels = (pt_width / 72 * dpi) * (pt_height / 72 * dpi)
            if pixels > self.max_pixels:
                dpi = max(1, int(dpi * math.sqrt(self.max_pixels / pixels)))
        self._render_dpi[int(page.number)] = dpi
        return dpi / POINTS_PER_INCH

    def build_payload(self, index: int) -> PagePayload:
        """Render, preprocess and encode page *index* (0-based)."""
        payload = super().build_payload(index)
        dpi = self._render_dpi.pop(index, None)
        if dpi is None:
            return payload
        provenance = {**payload.provenance, "render_dpi": dpi, "effective_dpi": dpi}
        return replace(payload, provenance=provenance)


def render_pdf_page(
    pdf_path: Path, page_index: int, img_cfg: dict[str, Any], model_type: str
) -> PageImage:
    """Render and encode the page at *page_index* (0-based) of a PDF.

    Raises:
        IndexError: The PDF has no such page.
    """
    with PdfPages(pdf_path, img_cfg, model_type) as pages:
        if not 0 <= page_index < len(pages):
            raise IndexError(
                f"Page index {page_index} out of range for {pdf_path.name} "
                f"({len(pages)} pages)"
            )
        return page_image(pages.build_payload(page_index), pages)


class PDFProcessor:
    """The page count of a PDF; the document opens only while in use."""

    def __init__(self, pdf_path: Path) -> None:
        self.pdf_path = pdf_path
        self.doc: fitz.Document | None = None

    def open_pdf(self) -> None:
        """Open the PDF document."""
        try:
            self.doc = fitz.open(self.pdf_path)
        except Exception as exc:
            logger.error("Failed to open PDF: %s, %s", self.pdf_path, exc)
            raise

    def close_pdf(self) -> None:
        """Close the opened PDF document."""
        if self.doc:
            self.doc.close()
            self.doc = None

    def __enter__(self) -> PDFProcessor:
        self.open_pdf()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.close_pdf()

    def page_count(self) -> int:
        """Return the number of pages; the document is closed afterwards."""
        with fitz.open(self.pdf_path) as doc:
            return int(doc.page_count)
