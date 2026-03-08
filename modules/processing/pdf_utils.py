"""PDF processing utilities for extracting pages as images.

Provides PDF page rendering via PyMuPDF for sending individual pages to
vision-capable LLMs. Ported from ChronoTranscriber with Tesseract paths removed.

Dependencies: PyMuPDF (fitz), Pillow.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional

import fitz
from PIL import Image

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Extracts pages from a PDF as PIL Images for LLM vision processing."""

    def __init__(self, pdf_path: Path) -> None:
        self.pdf_path = pdf_path
        self.doc: Optional[fitz.Document] = None

    def open_pdf(self) -> None:
        """Open the PDF document."""
        try:
            self.doc = fitz.open(self.pdf_path)
        except Exception as e:
            logger.error("Failed to open PDF: %s, %s", self.pdf_path, e)
            raise

    def close_pdf(self) -> None:
        """Close the opened PDF document."""
        if self.doc:
            self.doc.close()
            self.doc = None

    def __enter__(self) -> "PDFProcessor":
        self.open_pdf()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close_pdf()

    def get_page_count(self) -> int:
        """Return the number of pages in the PDF."""
        if self.doc is None:
            self.open_pdf()
        assert self.doc is not None
        return self.doc.page_count

    def render_page_to_pil(
        self, page_index: int, dpi: int = 300, max_pixels: int = 0
    ) -> Image.Image:
        """Render a single PDF page to a PIL Image.

        Args:
            page_index: Zero-based page index.
            dpi: Target rendering resolution.
            max_pixels: If > 0, reduce DPI so the rendered page stays within
                this pixel budget. Set to 0 to disable dynamic scaling.

        Returns:
            PIL Image in RGB mode.
        """
        if self.doc is None:
            self.open_pdf()
        assert self.doc is not None
        page = self.doc[page_index]
        effective_dpi = dpi
        if max_pixels > 0:
            rect = page.rect
            pixels_at_dpi = (rect.width / 72 * dpi) * (rect.height / 72 * dpi)
            if pixels_at_dpi > max_pixels:
                effective_dpi = max(1, int(dpi * math.sqrt(max_pixels / pixels_at_dpi)))
                logger.info(
                    "Page %d: %.0f MP at %d DPI exceeds limit (%.0f MP); reducing to %d DPI",
                    page_index + 1,
                    pixels_at_dpi / 1e6,
                    dpi,
                    max_pixels / 1e6,
                    effective_dpi,
                )
        pix = page.get_pixmap(matrix=fitz.Matrix(effective_dpi / 72, effective_dpi / 72), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return img

    def extract_pages_as_images(
        self,
        dpi: int = 300,
        page_indices: Optional[List[int]] = None,
        max_pixels: int = 0,
    ) -> List[Image.Image]:
        """Render multiple PDF pages to PIL Images.

        Args:
            dpi: Target rendering resolution.
            page_indices: Optional list of zero-based page indices.
                         If None, all pages are rendered.
            max_pixels: If > 0, reduce DPI per page to stay within this pixel
                budget. Set to 0 to disable dynamic scaling.

        Returns:
            List of PIL Images in RGB mode.
        """
        if self.doc is None:
            self.open_pdf()
        assert self.doc is not None

        indices = page_indices if page_indices is not None else list(range(self.doc.page_count))
        images: List[Image.Image] = []
        for idx in indices:
            try:
                images.append(self.render_page_to_pil(idx, dpi, max_pixels=max_pixels))
            except Exception as e:
                logger.error(
                    "Error rendering page %d from %s: %s",
                    idx + 1, self.pdf_path.name, e,
                )
        return images
