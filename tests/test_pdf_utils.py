"""Tests for modules/processing/pdf_utils.py."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from modules.images.pdf_utils import PDFProcessor


class TestPDFProcessorInit:
    def test_init_stores_path(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy")
        proc = PDFProcessor(pdf_path)
        assert proc.pdf_path == pdf_path
        assert proc.doc is None


class TestPDFProcessorContextManager:
    def test_context_manager_opens_and_closes(self, tmp_path):
        """Test that context manager calls open_pdf and close_pdf."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy")
        proc = PDFProcessor(pdf_path)

        with patch.object(proc, "open_pdf") as mock_open, \
             patch.object(proc, "close_pdf") as mock_close:
            with proc:
                mock_open.assert_called_once()
            mock_close.assert_called_once()


class TestPDFProcessorWithFixture:
    """Tests that require a real (minimal) PDF file."""

    @pytest.fixture
    def minimal_pdf(self, tmp_path):
        """Create a minimal valid PDF with one page using PyMuPDF."""
        import fitz
        pdf_path = tmp_path / "minimal.pdf"
        doc = fitz.open()
        page = doc.new_page(width=200, height=300)
        # Insert some text so the page isn't completely blank
        page.insert_text((50, 100), "Hello World", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path

    def test_get_page_count(self, minimal_pdf):
        with PDFProcessor(minimal_pdf) as proc:
            assert proc.get_page_count() == 1

    def test_render_page_to_pil(self, minimal_pdf):
        with PDFProcessor(minimal_pdf) as proc:
            img = proc.render_page_to_pil(0, dpi=72)
            assert img.mode == "RGB"
            assert img.size[0] > 0
            assert img.size[1] > 0

    def test_render_page_higher_dpi(self, minimal_pdf):
        with PDFProcessor(minimal_pdf) as proc:
            img_72 = proc.render_page_to_pil(0, dpi=72)
            img_300 = proc.render_page_to_pil(0, dpi=300)
            # Higher DPI should produce a larger image
            assert img_300.size[0] > img_72.size[0]
            assert img_300.size[1] > img_72.size[1]

    def test_extract_pages_as_images_all(self, minimal_pdf):
        with PDFProcessor(minimal_pdf) as proc:
            images = proc.extract_pages_as_images(dpi=72)
            assert len(images) == 1
            assert images[0].mode == "RGB"

    def test_extract_pages_with_indices(self, minimal_pdf):
        with PDFProcessor(minimal_pdf) as proc:
            images = proc.extract_pages_as_images(dpi=72, page_indices=[0])
            assert len(images) == 1

    def test_extract_pages_empty_indices(self, minimal_pdf):
        with PDFProcessor(minimal_pdf) as proc:
            images = proc.extract_pages_as_images(dpi=72, page_indices=[])
            assert len(images) == 0


class TestPDFProcessorMultiPage:
    @pytest.fixture
    def three_page_pdf(self, tmp_path):
        import fitz
        pdf_path = tmp_path / "three_pages.pdf"
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page(width=200, height=300)
            page.insert_text((50, 100), f"Page {i + 1}", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path

    def test_page_count_three(self, three_page_pdf):
        with PDFProcessor(three_page_pdf) as proc:
            assert proc.get_page_count() == 3

    def test_extract_all_pages(self, three_page_pdf):
        with PDFProcessor(three_page_pdf) as proc:
            images = proc.extract_pages_as_images(dpi=72)
            assert len(images) == 3

    def test_extract_subset(self, three_page_pdf):
        with PDFProcessor(three_page_pdf) as proc:
            images = proc.extract_pages_as_images(dpi=72, page_indices=[0, 2])
            assert len(images) == 2
