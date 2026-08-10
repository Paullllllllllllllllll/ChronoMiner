"""Tests for modules/processing/pdf_utils.py."""

from unittest.mock import patch

import pytest

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

        with (
            patch.object(proc, "open_pdf") as mock_open,
            patch.object(proc, "close_pdf") as mock_close,
        ):
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

    def test_render_page_with_dpi(self, minimal_pdf):
        with PDFProcessor(minimal_pdf) as proc:
            img, effective_dpi = proc.render_page_with_dpi(0, dpi=72)
            assert img.mode == "RGB"
            assert img.size[0] > 0
            assert img.size[1] > 0
            assert effective_dpi == 72

    def test_render_page_higher_dpi(self, minimal_pdf):
        with PDFProcessor(minimal_pdf) as proc:
            img_72, _ = proc.render_page_with_dpi(0, dpi=72)
            img_300, _ = proc.render_page_with_dpi(0, dpi=300)
            # Higher DPI should produce a larger image
            assert img_300.size[0] > img_72.size[0]
            assert img_300.size[1] > img_72.size[1]

    def test_render_page_respects_max_pixels(self, minimal_pdf):
        with PDFProcessor(minimal_pdf) as proc:
            img, effective_dpi = proc.render_page_with_dpi(
                0, dpi=300, max_pixels=10_000
            )
            assert effective_dpi < 300
            assert img.size[0] * img.size[1] <= 10_000 * 1.1


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

    def test_render_each_page(self, three_page_pdf):
        with PDFProcessor(three_page_pdf) as proc:
            images = [proc.render_page_with_dpi(i, dpi=72)[0] for i in range(3)]
            assert len(images) == 3
            assert all(img.mode == "RGB" for img in images)
