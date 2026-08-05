"""Tests for visual input routing in file_processor.py."""

import logging

import pytest

import modules.extract.file_processor as fp_module
from modules.config.capabilities import detect_capabilities
from modules.extract.file_processor import (
    FileProcessor,
    warn_if_original_detail_downgraded,
)


class TestIsVisualInput:
    def test_png_is_visual(self, tmp_path):
        f = tmp_path / "page.png"
        f.write_bytes(b"dummy")
        assert FileProcessor._is_visual_input(f) is True

    def test_jpg_is_visual(self, tmp_path):
        f = tmp_path / "page.jpg"
        f.write_bytes(b"dummy")
        assert FileProcessor._is_visual_input(f) is True

    def test_pdf_is_visual(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"dummy")
        assert FileProcessor._is_visual_input(f) is True

    def test_txt_is_not_visual(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        assert FileProcessor._is_visual_input(f) is False

    def test_directory_with_images_is_not_visual(self, tmp_path):
        # Directories are rejected: the visual pipeline processes single
        # files only; stream_page_payloads would crash on a directory.
        (tmp_path / "a.png").write_bytes(b"x")
        assert FileProcessor._is_visual_input(tmp_path) is False

    def test_directory_with_only_text_is_not_visual(self, tmp_path):
        (tmp_path / "a.txt").write_text("text")
        assert FileProcessor._is_visual_input(tmp_path) is False

    def test_empty_directory_is_not_visual(self, tmp_path):
        assert FileProcessor._is_visual_input(tmp_path) is False

    def test_tiff_is_visual(self, tmp_path):
        f = tmp_path / "page.tiff"
        f.write_bytes(b"dummy")
        assert FileProcessor._is_visual_input(f) is True

    def test_bmp_is_visual(self, tmp_path):
        f = tmp_path / "page.bmp"
        f.write_bytes(b"dummy")
        assert FileProcessor._is_visual_input(f) is True

    def test_gif_is_visual(self, tmp_path):
        f = tmp_path / "page.gif"
        f.write_bytes(b"dummy")
        assert FileProcessor._is_visual_input(f) is True

    def test_webp_is_visual(self, tmp_path):
        f = tmp_path / "page.webp"
        f.write_bytes(b"dummy")
        assert FileProcessor._is_visual_input(f) is True

    def test_nonexistent_path(self, tmp_path):
        fake = tmp_path / "nonexistent"
        assert FileProcessor._is_visual_input(fake) is False


@pytest.fixture()
def _clear_original_detail_warned():
    """Reset the per-process warn-once ledger around each test."""
    fp_module._ORIGINAL_DETAIL_WARNED.clear()
    yield
    fp_module._ORIGINAL_DETAIL_WARNED.clear()


@pytest.mark.usefixtures("_clear_original_detail_warned")
class TestOriginalDetailDowngradeWarning:
    """``llm_detail: original`` is honored only on the Responses route."""

    def test_warns_for_chat_completions_model(self, caplog):
        caps = detect_capabilities("gpt-5.4-mini", provider="openai")
        with caplog.at_level(logging.WARNING, logger=fp_module.logger.name):
            emitted = warn_if_original_detail_downgraded(
                "gpt-5.4-mini", "original", caps
            )

        assert emitted is True
        messages = [r.getMessage() for r in caplog.records]
        assert any("gpt-5.4-mini" in m and "'high'" in m for m in messages)
        assert any("gpt-5.6-sol" in m for m in messages)

    def test_warns_only_once_per_process(self, caplog):
        caps = detect_capabilities("gpt-5.4-mini", provider="openai")
        with caplog.at_level(logging.WARNING, logger=fp_module.logger.name):
            first = warn_if_original_detail_downgraded("gpt-5.4-mini", "original", caps)
            second = warn_if_original_detail_downgraded(
                "gpt-5.4-mini", "original", caps
            )

        assert (first, second) == (True, False)
        assert len([r for r in caplog.records if "original" in r.getMessage()]) == 1

    def test_no_warning_for_responses_only_model(self, caplog):
        caps = detect_capabilities("gpt-5.6-sol", provider="openai")
        with caplog.at_level(logging.WARNING, logger=fp_module.logger.name):
            emitted = warn_if_original_detail_downgraded(
                "gpt-5.6-sol", "original", caps
            )

        assert emitted is False
        assert caplog.records == []

    def test_no_warning_for_non_original_detail(self, caplog):
        caps = detect_capabilities("gpt-5.6-luna", provider="openai")
        with caplog.at_level(logging.WARNING, logger=fp_module.logger.name):
            emitted = warn_if_original_detail_downgraded("gpt-5.6-luna", "high", caps)

        assert emitted is False
        assert caplog.records == []
