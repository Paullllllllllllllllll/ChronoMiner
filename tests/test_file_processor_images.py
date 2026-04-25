"""Tests for visual input routing in file_processor.py."""

from pathlib import Path

import pytest

from modules.extract.file_processor import FileProcessor


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

    def test_directory_with_images_is_visual(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"x")
        assert FileProcessor._is_visual_input(tmp_path) is True

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
