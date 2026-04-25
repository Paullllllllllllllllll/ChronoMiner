"""Tests for modules/config/constants.py."""

from modules.config.constants import (
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS,
    SUPPORTED_VISUAL_EXTENSIONS,
)


class TestSupportedImageFormats:
    def test_png_format(self):
        assert SUPPORTED_IMAGE_FORMATS[".png"] == "image/png"

    def test_jpg_format(self):
        assert SUPPORTED_IMAGE_FORMATS[".jpg"] == "image/jpeg"

    def test_jpeg_format(self):
        assert SUPPORTED_IMAGE_FORMATS[".jpeg"] == "image/jpeg"

    def test_tiff_format(self):
        assert SUPPORTED_IMAGE_FORMATS[".tiff"] == "image/tiff"

    def test_tif_format(self):
        assert SUPPORTED_IMAGE_FORMATS[".tif"] == "image/tiff"

    def test_bmp_format(self):
        assert SUPPORTED_IMAGE_FORMATS[".bmp"] == "image/bmp"

    def test_gif_format(self):
        assert SUPPORTED_IMAGE_FORMATS[".gif"] == "image/gif"

    def test_webp_format(self):
        assert SUPPORTED_IMAGE_FORMATS[".webp"] == "image/webp"

    def test_unsupported_format_not_in_map(self):
        assert ".doc" not in SUPPORTED_IMAGE_FORMATS


class TestSupportedExtensions:
    def test_image_extensions_match_formats(self):
        assert SUPPORTED_IMAGE_EXTENSIONS == set(SUPPORTED_IMAGE_FORMATS.keys())

    def test_pdf_extensions(self):
        assert ".pdf" in SUPPORTED_PDF_EXTENSIONS

    def test_visual_extensions_union(self):
        assert SUPPORTED_VISUAL_EXTENSIONS == SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS

    def test_png_in_visual(self):
        assert ".png" in SUPPORTED_VISUAL_EXTENSIONS

    def test_pdf_in_visual(self):
        assert ".pdf" in SUPPORTED_VISUAL_EXTENSIONS

    def test_txt_not_in_visual(self):
        assert ".txt" not in SUPPORTED_VISUAL_EXTENSIONS
