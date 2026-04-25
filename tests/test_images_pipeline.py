"""Interface-level tests for the ``modules.images`` package.

Exercises the public surface (:class:`ImageProcessor`, :class:`PDFProcessor`,
:func:`build_image_content_block`, :func:`create_data_url`,
:func:`detect_model_type`, :func:`encode_image_to_base64`,
:func:`get_image_config_section_name`) without reaching into private
helpers. Uses a tiny synthetic 1x1 PNG so the tests don't depend on any
fixture file. The Pillow dependency is already a hard requirement of the
package so it is safe to exercise real I/O.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from PIL import Image

from modules.images import (
    ImageProcessor,
    PDFProcessor,
    build_image_content_block,
    create_data_url,
    detect_model_type,
    encode_image_to_base64,
    get_image_config_section_name,
)


def _write_png(path: Path, size: tuple[int, int] = (1, 1)) -> Path:
    """Create a solid-color PNG at *path* for testing."""
    img = Image.new("RGB", size, (10, 20, 30))
    img.save(path, format="PNG")
    return path


@pytest.mark.unit
class TestDetectModelType:
    def test_explicit_providers(self):
        assert detect_model_type("openai", "gpt-4o") == "openai"
        assert detect_model_type("anthropic", "claude-sonnet-4") == "anthropic"
        assert detect_model_type("google", "gemini-2.5-flash") == "google"
        assert detect_model_type("custom", "local-model") == "custom"

    def test_openrouter_infers_from_model_name(self):
        assert detect_model_type("openrouter", "anthropic/claude-3-5-sonnet") == "anthropic"
        assert detect_model_type("openrouter", "google/gemini-2.5-pro") == "google"
        assert detect_model_type("openrouter", "openai/gpt-5-mini") == "openai"

    def test_unknown_provider_defaults_to_openai(self):
        assert detect_model_type("mystery", "") == "openai"


@pytest.mark.unit
class TestImageConfigSectionName:
    def test_known_sections(self):
        assert get_image_config_section_name("openai") == "api_image_processing"
        assert get_image_config_section_name("google") == "google_image_processing"
        assert get_image_config_section_name("anthropic") == "anthropic_image_processing"
        assert get_image_config_section_name("custom") == "custom_image_processing"


@pytest.mark.unit
class TestEncoding:
    def test_encode_image_to_base64_returns_mime_and_data(self, tmp_path):
        png = _write_png(tmp_path / "sample.png")
        data, mime = encode_image_to_base64(png)
        assert mime == "image/png"
        # The result must be valid base64 that decodes to non-empty bytes.
        decoded = base64.b64decode(data)
        assert len(decoded) > 0

    def test_encode_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "sample.xyz"
        path.write_bytes(b"data")
        with pytest.raises(ValueError, match="Unsupported"):
            encode_image_to_base64(path)

    def test_create_data_url_has_expected_shape(self):
        url = create_data_url("BASE64DATA", "image/png")
        assert url == "data:image/png;base64,BASE64DATA"


@pytest.mark.unit
class TestContentBlock:
    def test_openai_shape(self):
        block = build_image_content_block(
            "AAA", "image/png", "openai", detail="high", supports_image_detail=True
        )
        assert block["type"] == "image_url"
        assert block["image_url"]["url"].startswith("data:image/png;base64,")
        assert block["image_url"]["detail"] == "high"

    def test_openai_omits_detail_when_unsupported(self):
        block = build_image_content_block(
            "AAA", "image/png", "openai", detail="high", supports_image_detail=False
        )
        assert "detail" not in block["image_url"]

    def test_anthropic_shape(self):
        block = build_image_content_block("AAA", "image/jpeg", "anthropic")
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/jpeg"
        assert block["source"]["data"] == "AAA"

    def test_google_shape(self):
        block = build_image_content_block("AAA", "image/png", "google")
        assert block["type"] == "image_url"
        # Google embeds the data URL directly as a string.
        assert block["image_url"].startswith("data:image/png;base64,")


@pytest.mark.unit
class TestImageProcessorInterface:
    def test_rejects_unsupported_extension(self, tmp_path):
        bad = tmp_path / "x.xyz"
        bad.write_bytes(b"")
        with pytest.raises(ValueError, match="Unsupported image format"):
            ImageProcessor(
                bad, provider="openai", model_name="gpt-4o", image_config={}
            )

    def test_process_image_writes_jpeg(self, tmp_path):
        src = _write_png(tmp_path / "src.png", size=(50, 50))
        out_stem = tmp_path / "processed"
        proc = ImageProcessor(
            src,
            provider="openai",
            model_name="gpt-4o",
            image_config={"api_image_processing": {}},
        )
        result = proc.process_image(out_stem)
        assert result.suffix == ".jpg"
        assert result.exists()
        # Loadable as a real JPEG.
        with Image.open(result) as out_img:
            assert out_img.format == "JPEG"


@pytest.mark.unit
class TestPDFProcessorInterface:
    def test_context_manager_open_close(self, tmp_path):
        # PyMuPDF ships a fitz that can create an empty document in-memory,
        # but creating one on disk requires inserting a page. Use a minimal
        # single-page PDF created via fitz itself.
        import fitz

        pdf_path = tmp_path / "tiny.pdf"
        doc = fitz.open()
        doc.new_page(width=100, height=100)
        doc.save(pdf_path)
        doc.close()

        with PDFProcessor(pdf_path) as pdf:
            assert pdf.get_page_count() == 1
            images = pdf.extract_pages_as_images(dpi=72)
        assert len(images) == 1
        assert images[0].mode == "RGB"
