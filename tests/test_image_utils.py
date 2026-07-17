"""Tests for modules/processing/image_utils.py."""

import base64

import pytest
from PIL import Image

from modules.images import (
    ImageProcessor,
    create_data_url,
    detect_model_type,
    encode_image_to_base64,
    get_image_config_section_name,
)


class TestDetectModelType:
    def test_openai_provider(self):
        assert detect_model_type("openai", "gpt-5-mini") == "openai"

    def test_anthropic_provider(self):
        assert detect_model_type("anthropic", "claude-sonnet-4-5") == "anthropic"

    def test_google_provider(self):
        assert detect_model_type("google", "gemini-2.5-flash") == "google"

    def test_openrouter_with_google_model(self):
        assert detect_model_type("openrouter", "google/gemini-2.5-flash") == "google"

    def test_openrouter_with_anthropic_model(self):
        assert (
            detect_model_type("openrouter", "anthropic/claude-sonnet-4-5")
            == "anthropic"
        )

    def test_openrouter_with_openai_model(self):
        assert detect_model_type("openrouter", "openai/gpt-5-mini") == "openai"

    def test_openrouter_gemini_keyword(self):
        assert detect_model_type("openrouter", "gemini-2.5-pro") == "google"

    def test_openrouter_claude_keyword(self):
        assert detect_model_type("openrouter", "claude-opus-4") == "anthropic"

    def test_unknown_defaults_to_openai(self):
        assert detect_model_type("unknown", "some-model") == "openai"

    def test_no_model_name_defaults_to_openai(self):
        assert detect_model_type("openrouter") == "openai"


class TestGetImageConfigSectionName:
    def test_google(self):
        assert get_image_config_section_name("google") == "google_image_processing"

    def test_anthropic(self):
        assert (
            get_image_config_section_name("anthropic") == "anthropic_image_processing"
        )

    def test_openai(self):
        assert get_image_config_section_name("openai") == "api_image_processing"

    def test_unknown_defaults_to_api(self):
        assert get_image_config_section_name("other") == "api_image_processing"


class TestEncodeImageToBase64:
    def test_encode_png(self, tmp_path):
        img = Image.new("RGB", (10, 10), color="red")
        img_path = tmp_path / "test.png"
        img.save(img_path, format="PNG")

        b64_data, mime_type = encode_image_to_base64(img_path)
        assert mime_type == "image/png"
        assert len(b64_data) > 0
        # Verify it's valid base64
        decoded = base64.b64decode(b64_data)
        assert len(decoded) > 0

    def test_encode_jpeg(self, tmp_path):
        img = Image.new("RGB", (10, 10), color="blue")
        img_path = tmp_path / "test.jpg"
        img.save(img_path, format="JPEG")

        b64_data, mime_type = encode_image_to_base64(img_path)
        assert mime_type == "image/jpeg"

    def test_unsupported_format_raises(self, tmp_path):
        dummy_path = tmp_path / "test.xyz"
        dummy_path.write_bytes(b"dummy")

        with pytest.raises(ValueError, match="Unsupported image format"):
            encode_image_to_base64(dummy_path)


class TestCreateDataUrl:
    def test_basic_data_url(self):
        result = create_data_url("abc123", "image/png")
        assert result == "data:image/png;base64,abc123"

    def test_jpeg_data_url(self):
        result = create_data_url("data", "image/jpeg")
        assert result == "data:image/jpeg;base64,data"


class TestResizeForDetail:
    def _make_img(self, w, h, mode="RGB"):
        return Image.new(mode, (w, h), color="white")

    def test_low_detail_downscales(self):
        img = self._make_img(1000, 800)
        cfg = {"low_max_side_px": 512}
        result = ImageProcessor.resize_for_detail(img, "low", cfg)
        assert max(result.size) <= 512

    def test_low_detail_no_upscale(self):
        img = self._make_img(200, 100)
        cfg = {"low_max_side_px": 512}
        result = ImageProcessor.resize_for_detail(img, "low", cfg)
        assert result.size == (200, 100)

    def test_high_detail_openai_pads_to_box(self):
        img = self._make_img(500, 300)
        cfg = {"high_target_box": [768, 1536]}
        result = ImageProcessor.resize_for_detail(img, "high", cfg, model_type="openai")
        assert result.size == (768, 1536)

    def test_high_detail_anthropic_caps_side(self):
        img = self._make_img(3000, 2000)
        cfg = {"high_max_side_px": 1568}
        result = ImageProcessor.resize_for_detail(
            img, "high", cfg, model_type="anthropic"
        )
        assert max(result.size) <= 1568

    def test_anthropic_no_resize_if_small(self):
        img = self._make_img(800, 600)
        cfg = {"high_max_side_px": 1568}
        result = ImageProcessor.resize_for_detail(
            img, "high", cfg, model_type="anthropic"
        )
        assert result.size == (800, 600)

    def test_original_detail_caps_side(self):
        img = self._make_img(8000, 4000)
        cfg = {"original_max_side_px": 6000, "original_max_pixels": 10240000}
        result = ImageProcessor.resize_for_detail(img, "original", cfg)
        assert max(result.size) <= 6000

    def test_resize_profile_none_skips(self):
        img = self._make_img(5000, 3000)
        cfg = {"resize_profile": "none"}
        result = ImageProcessor.resize_for_detail(img, "high", cfg)
        assert result.size == (5000, 3000)

    def test_grayscale_image_pads_correctly(self):
        img = self._make_img(500, 300, mode="L")
        cfg = {"high_target_box": [768, 1536]}
        result = ImageProcessor.resize_for_detail(img, "high", cfg, model_type="openai")
        assert result.size == (768, 1536)
        assert result.mode == "L"


class TestComputeResizeScale:
    """The downscale ratio must mirror what resize_for_detail applies."""

    def test_none_profile_no_scale(self):
        cfg = {"resize_profile": "none"}
        assert ImageProcessor.compute_resize_scale(5000, 3000, "high", cfg) == 1.0

    def test_low_scale(self):
        cfg = {"low_max_side_px": 512}
        assert ImageProcessor.compute_resize_scale(1000, 800, "low", cfg) == 512 / 1000

    def test_low_no_upscale(self):
        cfg = {"low_max_side_px": 512}
        assert ImageProcessor.compute_resize_scale(200, 100, "low", cfg) == 1.0

    def test_original_within_caps(self):
        cfg = {"original_max_side_px": 6000, "original_max_pixels": 10240000}
        assert ImageProcessor.compute_resize_scale(1000, 800, "original", cfg) == 1.0

    def test_original_pixel_cap_binds(self):
        cfg = {"original_max_side_px": 6000, "original_max_pixels": 10240000}
        # 8000x4000 = 32 MP > 6000 side and > 10.24 MP: pixel cap is tighter
        scale = ImageProcessor.compute_resize_scale(8000, 4000, "original", cfg)
        assert scale == pytest.approx((10240000 / (8000 * 4000)) ** 0.5)

    def test_anthropic_high_uses_high_max_side(self):
        cfg = {"high_max_side_px": 2576}
        scale = ImageProcessor.compute_resize_scale(
            3000, 2000, "high", cfg, model_type="anthropic"
        )
        assert scale == 2576 / 3000

    def test_box_fit_downscale(self):
        cfg = {"high_target_box": [768, 1536]}
        scale = ImageProcessor.compute_resize_scale(
            1536, 2000, "high", cfg, model_type="openai"
        )
        assert scale == 0.5

    def test_box_fit_clamped_no_upscale(self):
        cfg = {"high_target_box": [768, 1536]}
        scale = ImageProcessor.compute_resize_scale(
            400, 300, "high", cfg, model_type="openai"
        )
        assert scale == 1.0


class TestComputeDirectRenderDpi:
    _ORIG_CFG = {
        "resize_profile": "original",
        "original_max_side_px": 6000,
        "original_max_pixels": 10240000,
    }

    def test_within_caps_keeps_target_dpi(self):
        # US Letter at 300 DPI = 2550x3300 = 8.4 MP, under all caps.
        dpi = ImageProcessor.compute_direct_render_dpi(
            612, 792, 300, "original", self._ORIG_CFG
        )
        assert dpi == 300

    def test_never_exceeds_target_dpi(self):
        # Tiny page would "want" a higher DPI for the box, but render never
        # upscales past target.
        cfg = {"resize_profile": "auto", "high_target_box": [768, 1536]}
        dpi = ImageProcessor.compute_direct_render_dpi(
            100, 100, 300, "high", cfg, model_type="openai"
        )
        assert dpi == 300

    def test_oversized_page_reduces_dpi(self):
        # Large fold-out: 1400x1000 pt at 300 DPI = 5833x4166 = 24.3 MP,
        # exceeds the 10.24 MP original pixel cap -> DPI drops.
        dpi = ImageProcessor.compute_direct_render_dpi(
            1400, 1000, 300, "original", self._ORIG_CFG
        )
        assert dpi < 300
        # Rendered pixels at the derived DPI stay within the cap.
        w = 1400 / 72 * dpi
        h = 1000 / 72 * dpi
        assert w * h <= 10240000 * 1.02  # rounding tolerance

    def test_anthropic_caps_long_edge(self):
        cfg = {"resize_profile": "auto", "high_max_side_px": 2576}
        # 800x600 pt at 300 DPI = 3333x2500; long edge 3333 > 2576 -> reduce.
        dpi = ImageProcessor.compute_direct_render_dpi(
            800, 600, 300, "auto", cfg, model_type="anthropic"
        )
        assert dpi < 300
        assert max(800 / 72 * dpi, 600 / 72 * dpi) <= 2576 * 1.02


class TestImageProcessorInit:
    def test_unsupported_extension_raises(self, tmp_path):
        bad_path = tmp_path / "test.doc"
        bad_path.write_bytes(b"dummy")
        with pytest.raises(ValueError, match="Unsupported image format"):
            ImageProcessor(bad_path, image_config={})


class TestImageProcessorProcessImage:
    def test_process_produces_jpeg(self, tmp_path):
        # Create a test PNG
        img = Image.new("RGBA", (100, 80), color=(255, 0, 0, 128))
        src = tmp_path / "test.png"
        img.save(src, format="PNG")

        cfg = {
            "api_image_processing": {
                "grayscale_conversion": True,
                "handle_transparency": True,
                "jpeg_quality": 85,
                "resize_profile": "none",
                "llm_detail": "high",
            }
        }
        processor = ImageProcessor(src, provider="openai", image_config=cfg)
        out_path = tmp_path / "output"
        result = processor.process_image(out_path)
        assert result.suffix == ".jpg"
        assert result.exists()

        # Verify the output is a valid JPEG
        with Image.open(result) as out_img:
            assert out_img.format == "JPEG"

    def test_process_handles_grayscale_toggle(self, tmp_path):
        img = Image.new("RGB", (50, 50), color="green")
        src = tmp_path / "test.png"
        img.save(src, format="PNG")

        cfg = {
            "api_image_processing": {
                "grayscale_conversion": False,
                "handle_transparency": True,
                "jpeg_quality": 95,
                "resize_profile": "none",
                "llm_detail": "high",
            }
        }
        processor = ImageProcessor(src, provider="openai", image_config=cfg)
        result = processor.process_image(tmp_path / "out")
        with Image.open(result) as out_img:
            assert out_img.mode == "RGB"
