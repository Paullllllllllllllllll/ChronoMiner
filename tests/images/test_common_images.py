"""Tests for common/images.py: preprocessing, names, sources and streaming."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from pathlib import Path
from typing import Any

import fitz
import pytest
from PIL import Image

from chronominer.common.images import (
    DEFAULT_HIGH_TARGET_BOX,
    FolderPageSource,
    PageError,
    PagePayload,
    PdfPageSource,
    _resize_box_fit,
    check_failure_rate,
    content_scale_factor,
    google_media_resolution,
    list_folder_images,
    natural_sort_key,
    preprocess_image,
    resampling_filter,
    resize_for_detail,
    sequence_number,
    stream_payloads,
)
from chronominer.common.native import image_settings_fingerprint

# OpenAI "original" detail bounds: longest side and total pixels.
ORIGINAL_MAX_SIDE_PX = 6000
ORIGINAL_MAX_PIXELS = 10_240_000


@pytest.fixture
def openai_config() -> dict[str, Any]:
    """OpenAI image config for testing."""
    return {
        "grayscale_conversion": True,
        "handle_transparency": True,
        "llm_detail": "high",
        "low_max_side_px": 512,
        "high_target_box": [768, 1536],
        "resize_profile": "auto",
    }


@pytest.fixture
def anthropic_config() -> dict[str, Any]:
    """Anthropic image config for testing."""
    return {
        "grayscale_conversion": True,
        "handle_transparency": True,
        "low_max_side_px": 512,
        "high_max_side_px": 1568,
        "resize_profile": "auto",
    }


class TestResizeForDetail:
    """Resizing by the requested image detail."""

    @pytest.mark.parametrize(("model_type", "cap"), [("openai", 512), ("custom", 250)])
    def test_low_detail_caps_size(self, model_type: str, cap: int) -> None:
        """Low detail caps the longest side to low_max_side_px from the config."""
        large_image = Image.new("RGB", (2000, 1500))
        result = resize_for_detail(
            large_image, "low", {"low_max_side_px": cap}, model_type
        )

        assert max(result.size) <= cap

    def test_low_detail_small_image_unchanged(
        self, openai_config: dict[str, Any]
    ) -> None:
        """Small images in low detail mode are unchanged."""
        small_image = Image.new("RGB", (300, 200))
        result = resize_for_detail(small_image, "low", openai_config, "openai")

        assert result.size == (300, 200)

    def test_high_detail_openai_box_fit(self, openai_config: dict[str, Any]) -> None:
        """OpenAI high detail uses box fitting with padding."""
        image = Image.new("RGB", (2000, 1500))
        result = resize_for_detail(image, "high", openai_config, "openai")

        assert result.size == (768, 1536)

    def test_high_detail_anthropic_max_side(
        self, anthropic_config: dict[str, Any]
    ) -> None:
        """Anthropic high detail caps longest side without padding."""
        image = Image.new("RGB", (3000, 2000))
        result = resize_for_detail(image, "high", anthropic_config, "anthropic")

        assert max(result.size) <= 1568
        assert result.size != (768, 1536)

    def test_resize_profile_none_skips_resize(self) -> None:
        """resize_profile='none' skips resizing entirely."""
        config = {"resize_profile": "none"}
        image = Image.new("RGB", (5000, 4000))
        result = resize_for_detail(image, "high", config, "openai")

        assert result.size == (5000, 4000)

    @pytest.mark.parametrize("detail", ["auto", "nonsense"])
    def test_auto_and_unknown_detail_treated_as_high(
        self, openai_config: dict[str, Any], detail: str
    ) -> None:
        """Auto and unknown detail values take the high-detail behavior."""
        image = Image.new("RGB", (2000, 1500))
        result = resize_for_detail(image, detail, openai_config, "openai")

        assert result.size == (768, 1536)

    def test_original_caps_oversized_image(self, openai_config: dict[str, Any]) -> None:
        """'original' caps an oversized scan to max side and pixel limit."""
        image = Image.new("RGB", (7000, 9000))
        result = resize_for_detail(image, "original", openai_config, "openai")

        w, h = result.size
        assert max(w, h) <= ORIGINAL_MAX_SIDE_PX
        assert w * h <= ORIGINAL_MAX_PIXELS
        # Aspect ratio preserved (7:9).
        assert abs((w / h) - (7000 / 9000)) < 0.01

    def test_original_passes_small_image_through(
        self, openai_config: dict[str, Any]
    ) -> None:
        """'original' leaves an image within the caps untouched (no upscale)."""
        image = Image.new("RGB", (1000, 1500))
        result = resize_for_detail(image, "original", openai_config, "openai")

        assert result.size == (1000, 1500)

    def test_original_overrides_resize_profile_none(self) -> None:
        """resize_profile='none' does not defeat the 'original' cap."""
        config = {"resize_profile": "none"}
        image = Image.new("RGB", (7000, 9000))
        result = resize_for_detail(image, "original", config, "openai")

        w, h = result.size
        assert max(w, h) <= ORIGINAL_MAX_SIDE_PX
        assert w * h <= ORIGINAL_MAX_PIXELS

    def test_custom_original_caps_from_config(self) -> None:
        """'original' reads original_max_side_px / original_max_pixels overrides."""
        config = {"original_max_side_px": 2000, "original_max_pixels": 1_000_000}
        image = Image.new("RGB", (7000, 9000))
        result = resize_for_detail(image, "original", config, "openai")

        w, h = result.size
        assert max(w, h) <= 2000
        assert w * h <= 1_000_000

    @pytest.mark.parametrize("model_type", ["google", "custom"])
    def test_original_without_cap_keeps_rendered_size(
        self, openai_config: dict[str, Any], model_type: str
    ) -> None:
        """Without a model cap, 'original' neither fits the box nor pads."""
        image = Image.new("RGB", (2000, 3500))
        result = resize_for_detail(image, "original", openai_config, model_type)

        assert result is image


class TestResizeHelpers:
    """The box-fit and max-side resize helpers and the resampling filter."""

    def test_resize_box_fit_pads_grayscale_canvas(self) -> None:
        """Box fitting an L-mode image keeps the canvas grayscale."""
        image = Image.new("L", (2000, 1500), color=128)
        result = _resize_box_fit(image, {"high_target_box": [768, 1536]})

        assert result.size == (768, 1536)
        assert result.mode == "L"

    def test_resize_box_fit_invalid_box_uses_defaults(self) -> None:
        """An invalid high_target_box falls back to default dimensions."""
        image = Image.new("RGB", (2000, 1500))
        result = _resize_box_fit(image, {"high_target_box": "not-a-box"})

        assert result.size == DEFAULT_HIGH_TARGET_BOX

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            (None, Image.Resampling.BILINEAR),
            ({"resampling_algorithm": "LANCZOS"}, Image.Resampling.LANCZOS),
            ({"resampling_algorithm": "x"}, Image.Resampling.BILINEAR),
        ],
    )
    def test_resampling_filter_names(
        self, config: dict[str, Any] | None, expected: Image.Resampling
    ) -> None:
        """A named algorithm is used; a missing or unknown one gives bilinear."""
        assert resampling_filter(config) == expected


class TestContentScaleFactor:
    """content_scale_factor, from which the render DPI is derived."""

    def test_box_fit_downscale_factor(self, openai_config: dict[str, Any]) -> None:
        """OpenAI box-fit returns the min-dimension downscale factor."""
        # 833x1250 source, box [768, 1536] -> width-bound: 768/833.
        factor = content_scale_factor((833.0, 1250.0), openai_config, "openai")
        assert factor == pytest.approx(768.0 / 833.0, rel=1e-6)

    def test_box_fit_can_exceed_one(self, openai_config: dict[str, Any]) -> None:
        """Box-fit upscales small pages, so the factor can exceed 1.0."""
        factor = content_scale_factor((200.0, 300.0), openai_config, "openai")
        assert factor > 1.0

    def test_anthropic_max_side_factor(self, anthropic_config: dict[str, Any]) -> None:
        """Anthropic obeys both its edge limit and patch limit."""
        factor = content_scale_factor((2000.0, 3000.0), anthropic_config, "anthropic")
        assert factor == pytest.approx(1344.0 / 3000.0, rel=1e-6)

    def test_within_cap_returns_one(self, anthropic_config: dict[str, Any]) -> None:
        """A source already within the cap yields a factor of 1.0."""
        factor = content_scale_factor((1000.0, 1200.0), anthropic_config, "anthropic")
        assert factor == 1.0

    def test_resize_profile_none_returns_one(self) -> None:
        """resize_profile='none' disables derivation (factor 1.0)."""
        cfg = {"resize_profile": "none", "llm_detail": "high"}
        factor = content_scale_factor((5000.0, 4000.0), cfg, "openai")
        assert factor == 1.0

    def test_original_within_caps_returns_one(self) -> None:
        """'original' whose source fits both caps yields 1.0."""
        cfg = {
            "llm_detail": "original",
            "original_max_side_px": 6000,
            "original_max_pixels": 10_240_000,
        }
        factor = content_scale_factor((833.0, 1250.0), cfg, "openai")
        assert factor == 1.0

    def test_original_oversized_downscales(self) -> None:
        """'original' whose source exceeds the caps yields a factor < 1."""
        cfg = {
            "llm_detail": "original",
            "original_max_side_px": 6000,
            "original_max_pixels": 10_240_000,
        }
        factor = content_scale_factor((7000.0, 9000.0), cfg, "openai")
        assert 0 < factor < 1

    @pytest.mark.parametrize("size", [(200.0, 300.0), (5000.0, 7000.0)])
    def test_original_without_cap_returns_one(self, size: tuple[float, float]) -> None:
        """'original' without a model cap keeps the rendered size."""
        cfg = {"media_resolution": "original", "high_target_box": [768, 1536]}
        assert content_scale_factor(size, cfg, "google") == 1.0


class TestPreprocessImage:
    """Mode conversion, transparency and resizing in preprocess_image()."""

    @pytest.mark.parametrize(("grayscale", "mode"), [(True, "L"), (False, "RGB")])
    def test_grayscale_follows_the_setting(
        self, openai_config: dict[str, Any], grayscale: bool, mode: str
    ) -> None:
        """RGB input is converted to grayscale only when enabled."""
        openai_config["grayscale_conversion"] = grayscale
        image = Image.new("RGB", (1000, 1500), color=(128, 64, 192))
        result = preprocess_image(image, openai_config, "openai")

        assert result.mode == mode

    @pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
    def test_transparency_flattened_to_rgb(
        self, openai_config: dict[str, Any], mode: str
    ) -> None:
        """RGBA, LA and palette input with transparency is flattened to RGB."""
        openai_config["grayscale_conversion"] = False
        image = Image.new(mode, (100, 100))
        if mode == "P":
            image.info["transparency"] = 0
        result = preprocess_image(image, openai_config, "openai")

        assert result.mode == "RGB"

    def test_transparency_kept_when_disabled(
        self, openai_config: dict[str, Any]
    ) -> None:
        """Transparency handling is skipped when disabled."""
        openai_config["handle_transparency"] = False
        openai_config["grayscale_conversion"] = False
        openai_config["resize_profile"] = "none"
        image = Image.new("RGBA", (100, 100), color=(128, 64, 192, 128))
        result = preprocess_image(image, openai_config, "openai")

        assert result.mode == "RGBA"

    def test_openai_box_fit_applied(self, openai_config: dict[str, Any]) -> None:
        """OpenAI input is fitted into the configured target box."""
        image = Image.new("RGB", (1000, 1500))
        result = preprocess_image(image, openai_config, "openai")

        assert result.size == (768, 1536)

    def test_google_uses_media_resolution(self) -> None:
        """Google model type reads the media_resolution config key."""
        cfg = {
            "grayscale_conversion": False,
            "handle_transparency": False,
            "media_resolution": "low",
            "low_max_side_px": 256,
            "high_target_box": [768, 768],
        }
        image = Image.new("RGB", (1000, 1000))
        result = preprocess_image(image, cfg, "google")

        assert max(result.size) <= 256

    def test_anthropic_uses_resize_profile(
        self, anthropic_config: dict[str, Any]
    ) -> None:
        """Anthropic model type caps the longest side without padding."""
        anthropic_config["grayscale_conversion"] = False
        image = Image.new("RGB", (2000, 3000))
        result = preprocess_image(image, anthropic_config, "anthropic")

        assert max(result.size) <= 1568

    def test_grayscale_already_grayscale_noop(
        self, openai_config: dict[str, Any]
    ) -> None:
        """Grayscale conversion on an already grayscale image is a no-op."""
        openai_config["resize_profile"] = "none"
        image = Image.new("L", (100, 100), color=128)
        result = preprocess_image(image, openai_config, "openai")

        assert result.mode == "L"
        assert result.size == (100, 100)

    @pytest.mark.parametrize("mode", ["I", "I;16"])
    def test_sixteen_bit_gray_rescaled_not_clipped(
        self, openai_config: dict[str, Any], mode: str
    ) -> None:
        """16-bit scans are scaled to 8 bits instead of clipped to white."""
        openai_config["resize_profile"] = "none"
        width = 256
        image = Image.new(mode, (width, 8))
        row = [x * 257 for x in range(width)]
        image.putdata(row * 8)

        result = preprocess_image(image, openai_config, "openai")

        assert result.mode == "L"
        values = list(result.tobytes())
        near_white = sum(1 for v in values if v >= 250) / len(values)
        assert near_white < 0.1
        assert max(values) - min(values) > 200

    def test_sixteen_bit_endian_variants_supported(
        self, openai_config: dict[str, Any]
    ) -> None:
        """The "I;16B/L/N" variants are promoted before point() is applied."""
        openai_config["resize_profile"] = "none"
        for mode in ("I;16B", "I;16L", "I;16N"):
            image = Image.new(mode, (16, 16))
            result = preprocess_image(image, openai_config, "openai")
            assert result.mode == "L"


def _settings(**overrides: Any) -> dict[str, Any]:
    """Unbounded custom-model settings: no cap, no resize."""
    return {
        "target_dpi": 72,
        "payload_format": "png",
        "max_image_bytes": 0,
        "resize_profile": "none",
        "resolved_detail": "high",
        "model_name": "test-model",
        "cap_policy": "test-cap",
        **overrides,
    }


def _make_pdf(path: Path, pages: int) -> Path:
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page(width=144, height=216)
    doc.save(path)
    doc.close()
    return path


def _collect(source: Any, indices: list[int] | None = None) -> list[Any]:
    async def run() -> list[Any]:
        return [item async for item in stream_payloads(source, indices)]

    return asyncio.run(run())


class TestNames:
    def test_natural_sort_orders_digit_runs_numerically(self) -> None:
        names = ["p_10.png", "p_2.png", "P_1.png"]
        assert sorted(names, key=natural_sort_key) == ["P_1.png", "p_2.png", "p_10.png"]

    def test_list_folder_images_filters_and_sorts(self, tmp_path: Path) -> None:
        for name in ("b_10.jpg", "b_2.PNG", "notes.txt"):
            (tmp_path / name).write_bytes(b"")
        (tmp_path / "dir.jpg").mkdir()

        listed = [p.name for p in list_folder_images(tmp_path)]

        assert listed == ["b_2.PNG", "b_10.jpg"]

    def test_list_folder_images_takes_extensions(self, tmp_path: Path) -> None:
        for name in ("a.jpg", "b.jp2"):
            (tmp_path / name).write_bytes(b"")
        assert [p.name for p in list_folder_images(tmp_path, {".jp2"})] == ["b.jp2"]

    @pytest.mark.parametrize(
        "name,expected",
        [("page_0007.jpg", 7), ("scan12b.png", 12), ("cover.png", 0)],
    )
    def test_sequence_number(self, name: str, expected: int) -> None:
        assert sequence_number(Path(name)) == expected

    @pytest.mark.parametrize(
        "detail,expected",
        [
            ("high", "MEDIA_RESOLUTION_HIGH"),
            (" Ultra_High ", "MEDIA_RESOLUTION_ULTRA_HIGH"),
            ("media_resolution_low", "MEDIA_RESOLUTION_LOW"),
            ("original", None),
            (None, None),
        ],
    )
    def test_google_media_resolution(
        self, detail: str | None, expected: str | None
    ) -> None:
        assert google_media_resolution(detail) == expected


class TestPageSources:
    def test_pdf_pages_use_the_name_template(self, tmp_path: Path) -> None:
        class Named(PdfPageSource):
            name_template = "page_{number:04d}_pre_processed.jpg"

        pdf = _make_pdf(tmp_path / "doc.pdf", 2)
        with Named(pdf, _settings(), "custom") as source:
            payload = source.build_payload(1)

        assert PdfPageSource.image_name(0) == "page_0001.jpg"
        assert payload.image_name == "page_0002_pre_processed.jpg"
        assert (payload.number, payload.page_index) == (2, 1)

    def test_pdf_payload_records_provenance(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "doc.pdf", 1)
        with PdfPageSource(
            pdf, _settings(), "custom", payload_sha256_key="image_sha256"
        ) as source:
            payload = source.build_payload(0)
            record = source.file_provenance()

        assert payload.mime_type == "image/png"
        assert (payload.provenance["width"], payload.provenance["height"]) == (
            144,
            216,
        )
        assert payload.provenance["effective_dpi"] == 72.0
        assert (
            payload.provenance["image_sha256"]
            == hashlib.sha256(base64.b64decode(payload.base64)).hexdigest()
        )
        assert record["cap_policy"] == "test-cap"
        assert record["size"] == pdf.stat().st_size
        assert record["image_settings_fingerprint"] == image_settings_fingerprint(
            _settings()
        )

    def test_folder_source_reads_images_in_order(self, tmp_path: Path) -> None:
        for name in ("p_10.png", "p_2.png"):
            Image.new("RGB", (40, 60), "white").save(tmp_path / name)

        with FolderPageSource(tmp_path, _settings(), "custom") as source:
            names = [source.image_name(i) for i in range(len(source))]
            payload = source.build_payload(0)
            record = source.file_provenance()

        assert names == ["p_2.png", "p_10.png"]
        assert payload.number == 2
        assert payload.provenance["width"] == 40
        assert "image_sha256" not in payload.provenance
        assert (record["image_count"], record["total_image_bytes"]) == (
            2,
            sum(p.stat().st_size for p in tmp_path.iterdir()),
        )

    def test_original_without_cap_sends_the_rendered_size(self, tmp_path: Path) -> None:
        """A PDF page at 'original' detail and no cap is neither boxed nor padded."""
        pdf = _make_pdf(tmp_path / "doc.pdf", 1)
        cfg = _settings(
            target_dpi=300, resize_profile="high", resolved_detail="original"
        )
        with PdfPageSource(pdf, cfg, "google") as source:
            payload = source.build_payload(0)

        provenance = payload.provenance
        assert (provenance["width"], provenance["height"]) == (600, 900)
        assert provenance["downscale_reason"] == "none"
        assert provenance["sent_dpi"] == pytest.approx(300.0)

    def test_original_without_cap_keeps_the_byte_guard(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "doc.pdf", 1)
        cfg = _settings(
            target_dpi=300,
            resize_profile="high",
            resolved_detail="original",
            max_image_bytes=100,
        )
        with (
            PdfPageSource(pdf, cfg, "google") as source,
            pytest.raises(ValueError, match="max_image_bytes"),
        ):
            source.build_payload(0)


class TestStreaming:
    def test_streams_selected_pages_in_order(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "doc.pdf", 4)
        with PdfPageSource(pdf, _settings(), "custom") as source:
            results = _collect(source, [2, 0, 9])

        assert all(isinstance(r, PagePayload) for r in results)
        assert [r.page_index for r in results] == [2, 0]

    def test_failed_page_yields_page_error(self, tmp_path: Path) -> None:
        for name in ("a_1.png", "a_2.png", "a_3.png"):
            Image.new("L", (20, 20), 255).save(tmp_path / name)
        (tmp_path / "a_2.png").write_bytes(b"not an image")

        with FolderPageSource(tmp_path, _settings(), "custom") as source:
            results = _collect(source)

        assert [type(r) for r in results] == [PagePayload, PageError, PagePayload]
        assert results[1].image_name == "a_2.png"

    def test_excessive_failures_raise(self, tmp_path: Path) -> None:
        for name in ("a_1.png", "a_2.png", "a_3.png"):
            (tmp_path / name).write_bytes(b"broken")

        with (
            FolderPageSource(tmp_path, _settings(), "custom") as source,
            pytest.raises(RuntimeError, match="3/3 pages"),
        ):
            _collect(source)

    def test_single_failure_only_warns(self) -> None:
        check_failure_rate("doc.pdf", 1, 1)
        with pytest.raises(RuntimeError):
            check_failure_rate("doc.pdf", 4, 2)
