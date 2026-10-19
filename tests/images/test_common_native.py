"""Tests for common/native.py: scan density, caps, sizing and encoding."""

from __future__ import annotations

import io
import math
from typing import Any

import fitz
import pytest
from PIL import Image

from chronominer.common.native import (
    MODEL_DERIVED_IMAGE_KEYS,
    changed_image_settings,
    encode_payload,
    format_downscale_log,
    guarded_payload,
    image_settings_fingerprint,
    model_image_cap,
    native_page_dpi,
    resized_size,
    resolve_target_size,
    validate_image_settings,
)


def scan_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    image.paste((40, 80, 120), (width // 4, height // 4, width // 2, height // 2))
    return encode_payload(image, "png", 95)[0]


def scan_page(doc: Any, dpi: int = 300, width: int = 144, height: int = 216) -> Any:
    page = doc.new_page(width=width, height=height)
    page.insert_image(
        page.rect,
        stream=scan_bytes(round(width * dpi / 72), round(height * dpi / 72)),
    )
    return page


def test_mrc_high_layer_and_small_stamp() -> None:
    with fitz.open() as doc:
        page = scan_page(doc, 150)
        page.insert_image(fitz.Rect(1, 1, 11, 11), stream=scan_bytes(600, 600))
        assert native_page_dpi(page, 300).dpi == pytest.approx(150)
        layer = Image.new("RGBA", (720, 1080), (0, 0, 0, 128))
        data = io.BytesIO()
        layer.save(data, format="PNG")
        page.insert_image(page.rect, stream=data.getvalue())
        page = doc.reload_page(page)
        assert native_page_dpi(page, 300).dpi == pytest.approx(360)


@pytest.mark.parametrize("geometry", ["rotate", "crop", "anisotropic", "diagonal"])
def test_native_geometry(geometry: str) -> None:
    with fitz.open() as doc:
        if geometry == "diagonal":
            with fitz.open() as original:
                scan_page(original, 300, 144, 144)
                page = doc.new_page(width=144, height=144)
                page.show_pdf_page(page.rect, original, 0, rotate=45)
                # Rotating into the same square shrinks the placement by sqrt(2).
                expected = 300 * math.sqrt(2)
        else:
            page = scan_page(doc, 300)
            expected = 300
            if geometry == "rotate":
                page.set_rotation(90)
            elif geometry == "crop":
                page.set_cropbox(fitz.Rect(72, 0, 144, 216))
            else:
                page = doc.new_page(width=144, height=216)
                page.insert_image(
                    page.rect, stream=scan_bytes(600, 450), keep_proportion=False
                )
        density = native_page_dpi(page, 72)
        assert density.dpi_x == pytest.approx(expected, abs=0.5)
        assert density.dpi_y == pytest.approx(
            150 if geometry == "anisotropic" else expected,
            abs=0.5,
        )


@pytest.mark.parametrize("kind", ["text", "pixel", "figure"])
def test_native_fallback(kind: str) -> None:
    with fitz.open() as doc:
        page = doc.new_page(width=144, height=216)
        if kind == "text":
            page.insert_text((10, 20), "Born digital")
        elif kind == "pixel":
            page.insert_image(page.rect, stream=scan_bytes(1, 1))
        else:
            page.insert_image(fitz.Rect(0, 0, 60, 60), stream=scan_bytes(600, 600))
        result = native_page_dpi(page, 240)
        assert result.source == "fallback"
        assert result.dpi == 240


def test_undecodable_image_fails_the_page() -> None:
    with fitz.open() as doc:
        page = scan_page(doc)
        xref = page.get_images()[0][0]
        doc.xref_set_key(xref, "Filter", "/JPXDecode")
        page = doc.reload_page(page)
        with pytest.raises(fitz.mupdf.FzErrorBase, match="JPX"):
            native_page_dpi(page, 300)


def test_anthropic_published_resize() -> None:
    assert resized_size(1075, 1520, 1568, 1568) == (924, 1307)


@pytest.mark.parametrize(
    "model_type,detail,flags,policy",
    [
        ("openai", "original", {}, "openai-original-10k-v1"),
        (
            "openai",
            "original",
            {"image_original_patch_cap_30k": True},
            "openai-original-30k-v1",
        ),
        (
            "openrouter",
            "original",
            {"image_original_patch_cap_30k": True},
            "openai-original-10k-v1",
        ),
        ("anthropic", "high", {}, "anthropic-standard-v1"),
        ("anthropic", "auto", {"image_high_res_tier": True}, "anthropic-high-v1"),
    ],
)
def test_caps_come_from_flags(
    model_type: str, detail: str, flags: dict[str, Any], policy: str
) -> None:
    cap = model_image_cap(model_type, "any-model", detail, flags)
    assert cap is not None and cap.policy == policy


@pytest.mark.parametrize(
    "model_type,detail",
    [("openai", "high"), ("anthropic", "low"), ("google", "original")],
)
def test_no_cap_outside_original_and_anthropic(model_type: str, detail: str) -> None:
    assert model_image_cap(model_type, "any-model", detail) is None


def test_unknown_original_is_conservative() -> None:
    size, reason = resolve_target_size(5000, 8000, "openai", "unknown", "original", {})
    assert math.ceil(size[0] / 32) * math.ceil(size[1] / 32) <= 10000
    assert reason == "model_cap"


def test_profile_sizes_without_cap() -> None:
    cfg = {"high_target_box": [768, 1536], "low_max_side_px": 512}
    assert resolve_target_size(1536, 1536, "google", "m", "high", cfg) == (
        (768, 768),
        "profile",
    )
    assert resolve_target_size(1000, 500, "google", "m", "low", cfg) == (
        (512, 256),
        "profile",
    )
    assert resolve_target_size(300, 400, "google", "m", "high", cfg) == (
        (300, 400),
        "none",
    )
    unbounded = {**cfg, "resize_profile": "none"}
    assert resolve_target_size(9000, 9000, "google", "m", "high", unbounded) == (
        (9000, 9000),
        "none",
    )
    assert resolve_target_size(5000, 7000, "google", "m", "original", cfg) == (
        (5000, 7000),
        "none",
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("target_dpi", 0),
        ("target_dpi", "300"),
        ("target_dpi", True),
        ("native_fallback_dpi", None),
        ("native_fallback_dpi", 1.5),
        ("payload_format", "gif"),
        ("max_image_bytes", -1),
    ],
)
def test_invalid_settings(key: str, value: Any) -> None:
    with pytest.raises(ValueError, match=f"{key}.*api_image_processing"):
        validate_image_settings({key: value}, "api_image_processing")


def test_native_target_dpi_is_valid() -> None:
    validate_image_settings({"target_dpi": "native"}, "api_image_processing")


def test_encode_keeps_mode_and_converts_others() -> None:
    gray, mime = encode_payload(Image.new("L", (8, 8), 128), "png", 95)
    assert mime == "image/png"
    assert Image.open(io.BytesIO(gray)).mode == "L"
    rgba, mime = encode_payload(Image.new("RGBA", (8, 8)), "jpeg", 95)
    assert mime == "image/jpeg"
    assert Image.open(io.BytesIO(rgba)).mode == "RGB"
    with pytest.raises(ValueError, match="payload_format"):
        encode_payload(Image.new("L", (8, 8)), "gif", 95)


def test_guarded_payload_without_limit_keeps_format() -> None:
    image = Image.new("L", (16, 16), 200)
    data, mime, fallback = guarded_payload(image, "png", 95, 0)
    assert (mime, fallback) == ("image/png", False)
    assert Image.open(io.BytesIO(data)).format == "PNG"


def test_fingerprint_ignores_key_order() -> None:
    settings = {"target_dpi": "native", "jpeg_quality": 95, "box": [768, 1536]}
    reordered = dict(reversed(list(settings.items())))
    assert image_settings_fingerprint(settings) == image_settings_fingerprint(reordered)
    assert image_settings_fingerprint(settings) != image_settings_fingerprint(
        {**settings, "jpeg_quality": 90}
    )


_RECORDED = {
    "model_name": "model-a",
    "provider": "openai",
    "model_type": "openai",
    "request_detail": "original",
    "resolved_detail": "original",
    "cap_policy": "openai-original-30k-v1",
    "image_original_patch_cap_30k": True,
    "image_high_res_tier": False,
    "target_dpi": "native",
    "jpeg_quality": 95,
}
_SWITCHED = {
    **_RECORDED,
    "model_name": "vendor/model-b",
    "provider": "openrouter",
    "request_detail": None,
    "cap_policy": "openai-original-10k-v1",
    "image_original_patch_cap_30k": False,
}


def test_changed_image_settings_skips_the_model_on_a_model_switch() -> None:
    assert changed_image_settings(_RECORDED, _SWITCHED) == []
    assert changed_image_settings(_RECORDED, {**_SWITCHED, "target_dpi": 200}) == [
        "target_dpi"
    ]


def test_changed_image_settings_compares_every_key_for_the_same_model() -> None:
    current = {**_RECORDED, "resolved_detail": "high", "jpeg_quality": 80}
    assert changed_image_settings(_RECORDED, current) == [
        "jpeg_quality",
        "resolved_detail",
    ]


def test_changed_image_settings_takes_extra_derived_keys() -> None:
    switched = {**_SWITCHED, "llm_detail": "high"}
    recorded = {**_RECORDED, "llm_detail": "original"}
    assert changed_image_settings(recorded, switched) == ["llm_detail"]
    derived = MODEL_DERIVED_IMAGE_KEYS | {"llm_detail"}
    assert changed_image_settings(recorded, switched, derived) == []


def test_downscale_log_names_sizes_and_policy() -> None:
    provenance = {
        "source_width": 800,
        "source_height": 1200,
        "width": 400,
        "height": 600,
        "source_dpi_x": 300.0,
        "sent_dpi": None,
        "dpi_source": "image",
        "downscale_reason": "model_cap",
        "cap_policy": "openai-original-10k-v1",
    }
    line = format_downscale_log(3, provenance, "model-x")
    assert line.startswith("page 3: image 300.0 dpi 800x1200")
    assert "sent unknown dpi 400x600" in line
    assert line.endswith("reason model_cap (model-x, openai-original-10k-v1)")
