"""Image settings of a model and their fingerprint."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from chronominer.common.native import (
    image_settings_fingerprint,
    model_image_cap,
    resolve_target_size,
)
from chronominer.extract import images as images_module
from chronominer.extract.images import (
    detect_model_type,
    file_provenance,
    image_section,
    resolve_image_settings,
    resume_images,
)
from chronominer.output.store import image_changes, image_settings_record
from chronominer.spec import ImageSpec


def settings(
    model: str = "gpt-6-astra", provider: str = "openai", **images: Any
) -> dict[str, Any]:
    return resolve_image_settings(provider, model, ImageSpec(**images))[0]


@pytest.fixture
def sections(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A writable copy of the recommended sections."""
    copy = json.loads(json.dumps(dict(images_module.IMAGE_PROCESSING)))
    monkeypatch.setattr(images_module, "IMAGE_PROCESSING", copy)
    return copy


@pytest.mark.parametrize(
    "provider,model,expected",
    [
        ("openai", "gpt-6-astra", "openai"),
        ("anthropic", "anything", "anthropic"),
        ("custom", "gemini-like", "custom"),
        ("openrouter", "google/gemini-3-flash", "google"),
        ("openrouter", "anthropic/claude-opus-5", "anthropic"),
        ("openrouter", "meta/llama-4", "custom"),
        ("openrouter", "openai/gpt-5-mini", "openai"),
    ],
)
def test_model_type_and_section(provider: str, model: str, expected: str) -> None:
    assert detect_model_type(provider, model) == expected
    cfg, model_type, section = resolve_image_settings(provider, model)
    assert model_type == expected
    assert section == image_section(expected)
    assert cfg["provider"] == provider
    assert cfg["model_name"] == model


def test_the_model_name_is_lowercased() -> None:
    assert settings("GPT-4o")["model_name"] == "gpt-4o"


def test_recommended_openai_settings() -> None:
    cfg = settings("gpt-5.6-luna")
    assert (cfg["resolved_detail"], cfg["detail"]) == ("original", "original")
    assert cfg["cap_policy"] == "openai-original-30k-v1"
    assert (cfg["target_dpi"], cfg["grayscale_conversion"]) == ("native", False)
    assert (cfg["max_pixels_per_page"], cfg["render_strategy"]) == (
        24_000_000,
        "direct",
    )


def test_recommended_custom_settings() -> None:
    cfg = settings("org/model", "custom")
    assert (cfg["target_dpi"], cfg["jpeg_quality"], cfg["resize_profile"]) == (
        150,
        95,
        "low",
    )
    assert (cfg["low_max_side_px"], cfg["high_target_box"]) == (768, [512, 1024])
    assert cfg["resolved_detail"] == "high"


def test_run_values_override_the_recommendation() -> None:
    cfg = settings("gpt-5.6-luna", dpi=150, format="png", jpeg_quality=70)
    assert (cfg["target_dpi"], cfg["payload_format"], cfg["jpeg_quality"]) == (
        150,
        "png",
        70,
    )
    assert settings(grayscale=True)["grayscale_conversion"] is True


def test_a_detail_applies_only_where_the_section_takes_it() -> None:
    assert settings("gpt-5.6-luna", detail="low")["resolved_detail"] == "low"
    google = settings("gemini-3-flash-preview", "google", detail="original")
    assert google["media_resolution"] == "high"
    anthropic = settings("claude-opus-5", "anthropic", detail="low")
    assert anthropic["resize_profile"] == "low"


def test_openai_keeps_auto_and_falls_back_to_high() -> None:
    assert settings("gpt-4o", detail="auto")["resolved_detail"] == "auto"
    cfg = settings("gpt-4o")
    assert cfg["resolved_detail"] == "high"
    assert cfg["cap_policy"] == "profile-v1"


def test_invalid_run_value_names_the_section() -> None:
    with pytest.raises(ValueError, match="target_dpi.*api_image_processing"):
        settings(dpi=0)


@pytest.mark.parametrize(
    "model,provider,side,patches,patch_px",
    [
        ("claude-opus-5", "anthropic", 2576, 4784, 28),
        ("claude-haiku-4-5", "anthropic", 1568, 1568, 28),
        ("gpt-5.4", "openai", 6000, 10000, 32),
        ("gpt-6-astra", "openai", 65535, 30000, 32),
    ],
)
def test_capability_caps(
    model: str, provider: str, side: int, patches: int, patch_px: int
) -> None:
    cfg = settings(model, provider)
    cap = model_image_cap(provider, model, cfg["resolved_detail"], cfg)
    assert cap is not None
    assert (cap.max_side, cap.patches, cap.patch_px) == (side, patches, patch_px)
    for width, height in [(5000, 8000), (10000, 100), (3199, 3201), (3200, 3200)]:
        (w, h), reason = resolve_target_size(
            width, height, provider, model, cfg["resolved_detail"], cfg
        )
        assert max(w, h) <= side
        assert math.ceil(w / patch_px) * math.ceil(h / patch_px) <= patches
        assert reason in ("none", "model_cap")


def test_luna_original_pages_are_limited_to_3000_px() -> None:
    cfg = settings("gpt-6-luna")
    assert cfg["original_max_side_px"] == 3000
    size, reason = resolve_target_size(
        4000, 6000, "openai", "gpt-6-luna", "original", cfg
    )
    assert (size, reason) == ((2000, 3000), "model_cap")


@pytest.mark.parametrize(
    "model,provider,detail",
    [
        ("gpt-6.1-sol", "openai", None),
        ("gpt-6-astra", "openai", None),
        ("gpt-5.6-luna", "openai", None),
        ("gpt-6-luna", "openai", "high"),
        ("openai/gpt-6-luna", "openrouter", None),
        ("claude-opus-5", "anthropic", None),
    ],
)
def test_other_models_and_details_keep_their_sizing(
    model: str, provider: str, detail: str | None
) -> None:
    cfg = settings(model, provider, detail=detail)
    assert "original_max_side_px" not in cfg


def test_the_setting_overrides_the_model_default() -> None:
    def side(model: str, value: int | None) -> int | None:
        cfg = resolve_image_settings("openai", model, None, value)[0]
        return cfg.get("original_max_side_px")

    assert side("gpt-6-luna", 2500) == 2500
    assert side("gpt-6.1-sol", 2000) == 2000
    assert side("gpt-6-luna", None) == 3000
    assert side("gpt-6.1-sol", None) is None


def test_openrouter_takes_no_original_detail() -> None:
    cfg = settings("openai/gpt-6-astra", "openrouter")
    assert cfg["resolved_detail"] == "high"
    assert not cfg["image_original_patch_cap_30k"]
    assert cfg["cap_policy"] == "profile-v1"


def test_openrouter_claude_is_capped_by_its_model() -> None:
    cfg = settings("anthropic/claude-opus-5", "openrouter")
    assert cfg["resolved_detail"] == "high"
    assert cfg["image_high_res_tier"]
    size, _ = resolve_target_size(
        4000, 7000, "anthropic", cfg["model_name"], cfg["resolved_detail"], cfg
    )
    assert max(size) > 1568


@pytest.mark.parametrize("detail", ["high", "low"])
def test_openrouter_takes_the_openai_section_detail(
    sections: dict[str, Any], detail: str
) -> None:
    sections["api_image_processing"]["llm_detail"] = detail
    sections["google_image_processing"]["media_resolution"] = "low"
    cfg = settings("google/gemini-3-flash", "openrouter")
    assert cfg["resolved_detail"] == detail


@pytest.mark.parametrize("provider", ["google", "custom"])
def test_unbounded_native_render_is_rejected(
    sections: dict[str, Any], provider: str
) -> None:
    sections[f"{provider}_image_processing"]["resize_profile"] = "none"
    with pytest.raises(ValueError, match=f"resize_profile.*{provider}_image"):
        settings("model", provider, dpi="native")


def test_the_stored_record_carries_the_fingerprinted_values() -> None:
    cfg = settings()
    record = image_settings_record(cfg, "openai", 24_000_000)
    current = {**cfg, "max_pixels_per_page": 24_000_000, "model_type": "openai"}
    assert image_settings_fingerprint(record) == image_settings_fingerprint(current)
    changed = image_settings_record({**cfg, "payload_format": "png"}, "openai")
    assert image_settings_fingerprint(changed) != image_settings_fingerprint(record)


def test_a_stored_item_resumes_with_its_recorded_image_values() -> None:
    model = "claude-sonnet-5-5"
    earlier = settings(model, "anthropic", grayscale=True, jpeg_quality=85)
    stored = image_settings_record(earlier, "anthropic")

    kept = resume_images(ImageSpec(), stored)
    current = image_settings_record(
        settings(model, "anthropic", **_values(kept)), "anthropic"
    )
    fresh = image_settings_record(settings(model, "anthropic"), "anthropic")

    assert (kept.grayscale, kept.jpeg_quality) == (True, 85)
    assert image_changes(stored, current) == []
    assert image_changes(stored, fresh)
    assert resume_images(ImageSpec(grayscale=False), stored).grayscale is False
    assert resume_images(ImageSpec(), None) == ImageSpec()


def _values(images: ImageSpec) -> dict[str, Any]:
    return {
        "dpi": images.dpi,
        "format": images.format,
        "jpeg_quality": images.jpeg_quality,
        "grayscale": images.grayscale,
    }


def test_the_file_record_fingerprints_the_settings(tmp_path: Path) -> None:
    path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(path)
    cfg, model_type, _section = resolve_image_settings("openai", "gpt-5.6-luna")

    record = file_provenance(path, cfg, model_type)

    assert record["source_file"] == "page.png"
    assert record["image_config"] == cfg
    assert (record["model_type"], record["detail"]) == ("openai", "original")
    assert record["image_settings_fingerprint"] == image_settings_fingerprint(
        {**cfg, "model_type": "openai"}
    )
    assert "resampling_algorithm" not in record["image_config"]
