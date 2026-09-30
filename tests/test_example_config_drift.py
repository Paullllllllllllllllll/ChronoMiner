from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from modules.images.settings import resolved_settings


@pytest.mark.unit
def test_schema_paths_template_matches_shipped_schema_names(repo_root: Path):
    """Regression: config/paths_config.example.yaml's schemas_paths keys must
    exactly match the "name" declared by each top-level schemas/*.json file.
    A stale key (e.g. a schema renamed without updating the template) makes
    the interactive wizard hard-exit when that schema is selected."""
    example_path = repo_root / "config" / "paths_config.example.yaml"
    example_config = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    template_keys = set(example_config["schemas_paths"].keys())

    schemas_dir = repo_root / "schemas"
    schema_names = set()
    for schema_file in schemas_dir.glob("*.json"):
        data = json.loads(schema_file.read_text(encoding="utf-8"))
        schema_names.add(data["name"])

    assert schema_names == template_keys, (
        f"Mismatch between shipped schema names and "
        f"paths_config.example.yaml schemas_paths keys.\n"
        f"Schemas without a template entry: {schema_names - template_keys}\n"
        f"Template entries without a shipped schema: "
        f"{template_keys - schema_names}"
    )


@pytest.mark.unit
def test_paths_config_example_general_has_relative_path_keys(repo_root: Path):
    """Regression: modules/config/loader.py reads general.allow_relative_paths
    (default False) and general.base_directory (default '.') via
    ConfigLoader._resolve_paths; the tracked template must declare both so a
    fresh clone documents the keys it silently defaults on."""
    example_path = repo_root / "config" / "paths_config.example.yaml"
    example_config = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    general = example_config["general"]

    assert "allow_relative_paths" in general
    assert "base_directory" in general


def test_image_example_config_drift() -> None:
    path = Path(__file__).parents[1] / "config/image_processing_config.example.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    providers = ["api", "anthropic", "google", "custom"]
    assert list(config)[:6] == [
        "render_strategy",
        "max_pixels_per_page",
        *(f"{name}_image_processing" for name in providers),
    ]
    common = [
        "target_dpi",
        "native_fallback_dpi",
        "payload_format",
        "max_image_bytes",
        "grayscale_conversion",
        "handle_transparency",
        "jpeg_quality",
    ]
    for name in providers:
        section = f"{name}_image_processing"
        detail = (
            ["media_resolution"]
            if name == "google"
            else ([] if name == "anthropic" else ["llm_detail"])
        )
        assert list(config[section]) == common + detail + [
            "resize_profile",
            "low_max_side_px",
            "high_target_box",
        ]
        provider = "openai" if name == "api" else name
        model = "claude-opus-5" if name == "anthropic" else "gpt-6-astra"
        resolved_settings(
            config[section],
            provider,
            model,
            provider,
            section,
            config["max_pixels_per_page"],
            config["render_strategy"],
        )
    assert [config[f"{name}_image_processing"]["target_dpi"] for name in providers] == [
        "native",
        "native",
        300,
        150,
    ]
    assert [
        config[f"{name}_image_processing"]["jpeg_quality"] for name in providers
    ] == [
        95,
        95,
        95,
        85,
    ]

    assert not config["api_image_processing"]["grayscale_conversion"]
    custom = config["custom_image_processing"]
    assert custom["resize_profile"] == "low"
    assert custom["low_max_side_px"] == 768
    assert custom["high_target_box"] == [512, 1024]
