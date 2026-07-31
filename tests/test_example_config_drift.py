from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


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
