from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_config_loader_resolves_relative_paths(config_loader):
    paths = config_loader.get_paths_config()
    general = paths["general"]

    assert general.get("allow_relative_paths") is True

    logs_dir = Path(general["logs_dir"])
    assert logs_dir.is_absolute()

    schema_paths = config_loader.get_schemas_paths()["TestSchema"]
    assert Path(schema_paths["input"]).is_absolute()
    assert Path(schema_paths["output"]).is_absolute()


@pytest.mark.unit
def test_get_config_loader_returns_cached_instance(config_loader):
    from modules.config.loader import get_config_loader

    # conftest patches the module-level cache to this instance
    assert get_config_loader() is config_loader


@pytest.mark.unit
def test_clear_config_cache_allows_reinit(tmp_config_dir, monkeypatch):
    import modules.config.loader as loader_module
    from modules.config.loader import ConfigLoader

    loader_module.clear_config_cache()

    def _factory():
        return ConfigLoader(config_dir=tmp_config_dir)

    monkeypatch.setattr(loader_module, "ConfigLoader", _factory)

    loader = loader_module.get_config_loader()
    assert isinstance(loader, ConfigLoader)
    assert loader.get_model_config()["transcription_model"]["name"] == "gpt-4o"
