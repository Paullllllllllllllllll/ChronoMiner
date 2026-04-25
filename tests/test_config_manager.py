from __future__ import annotations

import pytest

from modules.config.manager import ConfigManager, ConfigValidationError


@pytest.mark.unit
def test_validate_paths_skips_when_allow_relative_paths_true(config_loader):
    mgr = ConfigManager(config_loader)
    assert mgr.validate_paths(config_loader.get_paths_config()) is True


@pytest.mark.unit
def test_validate_paths_raises_for_relative_when_not_allowed(config_loader):
    mgr = ConfigManager(config_loader)

    bad = {
        "general": {
            "allow_relative_paths": False,
            "logs_dir": "relative/logs",
        },
        "schemas_paths": {
            "X": {"input": "relative/input", "output": "relative/output"}
        },
    }

    with pytest.raises(ConfigValidationError):
        mgr.validate_paths(bad, raise_on_error=True)
