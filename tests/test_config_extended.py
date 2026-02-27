"""Extended tests for modules/config/loader.py and modules/config/manager.py.

Covers uncovered paths in ConfigLoader and ConfigManager:
- ConfigManager: validate_paths with non-absolute paths, load_developer_message,
  get_schemas_paths, get_validation_errors
- ConfigLoader: _ensure_image_support, _resolve_paths with relative paths,
  error paths in _load_yaml, clear_config_cache
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from modules.config.loader import ConfigLoader, clear_config_cache, get_config_loader
from modules.config.manager import ConfigManager, ConfigValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# ConfigManager — validate_paths
# ---------------------------------------------------------------------------

class TestConfigManagerValidatePaths:
    def test_passes_with_relative_paths_allowed(self, config_loader):
        manager = ConfigManager(config_loader)
        paths_config = {
            "general": {"allow_relative_paths": True},
            "schemas_paths": {"S": {"input": "relative/path", "output": "relative/out"}},
        }
        assert manager.validate_paths(paths_config) is True

    def test_fails_with_relative_logs_dir(self, config_loader):
        manager = ConfigManager(config_loader)
        paths_config = {
            "general": {"allow_relative_paths": False, "logs_dir": "relative/logs"},
            "schemas_paths": {},
        }
        with pytest.raises(ConfigValidationError, match="logs_dir"):
            manager.validate_paths(paths_config, raise_on_error=True)

    def test_fails_with_relative_schema_input(self, config_loader):
        manager = ConfigManager(config_loader)
        paths_config = {
            "general": {"allow_relative_paths": False},
            "schemas_paths": {"MySchema": {"input": "relative/input", "output": "/abs/output"}},
        }
        with pytest.raises(ConfigValidationError, match="input path"):
            manager.validate_paths(paths_config, raise_on_error=True)

    def test_fails_with_relative_schema_output(self, config_loader):
        manager = ConfigManager(config_loader)
        paths_config = {
            "general": {"allow_relative_paths": False},
            "schemas_paths": {"MySchema": {"input": "/abs/input", "output": "relative/output"}},
        }
        with pytest.raises(ConfigValidationError, match="output path"):
            manager.validate_paths(paths_config, raise_on_error=True)

    def test_returns_false_without_raise(self, config_loader):
        manager = ConfigManager(config_loader)
        paths_config = {
            "general": {"allow_relative_paths": False, "logs_dir": "relative/logs"},
            "schemas_paths": {},
        }
        assert manager.validate_paths(paths_config, raise_on_error=False) is False

    def test_passes_with_absolute_paths(self, config_loader):
        import sys
        manager = ConfigManager(config_loader)
        # On Windows, absolute paths need a drive letter
        if sys.platform == "win32":
            logs = "C:\\abs\\logs"
            inp = "C:\\abs\\in"
            out = "C:\\abs\\out"
        else:
            logs = "/abs/logs"
            inp = "/abs/in"
            out = "/abs/out"
        paths_config = {
            "general": {"allow_relative_paths": False, "logs_dir": logs},
            "schemas_paths": {"S": {"input": inp, "output": out}},
        }
        assert manager.validate_paths(paths_config) is True

    def test_get_validation_errors(self, config_loader):
        manager = ConfigManager(config_loader)
        paths_config = {
            "general": {"allow_relative_paths": False, "logs_dir": "rel"},
            "schemas_paths": {"S": {"input": "rel/in", "output": "rel/out"}},
        }
        manager.validate_paths(paths_config, raise_on_error=False)
        errors = manager.get_validation_errors()
        assert len(errors) == 3  # logs_dir + input + output


# ---------------------------------------------------------------------------
# ConfigManager — load_developer_message
# ---------------------------------------------------------------------------

class TestConfigManagerLoadDeveloperMessage:
    def test_file_not_found_raises(self, config_loader, tmp_path, monkeypatch):
        manager = ConfigManager(config_loader)
        monkeypatch.chdir(tmp_path)

        with pytest.raises(FileNotFoundError):
            manager.load_developer_message("nonexistent_schema", raise_on_error=True)

    def test_file_not_found_returns_none(self, config_loader, tmp_path, monkeypatch):
        manager = ConfigManager(config_loader)
        monkeypatch.chdir(tmp_path)

        result = manager.load_developer_message("nonexistent_schema", raise_on_error=False)
        assert result is None

    def test_existing_file(self, config_loader, tmp_path, monkeypatch):
        manager = ConfigManager(config_loader)
        monkeypatch.chdir(tmp_path)
        dev_dir = tmp_path / "developer_messages"
        dev_dir.mkdir()
        (dev_dir / "TestSchema.txt").write_text("Test dev message", encoding="utf-8")

        result = manager.load_developer_message("TestSchema")
        assert result == "Test dev message"

    def test_read_error_raises_ioerror(self, config_loader, tmp_path, monkeypatch):
        manager = ConfigManager(config_loader)
        monkeypatch.chdir(tmp_path)
        dev_dir = tmp_path / "developer_messages"
        dev_dir.mkdir()
        msg_file = dev_dir / "BadSchema.txt"
        msg_file.write_text("content", encoding="utf-8")

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "open", side_effect=PermissionError("denied")):
                with pytest.raises(IOError, match="Failed to read"):
                    manager.load_developer_message("BadSchema", raise_on_error=True)

    def test_read_error_returns_none(self, config_loader, tmp_path, monkeypatch):
        manager = ConfigManager(config_loader)
        monkeypatch.chdir(tmp_path)
        dev_dir = tmp_path / "developer_messages"
        dev_dir.mkdir()
        msg_file = dev_dir / "BadSchema.txt"
        msg_file.write_text("content", encoding="utf-8")

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "open", side_effect=PermissionError("denied")):
                result = manager.load_developer_message("BadSchema", raise_on_error=False)
                assert result is None


# ---------------------------------------------------------------------------
# ConfigManager — get_schemas_paths
# ---------------------------------------------------------------------------

class TestConfigManagerGetSchemasPaths:
    def test_returns_schemas_paths(self, config_loader):
        manager = ConfigManager(config_loader)
        paths = manager.get_schemas_paths()
        assert isinstance(paths, dict)
        assert "TestSchema" in paths


# ---------------------------------------------------------------------------
# ConfigLoader — _ensure_image_support
# ---------------------------------------------------------------------------

class TestEnsureImageSupport:
    def test_no_image_support_raises(self, tmp_config_dir):
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load_configs()

        # o3-mini does not support images
        with pytest.raises(ValueError, match="does not support image inputs"):
            loader._ensure_image_support("o3-mini", expects_images=True)

    def test_image_support_passes(self, tmp_config_dir):
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load_configs()

        # gpt-4o supports images
        loader._ensure_image_support("gpt-4o", expects_images=True)

    def test_expects_false_is_noop(self, tmp_config_dir):
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load_configs()

        loader._ensure_image_support("o3-mini", expects_images=False)


# ---------------------------------------------------------------------------
# ConfigLoader — _load_yaml error
# ---------------------------------------------------------------------------

class TestLoadYamlErrors:
    def test_missing_file_raises(self, tmp_config_dir):
        loader = ConfigLoader(config_dir=tmp_config_dir)
        (tmp_config_dir / "model_config.yaml").unlink()

        with pytest.raises(FileNotFoundError):
            loader._load_yaml("model_config.yaml")

    def test_invalid_yaml_raises(self, tmp_config_dir):
        (tmp_config_dir / "model_config.yaml").write_text(
            "invalid: yaml: [unclosed", encoding="utf-8"
        )
        loader = ConfigLoader(config_dir=tmp_config_dir)

        with pytest.raises(yaml.YAMLError):
            loader._load_yaml("model_config.yaml")


# ---------------------------------------------------------------------------
# ConfigLoader — _validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def test_missing_key_raises(self, tmp_config_dir):
        loader = ConfigLoader(config_dir=tmp_config_dir)
        with pytest.raises(KeyError, match="Missing 'required_key'"):
            loader._validate_config({}, ["required_key"], "test_config.yaml")

    def test_valid_config_passes(self, tmp_config_dir):
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader._validate_config({"key": "value"}, ["key"], "test_config.yaml")


# ---------------------------------------------------------------------------
# ConfigLoader — _resolve_paths
# ---------------------------------------------------------------------------

class TestResolvePaths:
    def test_skip_when_not_allowed(self, tmp_config_dir):
        loader = ConfigLoader(config_dir=tmp_config_dir)
        config = {
            "general": {"allow_relative_paths": False},
            "schemas_paths": {"S": {"input": "relative/path"}},
        }
        loader._resolve_paths(config)
        # Path should remain unchanged
        assert config["schemas_paths"]["S"]["input"] == "relative/path"

    def test_resolves_relative_logs_dir(self, tmp_config_dir, tmp_path):
        loader = ConfigLoader(config_dir=tmp_config_dir)
        config = {
            "general": {
                "allow_relative_paths": True,
                "base_directory": str(tmp_path),
                "logs_dir": "logs",
            },
            "schemas_paths": {},
        }
        loader._resolve_paths(config)
        resolved = Path(config["general"]["logs_dir"])
        assert resolved.is_absolute()

    def test_resolves_schema_paths(self, tmp_config_dir, tmp_path):
        loader = ConfigLoader(config_dir=tmp_config_dir)
        config = {
            "general": {
                "allow_relative_paths": True,
                "base_directory": str(tmp_path),
            },
            "schemas_paths": {
                "S": {"input": "data/in", "output": "data/out"},
            },
        }
        loader._resolve_paths(config)
        assert Path(config["schemas_paths"]["S"]["input"]).is_absolute()
        assert Path(config["schemas_paths"]["S"]["output"]).is_absolute()

    def test_nonexistent_base_uses_cwd(self, tmp_config_dir):
        loader = ConfigLoader(config_dir=tmp_config_dir)
        config = {
            "general": {
                "allow_relative_paths": True,
                "base_directory": "/nonexistent/path/xyz",
                "logs_dir": "logs",
            },
            "schemas_paths": {},
        }
        loader._resolve_paths(config)
        resolved = Path(config["general"]["logs_dir"])
        assert resolved.is_absolute()


# ---------------------------------------------------------------------------
# clear_config_cache
# ---------------------------------------------------------------------------

class TestClearConfigCache:
    def test_clears_cache(self):
        import modules.config.loader as loader_module
        # The conftest fixture sets _config_cache, so it should be set
        assert loader_module._config_cache is not None
        clear_config_cache()
        assert loader_module._config_cache is None


# ---------------------------------------------------------------------------
# ConfigLoader — validate_model_config with image support
# ---------------------------------------------------------------------------

class TestValidateModelConfigImageSupport:
    def test_image_config_validated(self, tmp_config_dir):
        # Write config with expects_image_inputs
        _write_yaml(
            tmp_config_dir / "model_config.yaml",
            {
                "transcription_model": {
                    "name": "gpt-4o",
                    "expects_image_inputs": True,
                    "max_output_tokens": 256,
                    "temperature": 0.0,
                }
            },
        )
        loader = ConfigLoader(config_dir=tmp_config_dir)
        loader.load_configs()  # Should not raise — gpt-4o supports images

    def test_image_config_invalid_model(self, tmp_config_dir):
        _write_yaml(
            tmp_config_dir / "model_config.yaml",
            {
                "transcription_model": {
                    "name": "o3-mini",
                    "expects_image_inputs": True,
                    "max_output_tokens": 256,
                    "temperature": 0.0,
                }
            },
        )
        loader = ConfigLoader(config_dir=tmp_config_dir)
        with pytest.raises(ValueError, match="does not support image inputs"):
            loader.load_configs()
