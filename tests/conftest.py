from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope="session")
def _ensure_repo_on_syspath(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@pytest.fixture()
def tmp_config_dir(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    _write_yaml(
        config_dir / "paths_config.yaml",
        {
            "general": {
                "interactive_mode": False,
                "retain_temporary_jsonl": True,
                "input_paths_is_output_path": False,
                "logs_dir": "logs",
                "allow_relative_paths": True,
                "base_directory": str(tmp_path),
            },
            "schemas_paths": {
                "TestSchema": {
                    "input": "input",
                    "output": "output",
                    "csv_output": False,
                    "docx_output": False,
                    "txt_output": False,
                }
            },
        },
    )

    _write_yaml(
        config_dir / "model_config.yaml",
        {
            "transcription_model": {
                "name": "gpt-4o",
                "max_output_tokens": 256,
                "temperature": 0.0,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            }
        },
    )

    _write_yaml(
        config_dir / "concurrency_config.yaml",
        {
            "concurrency": {
                "extraction": {
                    "concurrency_limit": 1,
                    "delay_between_tasks": 0.0,
                    "retry": {
                        "attempts": 1,
                        "wait_min_seconds": 0.0,
                        "wait_max_seconds": 0.0,
                        "jitter_max_seconds": 0.0,
                    },
                    "timeouts": {"total": 1.0},
                }
            },
            "daily_token_limit": {"enabled": False, "daily_tokens": 1000000},
        },
    )

    _write_yaml(
        config_dir / "chunking_and_context.yaml",
        {
            "chunking": {"default_tokens_per_chunk": 10},
            "context": {},
            "matching": {},
            "retry": {},
        },
    )

    return config_dir


@pytest.fixture()
def config_loader(tmp_config_dir: Path):
    from modules.config.loader import ConfigLoader

    loader = ConfigLoader(config_dir=tmp_config_dir)
    loader.load_configs()
    return loader


@pytest.fixture(autouse=True)
def _patch_global_config_cache(config_loader):
    import modules.config.loader as loader_module

    loader_module._config_cache = config_loader
    yield
    loader_module._config_cache = None


@pytest.fixture(autouse=True)
def _reset_token_tracker(tmp_path: Path):
    import modules.core.token_tracker as token_tracker

    token_tracker._tracker_instance = None
    token_tracker._TOKEN_TRACKER_FILE = tmp_path / "token_state.json"
    yield
    token_tracker._tracker_instance = None
