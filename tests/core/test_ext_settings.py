"""Extra settings blocks: the extension hook and the loader."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chronominer import ext
from chronominer.settings import Settings, SettingsError, load

BLOCK = "extra_block: {limit: 3, label: alpha}\n"


def _parse(value: Any) -> tuple[int, str]:
    if not isinstance(value, dict):
        raise TypeError("must be a mapping")
    return int(value["limit"]), str(value["label"])


def _reject(value: Any) -> Any:
    raise ValueError("limit must be positive")


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _patch_blocks(
    monkeypatch: pytest.MonkeyPatch, parser: Callable[[Any], Any]
) -> None:
    monkeypatch.setattr(ext, "settings_blocks", lambda: {"extra_block": parser})


def test_a_registered_block_is_parsed_into_the_extensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_blocks(monkeypatch, _parse)
    path = _write(tmp_path, BLOCK + "concurrency:\n  image_processing: 5\n")

    settings = load(path, blocks=ext.settings_blocks())

    assert settings.extensions["extra_block"] == (3, "alpha")
    assert settings.concurrency.image_processing == 5


def test_an_absent_block_is_not_parsed(tmp_path: Path) -> None:
    calls: list[Any] = []
    path = _write(tmp_path, "concurrency:\n  image_processing: 5\n")

    settings = load(path, blocks={"extra_block": calls.append})

    assert dict(settings.extensions) == {}
    assert calls == []


def test_without_the_hook_the_block_is_an_unknown_key(tmp_path: Path) -> None:
    path = _write(tmp_path, BLOCK)

    with pytest.raises(SettingsError, match="extra_block"):
        load(path)


def test_other_unknown_keys_stay_errors(tmp_path: Path) -> None:
    path = _write(tmp_path, BLOCK + "surprise: 1\n")

    with pytest.raises(SettingsError, match="surprise"):
        load(path, blocks={"extra_block": _parse})


@pytest.mark.parametrize("parser", [_reject, _parse], ids=["raises", "wrong_type"])
def test_a_parser_error_names_the_key(
    tmp_path: Path, parser: Callable[[Any], Any]
) -> None:
    path = _write(tmp_path, "extra_block: [1, 2]\n")

    with pytest.raises(SettingsError, match=r"settings\.yaml: extra_block: "):
        load(path, blocks={"extra_block": parser})


def test_a_block_may_not_reuse_a_built_in_key() -> None:
    with pytest.raises(ValueError, match="retry"):
        Settings.from_mapping({}, blocks={"retry": _parse})
