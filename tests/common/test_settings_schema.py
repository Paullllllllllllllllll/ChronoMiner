"""The shared settings blocks keep a tool's defaults and accepted keys."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from chronominer.common.settings_loader import SettingsError
from chronominer.common.settings_schema import (
    Concurrency,
    PromptCaching,
    Reader,
    RetrySettings,
    ValueKind,
    check_blocks,
    log_file,
    read_concurrency,
    read_prompt_caching,
    read_rate_limits,
    read_retry,
    read_state_dir,
    state_directory,
)


@dataclass(frozen=True)
class _Workers(Concurrency):
    audio: int = 2


_WORKER_KEYS: tuple[tuple[str, ValueKind], ...] = (
    ("image_processing", "positive"),
    ("audio", "positive"),
    ("delay_between_tasks", "number"),
)


@dataclass(frozen=True)
class _Caching(PromptCaching):
    enabled: bool = False


def _reader(tmp_path: Path) -> Reader:
    return Reader("settings.yaml", tmp_path)


def test_a_subclass_keeps_its_type_and_defaults(tmp_path: Path) -> None:
    reader = _reader(tmp_path)
    caching = read_prompt_caching(reader, None, _Caching())
    assert caching == _Caching()
    workers = read_concurrency(reader, {"audio": 5}, _Workers(), _WORKER_KEYS)
    assert workers == _Workers(audio=5)


def test_keys_a_tool_does_not_accept_stay_errors(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="unknown keys audio"):
        read_concurrency(_reader(tmp_path), {"audio": 5}, Concurrency())


def test_flat_keys_are_read_in_order(tmp_path: Path) -> None:
    raw = {"audio": 0, "delay_between_tasks": -1}
    with pytest.raises(SettingsError, match=r"concurrency\.audio"):
        read_concurrency(_reader(tmp_path), raw, _Workers(), _WORKER_KEYS)


def test_retry_block(tmp_path: Path) -> None:
    raw = {"max_attempts": 4, "page_timeout": "off"}
    retry = read_retry(_reader(tmp_path), raw, RetrySettings())
    assert retry == RetrySettings(max_attempts=4, page_timeout=0.0)
    assert retry.page_deadline(900.0) is None
    assert RetrySettings().page_deadline(900.0) == 900.0 * 3 + 300.0


@pytest.mark.parametrize("raw", [None, {}, {"max_attempts": 4}, {"page_timeout": None}])
@pytest.mark.parametrize("deadline", [60.0, 0.0])
def test_retry_keeps_an_injected_deadline(
    tmp_path: Path, raw: dict[str, object] | None, deadline: float
) -> None:
    base = RetrySettings(page_timeout=deadline)
    assert read_retry(_reader(tmp_path), raw, base).page_timeout == deadline


def test_rate_limits_take_the_injected_default(tmp_path: Path) -> None:
    default = ((1, 1),)
    assert read_rate_limits(_reader(tmp_path), None, default) == default
    with pytest.raises(SettingsError, match="rate_limits"):
        read_rate_limits(_reader(tmp_path), [[1, 0]], default)


def test_state_and_log_locations(tmp_path: Path) -> None:
    reader = _reader(tmp_path)
    assert read_state_dir(reader, {"state_dir": ""}) is None
    assert read_state_dir(reader, {"state_dir": "s"}) == Path("s")
    assert state_directory(None, ".tool") == Path.home() / ".tool"
    assert log_file(None, tmp_path) == tmp_path / "application.log"
    assert log_file(tmp_path / "x.log", tmp_path) == tmp_path / "x.log"


def test_blocks_may_not_reuse_builtin_keys() -> None:
    with pytest.raises(ValueError, match="retry"):
        check_blocks(frozenset({"retry"}), {"retry": dict})
