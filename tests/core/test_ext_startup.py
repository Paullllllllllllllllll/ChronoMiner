"""The extension seam as the CLI calls it after planning.

``ext.startup`` receives the spec, the settings and the parsed arguments,
dry runs included; a SettingsError it raises is a configuration error
(exit 2, one JSON line). The tests need the extract CLI and are skipped
until it exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from chronominer import ext
from chronominer.settings import Settings, SettingsError
from chronominer.spec import RunSpec

extract_cli = pytest.importorskip("chronominer.cli.extract")
entry = pytest.importorskip("chronominer.__main__")


class _Startup:
    """Stand-in for ``ext.startup`` that records its calls and may raise."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[RunSpec, Settings, argparse.Namespace]] = []

    def __call__(
        self, spec: RunSpec, *, settings: Settings, args: argparse.Namespace
    ) -> None:
        self.calls.append((spec, settings, args))
        if self.error is not None:
            raise self.error


@pytest.fixture
def text_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a text input; the run's key variable is set to a dummy value."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    path = tmp_path / "wages.txt"
    path.write_text("Johann Meier, mason, 12 days, 6 fl.\n", encoding="utf-8")
    return path


def _install(monkeypatch: pytest.MonkeyPatch, startup: _Startup) -> _Startup:
    monkeypatch.setattr(ext, "startup", startup)
    return startup


def _no_run(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("run must not be called")

    monkeypatch.setattr(extract_cli, "run", run)


def _extract(path: Path, out: Path, *extra: str) -> list[str]:
    return [
        "extract",
        "--input",
        str(path),
        "--schema",
        "WageRecords",
        "--output",
        str(out),
        *extra,
    ]


def test_a_dry_run_passes_the_spec_the_settings_and_the_parsed_arguments(
    text_input: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    startup = _install(monkeypatch, _Startup())
    _no_run(monkeypatch)

    assert entry.main(_extract(text_input, tmp_path / "out", "--dry-run")) == 0

    ((spec, settings, args),) = startup.calls
    assert isinstance(spec, RunSpec)
    assert spec.input.path == text_input
    assert spec.run.dry_run
    assert isinstance(settings, Settings)
    assert args.command == "extract"


@pytest.mark.parametrize("dry_run", [False, True])
def test_a_settings_error_from_startup_exits_2_with_one_json_line(
    text_input: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dry_run: bool,
) -> None:
    _install(monkeypatch, _Startup(SettingsError("extension block is invalid")))
    _no_run(monkeypatch)
    out = tmp_path / "out"
    extra = ("--json", "--dry-run") if dry_run else ("--json",)

    assert entry.main(_extract(text_input, out, *extra)) == 2

    captured = capsys.readouterr()
    (line,) = captured.out.splitlines()
    assert json.loads(line)["items_complete"] == 0
    assert json.loads(line)["dry_run"] is dry_run
    assert "extension block is invalid" in captured.err
    assert not out.exists()
