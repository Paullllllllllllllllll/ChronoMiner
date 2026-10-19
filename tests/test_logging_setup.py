"""Logging is configured by the front end, never at import."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from chronominer.cli.command import LOGGER_NAME, setup_logging
from tests.conftest import PROJECT_ROOT, make_settings

_IMPORT_PROBE = """
import importlib, json, logging, pkgutil
import chronominer
names = ["chronominer.__main__"]
names += [
    info.name
    for info in pkgutil.walk_packages(
        chronominer.__path__, "chronominer."
    )
]
for name in names:
    importlib.import_module(name)
print(json.dumps({
    "imported": len(names),
    "tool": [repr(h) for h in logging.getLogger("chronominer").handlers],
    "root": [repr(h) for h in logging.getLogger().handlers],
}))
"""


def _file_handlers() -> list[logging.FileHandler]:
    return [
        handler
        for handler in logging.getLogger(LOGGER_NAME).handlers
        if isinstance(handler, logging.FileHandler)
    ]


def test_importing_every_module_configures_nothing(tmp_path: Path) -> None:
    repo_logs_existed = (PROJECT_ROOT / "logs").exists()
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        cwd=tmp_path,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["imported"] > 5
    assert report["tool"] == []
    assert report["root"] == []
    assert not (tmp_path / "logs").exists()
    assert (PROJECT_ROOT / "logs").exists() == repo_logs_existed


@pytest.mark.parametrize(
    ("logs_dir", "expected"),
    [("run", "run/application.log"), ("run/custom.log", "run/custom.log")],
)
def test_logs_dir_names_the_folder_or_the_file(
    tmp_path: Path, logs_dir: str, expected: str
) -> None:
    settings = make_settings(logs_dir=tmp_path / logs_dir)
    assert settings.log_file() == tmp_path / expected


def test_setup_logging_writes_module_records_to_the_settings_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings(logs_dir=tmp_path / "run")
    setup_logging(settings)

    logging.getLogger("chronominer.tests.probe").warning("probe message")
    for handler in _file_handlers():
        handler.flush()

    text = (tmp_path / "run" / "application.log").read_text(encoding="utf-8")
    assert "WARNING - chronominer.tests.probe - probe message" in text
    assert capsys.readouterr().err == ""


def test_without_settings_no_file_is_written(tmp_path: Path) -> None:
    setup_logging()
    logging.getLogger("chronominer.tests.probe").warning("probe")
    assert _file_handlers() == []
    assert not (tmp_path / "logs").exists()


def test_a_second_setup_replaces_the_handlers(tmp_path: Path) -> None:
    setup_logging(make_settings(logs_dir=tmp_path / "a"))
    setup_logging(make_settings(logs_dir=tmp_path / "b"))

    (handler,) = _file_handlers()
    assert Path(handler.baseFilename) == tmp_path / "b" / "application.log"
