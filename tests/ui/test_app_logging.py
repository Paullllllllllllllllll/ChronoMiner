"""The ui and the CLI write the log file only for runs that are not dry runs.

The ui keeps wizard log records off stderr and attaches the settings' log
file once a run that is not a dry run starts; a dry run, a copied command
and Quit write no log file. The CLI's dry runs write none either.
"""

from __future__ import annotations

import dataclasses
import io
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.console import Console

from chronominer import __main__ as entry
from chronominer.common.testing import KEEP, ScriptedPrompter
from chronominer.common.wizard import Review, State
from chronominer.settings import load
from chronominer.ui import app
from chronominer.ui.session import Session
from tests.characterization.inputs import make_text_file
from tests.output.test_store import build_store, chunk
from tests.ui.conftest import DUMMY_KEYS, SettingsWriter

MARKER = "review log record of the wizard"


def _launch(answers: list[Any], settings_path: Path) -> tuple[int, ScriptedPrompter]:
    console = Console(file=io.StringIO(), width=120, color_system=None)
    prompter = ScriptedPrompter(answers)
    with prompter:
        code = app.run_app(
            settings_path,
            prompter=prompter,
            console=console,
            require_tty=False,
            environ=dict(DUMMY_KEYS),
        )
    return code, prompter


def _file_handlers() -> list[Path]:
    return [
        Path(handler.baseFilename)
        for handler in logging.getLogger("chronominer").handlers
        if isinstance(handler, logging.FileHandler)
    ]


def _logging_review(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the render task's review log :data:`MARKER`."""
    render = app.FLOWS["render"]

    def review(state: State, session: Session) -> Review:
        logging.getLogger("chronominer.ui.store").warning(MARKER)
        return render().review(state, session)

    monkeypatch.setitem(
        app.FLOWS, "render", lambda: dataclasses.replace(render(), review=review)
    )


def _seeded(root: Path) -> Path:
    build_store(root, [chunk(0), chunk(1)])
    return root


def test_a_quit_session_keeps_records_off_stderr_and_writes_no_log(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    write_settings: SettingsWriter,
) -> None:
    _logging_review(monkeypatch)
    settings_path = write_settings()
    root = _seeded(tmp_path / "out")

    code, prompter = _launch(
        ["render", str(root), KEEP, KEEP, KEEP, "quit"], settings_path
    )

    assert code == 0, prompter.transcript
    assert MARKER not in capsys.readouterr().err
    assert _file_handlers() == []
    assert not load(settings_path).log_file().exists()


def test_a_ui_dry_run_writes_no_log_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, write_settings: SettingsWriter
) -> None:
    _logging_review(monkeypatch)
    settings_path = write_settings()
    root = _seeded(tmp_path / "out")

    code, prompter = _launch(
        ["render", str(root), KEEP, True, True, "run"], settings_path
    )

    assert code == 0, prompter.transcript
    assert _file_handlers() == []
    assert not load(settings_path).log_file().exists()


def test_a_ui_run_writes_to_the_log_file(
    tmp_path: Path, write_settings: SettingsWriter
) -> None:
    settings_path = write_settings()
    root = _seeded(tmp_path / "out")

    code, prompter = _launch(
        ["render", str(root), KEEP, KEEP, KEEP, "run"], settings_path
    )

    assert code == 0, prompter.transcript
    log_file = load(settings_path).log_file()
    assert _file_handlers() == [log_file]
    logging.getLogger("chronominer.ui.store").warning(MARKER)
    assert MARKER in log_file.read_text(encoding="utf-8")


def _cli_settings(tmp_path: Path) -> tuple[Path, Path]:
    log_file = tmp_path / "logs" / "cli.log"
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump({"logs_dir": str(log_file)}), encoding="utf-8")
    return path, log_file


@pytest.mark.parametrize(
    "command",
    [
        ["extract", "--schema", "WageRecords"],
        ["ranges", "generate"],
    ],
    ids=["extract", "ranges-generate"],
)
def test_a_cli_dry_run_writes_no_log_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: list[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings_path, log_file = _cli_settings(tmp_path)
    text = make_text_file(tmp_path / "inputs" / "ledger.txt")
    argv = [*command, "--input", str(text), "--settings", str(settings_path)]

    assert entry.main([*argv, "--dry-run"]) == 0
    assert _file_handlers() == []
    assert not log_file.exists()
    assert not log_file.parent.exists()


def test_importing_the_app_attaches_no_log_file() -> None:
    assert app.LOGGER_NAME == "chronominer"
    assert _file_handlers() == []
