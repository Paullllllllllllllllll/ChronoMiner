"""The ui subcommand: terminal check, settings, task switches, decisions, Ctrl+C."""

from __future__ import annotations

import argparse
import dataclasses
import io
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import chronominer.run as core
from chronominer.common.testing import KEEP, ScriptedPrompter
from chronominer.common.wizard import BACK, Review
from chronominer.ui import app
from chronominer.ui.session import TASKS, RunRequest, TaskFlow
from tests.ui.conftest import DUMMY_KEYS, SettingsWriter

INTERRUPT = object()


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


class _Interrupting(ScriptedPrompter):
    """Raise KeyboardInterrupt where the script holds ``INTERRUPT``."""

    def _take(self, message: str) -> Any:
        answer = super()._take(message)
        if answer is INTERRUPT:
            raise KeyboardInterrupt
        return answer


@pytest.fixture
def settings_path(write_settings: SettingsWriter) -> Path:
    """A settings file without defaults."""
    return write_settings()


def _render(folder: Path, *review: Any) -> list[Any]:
    """Answers of a render task on *folder*, then the *review* answers."""
    return [
        "render",
        str(folder),
        KEEP,  # formats
        KEEP,  # force
        KEEP,  # dry run
        *review,
    ]


def _console() -> Console:
    return Console(file=io.StringIO(), width=120, color_system=None)


def _text(console: Console) -> str:
    file = console.file
    assert isinstance(file, io.StringIO)
    return file.getvalue()


def _launch(
    answers: Sequence[Any],
    settings_path: Path | None,
    *,
    prompter: ScriptedPrompter | None = None,
    **kwargs: Any,
) -> tuple[int, Console, ScriptedPrompter]:
    console = _console()
    scripted = prompter if prompter is not None else ScriptedPrompter(answers)
    with scripted:
        code = app.run_app(
            settings_path,
            prompter=scripted,
            console=console,
            require_tty=False,
            environ=dict(DUMMY_KEYS),
            **kwargs,
        )
    return code, console, scripted


def _no_run(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run(*args: Any, **kwargs: Any) -> core.RunResult:
        raise AssertionError("core.run must not be called")

    monkeypatch.setattr(core, "run", run)


def test_every_task_has_a_flow_in_menu_order() -> None:
    assert set(app.FLOWS) == set(TASKS)
    assert list(app.FLOWS) == list(TASKS)
    assert app.FIRST_TASK == "extract"
    for name, make in app.FLOWS.items():
        flow = make()
        assert (flow.name, flow.title) == (name, TASKS[name])


@pytest.mark.parametrize("stream", ["stdin", "stdout"])
def test_a_non_interactive_stream_is_refused_before_reading_settings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    settings_path: Path,
    stream: str,
) -> None:
    loads: list[Any] = []
    monkeypatch.setattr(app, "load", lambda *a, **kw: loads.append(a))
    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(sys, "stdout", _Tty())
    monkeypatch.setattr(sys, stream, io.StringIO())
    code = app.execute(argparse.Namespace(command="ui", settings=settings_path))
    assert code == 2
    assert loads == []
    err = capsys.readouterr().err
    assert "'chronominer extract'" in err
    assert "chronominer extract --help" in err


def test_a_settings_error_exits_2(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    code, console, prompter = _launch([], missing)
    assert code == 2
    assert f"Settings file not found: {missing}" in _text(console)
    assert prompter.lines == []


def test_back_at_the_task_step_and_quit_exits_0(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_run(monkeypatch)
    code, _console, prompter = _launch([BACK, "quit"], settings_path)
    assert code == 0
    assert "ChronoMiner | new extraction   step 1" in prompter.transcript
    assert "? Quit without running?" in prompter.lines


def _headers(prompter: ScriptedPrompter) -> list[str]:
    return [line for line in prompter.lines if line.startswith("ChronoMiner | ")]


@pytest.mark.parametrize("task", [name for name in TASKS if name != app.FIRST_TASK])
def test_a_task_switch_starts_at_the_first_step_of_the_new_task(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch, task: str
) -> None:
    _no_run(monkeypatch)
    code, _console, prompter = _launch([task, BACK, BACK, "quit"], settings_path)
    assert code == 0
    title = TASKS[task]
    headers = _headers(prompter)
    assert headers[0] == "ChronoMiner | new extraction   step 1"
    assert headers[1] == f"ChronoMiner | {title}   step 2"
    assert headers[2] == f"ChronoMiner | {title}   step 1"
    asked = [index for index, line in enumerate(prompter.lines) if line == "? Task:"]
    assert len(asked) == 2
    assert prompter.lines[asked[1] + 1 + list(TASKS).index(task)] == f" > {title}"
    assert "? Quit without running?" in prompter.lines


def test_edit_a_step_on_the_review_switches_the_task(
    settings_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_run(monkeypatch)
    answers = _render(tmp_path, "edit", "task", "drop")
    code, _console, prompter = _launch([*answers, BACK, BACK, "quit"], settings_path)
    assert code == 0
    assert "Edit which step?" in prompter.transcript
    assert _headers(prompter)[-2] == f"ChronoMiner | {TASKS['drop']}   step 2"


def test_quit_on_the_review_exits_0(
    settings_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_run(monkeypatch)
    folder = tmp_path / "out"
    folder.mkdir()
    code, _console, _prompter = _launch(_render(folder, "quit"), settings_path)
    assert code == 0
    assert list(folder.iterdir()) == []


def test_copy_prints_the_command_and_copies_it(
    settings_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_run(monkeypatch)
    copied: list[tuple[list[str], bytes]] = []

    def clipboard(argv: Sequence[str], data: bytes) -> None:
        copied.append((list(argv), data))

    stdout = io.StringIO()
    folder = tmp_path / "out"
    folder.mkdir()
    code, console, _prompter = _launch(
        _render(folder, "copy"),
        settings_path,
        stdout=stdout,
        clipboard=clipboard,
    )
    assert code == 0
    lines = stdout.getvalue().splitlines()
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("chronominer render --input ")
    assert str(folder) in line
    if sys.platform in ("win32", "darwin"):
        assert len(copied) == 1
        encoding = "utf-16" if sys.platform == "win32" else "utf-8"
        assert copied[0][1].decode(encoding) == line
        assert "Copied the command to the clipboard." in _text(console)
    assert list(folder.iterdir()) == []


def _stub(flow: TaskFlow, run: Callable[[RunRequest], int]) -> TaskFlow:
    """Return *flow* without steps, with an empty review and *run*."""
    return dataclasses.replace(
        flow,
        steps=lambda _session: [],
        review=lambda _state, _session: Review(()),
        command=lambda _state, _session: [*flow.program],
        run=run,
    )


def test_the_run_receives_the_ui_arguments(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[RunRequest] = []

    def record(request: RunRequest) -> int:
        requests.append(request)
        return 1

    check_flow = app.FLOWS["batch-check"]
    monkeypatch.setitem(app.FLOWS, "batch-check", lambda: _stub(check_flow(), record))
    args = argparse.Namespace(command="ui", settings=settings_path)
    code, console, _prompter = _launch(["batch-check", "run"], settings_path, args=args)
    assert code == 1
    assert len(requests) == 1
    assert requests[0].args is args
    assert requests[0].console is console
    assert requests[0].session.settings_path == settings_path
    assert requests[0].session.args is args


def test_ctrl_c_in_the_wizard_exits_130(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_run(monkeypatch)
    prompter = _Interrupting([KEEP, INTERRUPT])
    code, console, _prompter = _launch([], settings_path, prompter=prompter)
    assert code == 130
    assert "Interrupted by the user." in _text(console)


def _stub_first_task(
    monkeypatch: pytest.MonkeyPatch, run: Callable[[RunRequest], int]
) -> None:
    first = app.FLOWS[app.FIRST_TASK]
    monkeypatch.setitem(app.FLOWS, app.FIRST_TASK, lambda: _stub(first(), run))


def test_ctrl_c_during_the_run_exits_130(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(_request: RunRequest) -> int:
        raise KeyboardInterrupt

    _stub_first_task(monkeypatch, interrupted)
    code, console, _prompter = _launch([KEEP, "run"], settings_path)
    assert code == 130
    assert "Interrupted by the user." in _text(console)


def test_an_unexpected_error_is_logged_and_exits_1(
    settings_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(_request: RunRequest) -> int:
        raise RuntimeError("boom")

    _stub_first_task(monkeypatch, broken)
    with caplog.at_level("ERROR", logger=app.__name__):
        code, console, _prompter = _launch([KEEP, "run"], settings_path)
    assert code == 1
    assert "Unexpected error: boom" in _text(console)
    assert "Critical error in the guided run: boom" in caplog.text
