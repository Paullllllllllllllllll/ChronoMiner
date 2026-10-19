"""Help of the console script and of every subcommand."""

from __future__ import annotations

import re

import pytest

from chronominer import __main__ as entry

WIDTHS = (80, 100)

COMMANDS: tuple[tuple[str, ...], ...] = (
    ("extract",),
    ("ranges", "generate"),
    ("ranges", "adjust"),
    ("batch", "check"),
    ("batch", "cancel"),
    ("repair",),
    ("render",),
    ("drop",),
    ("ui",),
)


def _help(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    with pytest.raises(SystemExit) as exc:
        entry.main([*argv, "--help"])
    assert exc.value.code == 0
    return capsys.readouterr().out


@pytest.fixture(params=WIDTHS, ids=[f"columns{w}" for w in WIDTHS])
def columns(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> int:
    width: int = request.param
    monkeypatch.setenv("COLUMNS", str(width))
    return width


def test_the_bare_command_prints_help_and_exits_0(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert entry.main([]) == 0
    bare = capsys.readouterr().out
    assert "usage: chronominer" in bare
    names = ("extract", "ranges", "batch", "repair", "render", "drop", "ui")
    for name in names:
        assert name in bare
    assert "guided run in an interactive terminal" in bare
    assert _help(capsys) == bare


def test_ui_help_lists_settings_and_fits_the_width(
    columns: int, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _help(capsys, "ui")
    assert "--settings" in set(re.findall(r"--[\w-]+", out))
    assert "--input" not in out
    for line in out.splitlines():
        assert len(line) <= columns, line


@pytest.mark.parametrize("command", COMMANDS, ids=" ".join)
def test_every_subcommand_has_help_that_fits_the_width(
    command: tuple[str, ...], columns: int, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _help(capsys, *command)
    assert out.startswith(f"usage: chronominer {' '.join(command)}")
    for line in out.splitlines():
        assert len(line) <= columns, line


def test_extract_help_never_splits_a_flag(
    columns: int, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _help(capsys, "extract")
    for line in out.splitlines():
        assert len(line) <= columns, line
        assert not re.search(r"[\w-]-$", line), line


@pytest.mark.parametrize(
    ("group", "commands"),
    [("batch", ("check", "cancel")), ("ranges", ("generate", "adjust"))],
)
def test_a_group_without_a_subcommand_prints_its_help(
    group: str, commands: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    assert entry.main([group]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"usage: chronominer {group}")
    for name in commands:
        assert name in out


EXIT_CODES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("extract",): ("several items", "without --all, --select or --files"),
    ("ranges", "generate"): ("every text file got its line ranges",),
    ("ranges", "adjust"): ("no text file with a line-ranges file",),
    ("batch", "check"): ("an --adopt or --abandon that names nothing",),
    ("batch", "cancel"): ("a cancellation or a provider listing failed",),
    ("repair",): ("a targeted chunk stayed unrepaired",),
    ("render",): ("another run holds a document",),
    ("drop",): ("a document was refused",),
}


@pytest.mark.parametrize("command", list(EXIT_CODES), ids=" ".join)
def test_each_command_lists_its_own_exit_codes(
    command: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    out = " ".join(_help(capsys, *command).split())
    assert "Exit codes: 0 " in out
    assert "130 interrupted (Ctrl+C or Ctrl+Break)" in out
    for phrase in EXIT_CODES[command]:
        assert phrase in out, phrase


def test_batch_repair_is_no_longer_a_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert entry.main(["batch", "repair"]) == 2
    assert "invalid choice" in capsys.readouterr().err


def test_removed_flags_are_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert entry.main(["extract", "--auto"]) == 2
    assert "unrecognized arguments" in capsys.readouterr().err
