"""The ui subcommand on the command line: the terminal check and usage errors.

Its help is covered by ``test_help.py`` and ``test_help_snapshots.py``.
"""

from __future__ import annotations

import pytest

from chronominer import __main__ as entry
from chronominer.cli.command import summary_for


def test_ui_without_a_terminal_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert entry.main(["ui"]) == 2
    captured = capsys.readouterr()
    assert "chronominer extract" in captured.err
    assert captured.out == ""


def test_a_ui_usage_error_writes_no_json_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert entry.main(["ui", "--json"]) == 2
    captured = capsys.readouterr()
    assert "unrecognized arguments: --json" in captured.err
    assert captured.out == ""
    assert summary_for(["ui", "--json"]).emit(2) == 2
    assert capsys.readouterr().out == ""
