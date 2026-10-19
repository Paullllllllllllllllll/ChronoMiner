"""Output formats at the command line.

``--formats`` takes csv only: ``{stem}_output.json`` and the database are
written whatever the formats, and the removed DOCX, TXT and JSON formats
are refused with exit 2.
"""

from __future__ import annotations

from typing import Any

import pytest

from chronominer import __main__ as entry
from tests.cli.conftest import MakeText, json_lines


def _one_line(capsys: pytest.CaptureFixture[str]) -> tuple[dict[str, Any], str]:
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1, captured.out
    (payload,) = json_lines(captured.out)
    return payload, captured.err


@pytest.fixture(autouse=True)
def dummy_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a dummy key variable so that a plan never fails on a missing key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _extract(make_text: MakeText, *extra: str) -> list[str]:
    return [
        "extract",
        "--schema",
        "price_list_entries",
        "--input",
        str(make_text()),
        *extra,
        "--dry-run",
        "--json",
    ]


@pytest.mark.parametrize("formats", ["json", "docx", "txt", "sqlite", "csv,txt"])
def test_removed_formats_exit_2_with_one_json_line(
    formats: str, make_text: MakeText, capsys: pytest.CaptureFixture[str]
) -> None:
    assert entry.main(_extract(make_text, "--formats", formats)) == 2

    payload, err = _one_line(capsys)
    assert payload["items_total"] == 0
    assert "is not one of csv" in err


@pytest.mark.parametrize("command", [["render"], ["batch", "check"]])
def test_other_commands_refuse_removed_formats(
    command: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert entry.main([*command, "--formats", "docx", "--json"]) == 2

    payload, err = _one_line(capsys)
    assert "is not one of csv" in err
    assert payload["dry_run"] is False
