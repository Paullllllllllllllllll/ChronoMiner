"""ranges generate and ranges adjust through the console entry point.

Run on Windows with:

    uv --directory <repo> run pytest tests/cli/test_ranges_command.py -q

generate runs for real (it calls no model); adjust plans for real and runs
through a fake. Every exit writes exactly one JSON line with ``--json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

import chronominer.cli.ranges as ranges_cli
from chronominer import __main__ as entry
from chronominer import ext
from chronominer.common.usage import Usage
from chronominer.output.store import DATABASE_NAME
from chronominer.ranges.lines import ranges_path, read_ranges
from chronominer.run import ItemResult, RangesPlan, RunResult
from chronominer.settings import Settings
from chronominer.spec import CommandSpec, RangesAdjustSpec
from tests.cli.conftest import MakeText, json_lines

SCHEMA = "price_list_entries"
SUMMARY_KEYS = {
    "dry_run",
    "items_total",
    "items_complete",
    "items_failed",
    "items_skipped",
    "items_blocked",
    "items_submitted",
    "outputs",
    "usage",
}


def _one_line(capsys: pytest.CaptureFixture[str]) -> tuple[dict[str, Any], str]:
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1, captured.out
    (payload,) = json_lines(captured.out)
    return payload, captured.err


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_ranges(text: Path) -> None:
    ranges_path(text).write_text("(1, 20)\n(21, 40)\n", encoding="utf-8")


def _generate(text: Path, *extra: str) -> int:
    return entry.main(
        ["ranges", "generate", "--input", str(text), "--chunk-tokens", "60", *extra]
    )


def _adjust(text: Path, *extra: str) -> int:
    return entry.main(
        ["ranges", "adjust", "--input", str(text), "--schema", SCHEMA, *extra]
    )


def test_generate_writes_the_ranges_and_the_database(
    make_text: MakeText, capsys: pytest.CaptureFixture[str]
) -> None:
    text = make_text()

    assert _generate(text, "--json") == 0

    payload, _err = _one_line(capsys)
    target = ranges_path(text)
    ranges = read_ranges(target)
    assert len(ranges) > 1
    assert ranges[0][0] == 1 and ranges[-1][1] == 40
    assert (payload["items_total"], payload["items_complete"]) == (1, 1)
    database = str((text.parent / DATABASE_NAME).resolve())
    assert payload["outputs"] == [str(target), database]


def test_generate_dry_run_writes_nothing(
    tmp_path: Path, make_text: MakeText, capsys: pytest.CaptureFixture[str]
) -> None:
    text = make_text()
    before = _files(tmp_path)

    assert _generate(text, "--dry-run", "--json") == 0

    payload, err = _one_line(capsys)
    assert set(payload) == SUMMARY_KEYS | {"to_process", "spec"}
    assert payload["dry_run"] is True
    assert payload["to_process"] == [{"name": "prices.txt", "type": "text"}]
    assert payload["spec"]["chunk_tokens"] == {"value": 60, "source": "flag"}
    assert "[DRY RUN]" in err
    assert _files(tmp_path) == before


def test_generate_of_a_missing_input_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _generate(tmp_path / "missing.txt", "--json") == 2

    payload, err = _one_line(capsys)
    assert set(payload) == SUMMARY_KEYS
    assert "Invalid arguments" in err


def test_adjust_without_the_key_exits_2_but_its_dry_run_warns(
    tmp_path: Path,
    make_text: MakeText,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    text = make_text()
    _write_ranges(text)
    before = _files(tmp_path)

    assert _adjust(text, "--dry-run", "--json") == 0
    payload, err = _one_line(capsys)
    assert payload["dry_run"] is True
    assert "[WARNING]" in err and "OPENAI_API_KEY" in err

    assert _adjust(text, "--json") == 2
    _payload, err = _one_line(capsys)
    assert "OPENAI_API_KEY" in err
    assert _files(tmp_path) == before


def test_adjust_without_a_ranges_file_exits_2_naming_generate(
    make_text: MakeText,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    assert _adjust(make_text(), "--json") == 2

    _payload, err = _one_line(capsys)
    assert "ranges generate" in err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("--provider", "anthropic"), "--model"),
        (("--schema", "no_such_schema"), "no_such_schema"),
    ],
    ids=["provider rule", "unknown schema"],
)
def test_adjust_refuses_a_bad_model_or_schema(
    make_text: MakeText,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: tuple[str, ...],
    message: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    text = make_text()
    _write_ranges(text)

    assert _adjust(text, *argv, "--dry-run", "--json") == 2

    _payload, err = _one_line(capsys)
    assert message in err


def test_adjust_without_a_schema_exits_2(
    make_text: MakeText, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["ranges", "adjust", "--input", str(make_text()), "--json"]

    assert entry.main(argv) == 2

    _payload, err = _one_line(capsys)
    assert "--schema" in err


class _Timeline:
    """Records the startup and run steps of one adjust command."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, failed: bool) -> None:
        self.steps: list[str] = []
        self.failed = failed
        monkeypatch.setattr(ext, "startup", self.startup)
        monkeypatch.setattr(ranges_cli, "run_ranges", self.run)

    def startup(
        self, spec: CommandSpec, *, settings: Settings, args: argparse.Namespace
    ) -> None:
        assert isinstance(spec, RangesAdjustSpec)
        assert isinstance(settings, Settings)
        assert args.ranges_command == "adjust"
        self.steps.append("startup")

    async def run(self, ranges_plan: RangesPlan, events: Any) -> RunResult:
        self.steps.append(f"run of {len(ranges_plan.files)} file(s)")
        items = tuple(
            ItemResult(path.name, not self.failed, (str(ranges_path(path)),))
            for path in ranges_plan.files
        )
        return RunResult(items=items, usage={"adjust": Usage(10, 2, 12)})


@pytest.mark.parametrize(("failed", "code"), [(False, 0), (True, 1)])
def test_adjust_starts_the_extension_then_runs_and_reports(
    make_text: MakeText,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failed: bool,
    code: int,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    timeline = _Timeline(monkeypatch, failed=failed)
    text = make_text()
    _write_ranges(text)

    assert _adjust(text, "--json") == code

    assert timeline.steps == ["startup", "run of 1 file(s)"]
    payload, _err = _one_line(capsys)
    assert (payload["items_complete"], payload["items_failed"]) == (
        int(not failed),
        int(failed),
    )
    assert payload["usage"]["adjust"]["total_tokens"] == 12


def test_adjust_dry_run_reports_the_spec_without_running(
    tmp_path: Path,
    make_text: MakeText,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    timeline = _Timeline(monkeypatch, failed=False)
    text = make_text()
    _write_ranges(text)
    before = _files(tmp_path)

    assert _adjust(text, "--context-window", "9", "--dry-run", "--json") == 0

    assert timeline.steps == ["startup"]
    payload, _err = _one_line(capsys)
    assert payload["items_total"] == 1
    assert payload["spec"]["schema"] == {"value": SCHEMA, "source": "flag"}
    assert payload["spec"]["context_window"] == {"value": 9, "source": "flag"}
    assert _files(tmp_path) == before
