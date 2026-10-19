"""repair: plan, then startup, then the repair work, and exit codes.

Run on Windows with:

    uv --directory <repo> run pytest tests/cli/test_repair_command.py -q

The repair core is replaced by a scripted plan where a test needs one;
missing inputs and refused databases go through the real planner.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import chronominer.cli.repair as repair_cli
from chronominer import __main__ as entry
from chronominer import ext
from chronominer.batch.repair import RepairResult
from chronominer.common.usage import Usage
from chronominer.llm.caller import CallEnv
from chronominer.output.store import DATABASE_NAME
from chronominer.settings import Settings, SettingsError, load
from chronominer.spec import CommandSpec, RepairSpec
from tests.cli.conftest import json_lines

ZERO_USAGE = {
    "total": {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
}


@dataclass
class _Plan:
    """A repair plan of *failures* chunks."""

    spec: RepairSpec
    failures: list[int]
    message: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def dry_run(self) -> bool:
        return self.spec.run.dry_run

    @property
    def to_repair(self) -> bool:
        return self.message is None and bool(self.failures)

    def outcome(self) -> RepairResult:
        return RepairResult(
            selected=len(self.failures) if self.to_repair else 0,
            dry_run=self.dry_run and self.to_repair,
            message=self.message,
        )


class Timeline:
    """Records the plan, startup and run steps of one repair command."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failures: list[int],
        *,
        refuse: bool = False,
        failed: int = 0,
    ) -> None:
        self.steps: list[str] = []
        self.refuse = refuse

        def plan(spec: RepairSpec, settings: Settings, **kwargs: Any) -> _Plan:
            assert isinstance(settings, Settings)
            self.steps.append(f"plan of {len(failures)} chunk(s)")
            message = None if failures else "No failed chunks detected."
            return _Plan(spec, failures, message, ("held by another run",))

        async def run(repair_plan: _Plan, events: Any) -> RepairResult:
            self.steps.append("run")
            CallEnv.create().record("extract", Usage(5, 1, 6))
            selected = len(repair_plan.failures)
            return RepairResult(
                repaired=selected - failed, failed=failed, selected=selected
            )

        monkeypatch.setattr(repair_cli, "plan_repair", plan)
        monkeypatch.setattr(repair_cli, "run_repair", run)
        monkeypatch.setattr(ext, "startup", self.startup)

    def startup(
        self, spec: CommandSpec, *, settings: Settings, args: argparse.Namespace
    ) -> None:
        assert isinstance(spec, RepairSpec)
        assert args.command == "repair"
        self.steps.append("startup")
        if self.refuse:
            raise SettingsError("extension refused the repair")


def repair(path: Path, *extra: str) -> int:
    return entry.main(["repair", "--input", str(path), "--json", *extra])


def one_line(capsys: pytest.CaptureFixture[str]) -> tuple[dict[str, Any], str]:
    captured = capsys.readouterr()
    (payload,) = json_lines(captured.out)
    assert len(captured.out.splitlines()) == 1
    return payload, captured.err


def contents(root: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize(("failed", "code"), [(0, 0), (1, 1)])
def test_startup_runs_after_the_plan_and_before_the_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failed: int,
    code: int,
) -> None:
    timeline = Timeline(monkeypatch, [0, 3], failed=failed)

    assert repair(tmp_path) == code

    assert timeline.steps == ["plan of 2 chunk(s)", "startup", "run"]
    payload, err = one_line(capsys)
    usage = payload.pop("usage")
    assert payload == {
        "dry_run": False,
        "command": "repair",
        "repaired": 2 - failed,
        "failed": failed,
    }
    assert usage["extract"]["total_tokens"] == 6
    assert usage["total"]["total_tokens"] == 6
    assert "held by another run" in err


def test_a_dry_run_calls_startup_but_not_the_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    timeline = Timeline(monkeypatch, [1])

    assert repair(tmp_path, "--dry-run") == 0

    assert timeline.steps == ["plan of 1 chunk(s)", "startup"]
    payload, err = one_line(capsys)
    assert payload == {
        "dry_run": True,
        "command": "repair",
        "repaired": 0,
        "failed": 0,
        "usage": ZERO_USAGE,
    }
    assert "[DRY RUN] 1 chunk(s) would be repaired." in err
    assert not load().log_file().parent.exists()


def test_nothing_to_repair_does_not_call_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    timeline = Timeline(monkeypatch, [])

    assert repair(tmp_path) == 0

    assert timeline.steps == ["plan of 0 chunk(s)"]
    payload, err = one_line(capsys)
    assert (payload["repaired"], payload["failed"]) == (0, 0)
    assert "No failed chunks detected" in err


def test_a_dry_run_with_nothing_to_repair_calls_startup_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    timeline = Timeline(monkeypatch, [])

    assert repair(tmp_path, "--dry-run") == 0

    assert timeline.steps == ["plan of 0 chunk(s)", "startup"]


@pytest.mark.parametrize("dry_run", [False, True])
def test_a_settings_error_from_startup_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dry_run: bool,
) -> None:
    timeline = Timeline(monkeypatch, [0], refuse=True)
    extra = ("--dry-run",) if dry_run else ()

    assert repair(tmp_path, *extra) == 2

    payload, err = one_line(capsys)
    assert payload["dry_run"] is dry_run
    assert "extension refused the repair" in err
    assert timeline.steps == ["plan of 1 chunk(s)", "startup"]


def test_an_error_of_the_repair_work_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    Timeline(monkeypatch, [0])

    async def broken(repair_plan: Any, events: Any) -> RepairResult:
        raise ValueError("page image unreadable")

    monkeypatch.setattr(repair_cli, "run_repair", broken)

    assert repair(tmp_path) == 1

    payload, err = one_line(capsys)
    assert payload == {
        "dry_run": False,
        "command": "repair",
        "repaired": 0,
        "failed": 0,
        "usage": ZERO_USAGE,
    }
    assert "Unexpected error: page image unreadable" in err
    assert "Invalid arguments" not in err


def test_the_provider_rule_exits_2_before_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    timeline = Timeline(monkeypatch, [0])

    assert repair(tmp_path, "--provider", "anthropic") == 2

    _payload, err = one_line(capsys)
    assert "--model" in err
    assert timeline.steps == []


def test_a_missing_input_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert repair(tmp_path / "missing") == 2

    payload, err = one_line(capsys)
    assert payload["command"] == "repair"
    assert "Invalid arguments" in err and "does not exist" in err


def test_a_versionless_database_exits_2_as_not_a_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    connection = sqlite3.connect(tmp_path / DATABASE_NAME)
    try:
        with connection:
            connection.execute("CREATE TABLE documents (name TEXT PRIMARY KEY)")
            connection.execute("CREATE TABLE pages (name TEXT, idx INTEGER)")
    finally:
        connection.close()
    before = contents(tmp_path)

    assert repair(tmp_path) == 2

    payload, err = one_line(capsys)
    assert payload["command"] == "repair"
    assert "is not a chronominer store" in err
    assert contents(tmp_path) == before
