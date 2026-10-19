"""The agent contract of extract: exit codes, the JSON line and dry runs.

Run on Windows with:

    uv --directory <repo> run pytest tests/cli/test_extract_command.py -q

The model run is replaced by a fake where a test needs one; plans and dry
runs go through the real core.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import chronominer.cli.extract as extract_cli
from chronominer import __main__ as entry
from chronominer import ext
from chronominer.common.heartbeat import ENV_VAR
from chronominer.common.usage import Usage
from chronominer.events import ChunkDone, ItemFinished, ItemStarted
from chronominer.llm.caller import CallEnv
from chronominer.output.store import DATABASE_NAME
from chronominer.run import ItemResult, RunPlan, RunResult
from chronominer.settings import Settings
from chronominer.spec import CommandSpec
from tests.cli.conftest import MakeText, json_lines

SCHEMA = "price_list_entries"
EXTRACT_KEYS = {
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
PLAN_KEYS = {"to_process", "skipped", "unstored", "blocked", "refused", "spec"}
ZERO_USAGE = {
    "total": {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
}


@pytest.fixture(autouse=True)
def _openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plans check the key variable of the provider they would call."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _extract(*argv: str) -> int:
    return entry.main(["extract", *argv])


def _one_line(capsys: pytest.CaptureFixture[str]) -> tuple[dict[str, Any], str]:
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1, captured.out
    (payload,) = json_lines(captured.out)
    return payload, captured.err


def _base(tmp_path: Path, make_text: MakeText) -> list[str]:
    return ["--schema", SCHEMA, "--input", str(make_text())]


def _usage_error(tmp_path: Path, make_text: MakeText) -> list[str]:
    return [*_base(tmp_path, make_text), "--concurrency", "zero"]


def _unknown_flag(tmp_path: Path, make_text: MakeText) -> list[str]:
    return [*_base(tmp_path, make_text), "--bogus"]


def _missing_settings(tmp_path: Path, make_text: MakeText) -> list[str]:
    return [*_base(tmp_path, make_text), "--settings", str(tmp_path / "nope.yaml")]


def _unknown_endpoint(tmp_path: Path, make_text: MakeText) -> list[str]:
    return [*_base(tmp_path, make_text), "--endpoint", "missing", "--model", "m"]


def _missing_input(tmp_path: Path, make_text: MakeText) -> list[str]:
    return ["--schema", SCHEMA, "--input", str(tmp_path / "missing.txt")]


def _no_input(tmp_path: Path, make_text: MakeText) -> list[str]:
    return ["--schema", SCHEMA]


def _no_schema(tmp_path: Path, make_text: MakeText) -> list[str]:
    return ["--input", str(make_text())]


def _unknown_schema(tmp_path: Path, make_text: MakeText) -> list[str]:
    return ["--schema", "no_such_schema", "--input", str(make_text())]


def _dry_run(tmp_path: Path, make_text: MakeText) -> list[str]:
    return [*_base(tmp_path, make_text), "--output", str(tmp_path / "out"), "--dry-run"]


@pytest.mark.parametrize(
    ("setup", "code", "total"),
    [
        pytest.param(_usage_error, 2, 0, id="usage_error"),
        pytest.param(_unknown_flag, 2, 0, id="unknown_flag"),
        pytest.param(_missing_settings, 2, 0, id="missing_settings"),
        pytest.param(_unknown_endpoint, 2, 0, id="unknown_endpoint"),
        pytest.param(_missing_input, 2, 0, id="missing_input"),
        pytest.param(_no_input, 2, 0, id="no_input"),
        pytest.param(_no_schema, 2, 0, id="no_schema"),
        pytest.param(_dry_run, 0, 1, id="dry_run"),
    ],
)
def test_every_exit_writes_one_json_line(
    tmp_path: Path,
    make_text: MakeText,
    capsys: pytest.CaptureFixture[str],
    setup: Callable[[Path, MakeText], list[str]],
    code: int,
    total: int,
) -> None:
    argv = setup(tmp_path, make_text)

    assert _extract(*argv, "--json") == code

    payload, err = _one_line(capsys)
    dry_run = "--dry-run" in argv
    assert set(payload) == EXTRACT_KEYS | (PLAN_KEYS if dry_run and not code else set())
    assert payload["dry_run"] is dry_run
    assert payload["items_total"] == total
    if code:
        assert "[ERROR]" in err


def test_an_unknown_schema_exits_2_naming_the_schemas(
    tmp_path: Path, make_text: MakeText, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _extract(*_unknown_schema(tmp_path, make_text), "--dry-run", "--json") == 2

    _payload, err = _one_line(capsys)
    assert "no_such_schema" in err and "PriceListEntries" in err


def test_a_dry_run_without_the_key_warns_and_writes_nothing(
    tmp_path: Path,
    make_text: MakeText,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    out = tmp_path / "out"

    assert _extract(*_dry_run(tmp_path, make_text), "--json") == 0

    payload, err = _one_line(capsys)
    assert payload["dry_run"] is True
    assert payload["items_total"] == 1
    assert "[WARNING]" in err and "OPENAI_API_KEY" in err
    assert not out.exists()
    argv = [*_base(tmp_path, make_text), "--output", str(out), "--json"]
    assert _extract(*argv) == 2
    _payload, err = _one_line(capsys)
    assert "OPENAI_API_KEY" in err
    assert not out.exists()


def test_a_dry_run_reports_the_plan_and_the_effective_spec(
    tmp_path: Path, make_text: MakeText, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _extract(*_dry_run(tmp_path, make_text), "--json") == 0

    payload, _err = _one_line(capsys)
    (item,) = payload["to_process"]
    assert item == {"name": "prices.txt", "type": "text", "state": "none"}
    assert payload["skipped"] == []
    assert payload["spec"]["prompt.schema"] == {"value": SCHEMA, "source": "flag"}
    assert payload["spec"]["output.formats"] == {"value": [], "source": "default"}
    assert payload["spec"]["model.model"] == {
        "value": "gpt-6-luna",
        "source": "default",
    }


def test_a_dry_run_writes_nothing(
    tmp_path: Path, make_text: MakeText, capsys: pytest.CaptureFixture[str]
) -> None:
    text = make_text("in/prices.txt")
    out = tmp_path / "out"

    argv = ["--schema", SCHEMA, "--input", str(text), "--output", str(out)]
    assert _extract(*argv, "--dry-run") == 0

    assert not out.exists()
    assert sorted(path.name for path in text.parent.iterdir()) == ["prices.txt"]
    assert not (tmp_path / DATABASE_NAME).exists()
    assert "[DRY RUN] 1 discovered" in capsys.readouterr().err


def test_a_usage_error_writes_the_usage_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _extract("--service-tier", "bogus", "--json") == 2
    payload, err = _one_line(capsys)
    assert set(payload) == EXTRACT_KEYS
    assert "usage: chronominer extract" in err
    assert "'bogus' is not one of" in err


@pytest.mark.parametrize(
    ("failure", "code"), [(KeyboardInterrupt(), 130), (RuntimeError("boom"), 1)]
)
def test_interrupt_and_crash_write_one_json_line(
    tmp_path: Path,
    make_text: MakeText,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
    code: int,
) -> None:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(extract_cli, "plan", _raise)

    assert _extract(*_base(tmp_path, make_text), "--json") == code
    payload, err = _one_line(capsys)
    assert payload == {
        "dry_run": False,
        "items_total": 0,
        "items_complete": 0,
        "items_failed": 0,
        "items_skipped": 0,
        "items_blocked": 0,
        "items_submitted": 0,
        "outputs": [],
        "usage": ZERO_USAGE,
    }
    assert "[ERROR]" in err


def _fake_run(
    outputs: list[str], *, failed: bool = False
) -> Callable[[RunPlan, Any], Any]:
    async def run(run_plan: RunPlan, events: Any) -> RunResult:
        items = tuple(
            ItemResult(item.name, not failed, () if failed else tuple(outputs))
            for item in run_plan.to_process
        )
        database = run_plan.output_dir or run_plan.input_path.parent
        return RunResult(
            items=items,
            usage={"extract": Usage(5, 1, 6)},
            databases=(str(database / DATABASE_NAME),),
        )

    return run


@pytest.mark.parametrize(("failed", "code"), [(False, 0), (True, 1)])
def test_a_run_reports_its_outputs_tokens_and_the_database(
    tmp_path: Path,
    make_text: MakeText,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failed: bool,
    code: int,
) -> None:
    out = tmp_path / "out"
    output = str(out / "prices_output.json")
    monkeypatch.setattr(extract_cli, "run", _fake_run([output], failed=failed))

    argv = [*_base(tmp_path, make_text), "--output", str(out), "--json"]
    assert _extract(*argv) == code

    payload, _err = _one_line(capsys)
    assert (payload["items_complete"], payload["items_failed"]) == (
        int(not failed),
        int(failed),
    )
    database = str(out.resolve() / DATABASE_NAME)
    expected = [database] if failed else [output, database]
    assert payload["outputs"] == expected
    assert payload["usage"]["extract"]["total_tokens"] == 6
    assert payload["usage"]["total"]["total_tokens"] == 6


def test_items_waiting_in_batches_count_as_submitted(
    tmp_path: Path,
    make_text: MakeText,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def run(run_plan: RunPlan, events: Any) -> RunResult:
        items = tuple(
            ItemResult(item.name, True, (), submitted=True)
            for item in run_plan.to_process
        )
        return RunResult(items=items)

    monkeypatch.setattr(extract_cli, "run", run)

    assert _extract(*_base(tmp_path, make_text), "--json") == 0

    payload, _err = _one_line(capsys)
    assert payload["items_submitted"] == 1
    assert (payload["items_complete"], payload["items_failed"]) == (0, 0)


@pytest.mark.parametrize(
    ("reason", "leased", "code"),
    [
        ("leased by run other-run until 2026-10-18 14:00:00", True, 1),
        # Chunks waiting for a batch are submitted work: named, counted, exit 0.
        ("2 unit(s) await an open or unresolved batch submission", False, 0),
    ],
    ids=["leased", "batch"],
)
def test_a_blocked_item_is_named_and_a_leased_one_exits_1(
    tmp_path: Path,
    make_text: MakeText,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    reason: str,
    leased: bool,
    code: int,
) -> None:
    async def run(run_plan: RunPlan, events: Any) -> RunResult:
        return RunResult(
            items=(),
            blocked=("prices.txt",),
            blocked_reasons=(reason,),
            leased=("prices.txt",) if leased else (),
        )

    monkeypatch.setattr(extract_cli, "run", run)

    assert _extract(*_base(tmp_path, make_text), "--json") == code

    payload, err = _one_line(capsys)
    assert (payload["items_blocked"], payload["items_skipped"]) == (1, 1)
    assert payload["items_failed"] == 0
    assert f"Not processed (blocked): 'prices.txt': {reason}." in err
    incomplete = "Processing incomplete: 1 item(s) leased by another run" in err
    assert incomplete is leased
    assert ("Processing complete!" in err) is not leased


def test_blocked_items_of_one_name_keep_their_own_reasons(
    tmp_path: Path,
    make_text: MakeText,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def run(run_plan: RunPlan, events: Any) -> RunResult:
        return RunResult(
            items=(),
            blocked=("prices.txt", "prices.txt"),
            blocked_reasons=("leased by run run-a", "leased by run run-b"),
            leased=("prices.txt", "prices.txt"),
        )

    monkeypatch.setattr(extract_cli, "run", run)

    assert _extract(*_base(tmp_path, make_text), "--json") == 1

    _payload, err = _one_line(capsys)
    for run_id in ("run-a", "run-b"):
        assert f"'prices.txt': leased by run {run_id}." in err


def test_an_item_pages_left_partial_counts_neither_complete_nor_failed(
    tmp_path: Path,
    make_text: MakeText,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def run(run_plan: RunPlan, events: Any) -> RunResult:
        items = tuple(
            ItemResult(item.name, True, (), partial=True)
            for item in run_plan.to_process
        )
        return RunResult(items=items)

    monkeypatch.setattr(extract_cli, "run", run)

    assert _extract(*_base(tmp_path, make_text), "--json") == 0

    payload, err = _one_line(capsys)
    assert (payload["items_complete"], payload["items_failed"]) == (0, 0)
    assert "1 item(s) left partial by --pages." in err


@pytest.mark.parametrize(
    ("failure", "code"), [(KeyboardInterrupt(), 130), (RuntimeError("boom"), 1)]
)
def test_an_interrupted_run_reports_the_tokens_spent(
    tmp_path: Path,
    make_text: MakeText,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
    code: int,
) -> None:
    async def spend(*_args: Any, **_kwargs: Any) -> Any:
        CallEnv.create().record("extract", Usage(5, 1, 6))
        raise failure

    monkeypatch.setattr(extract_cli, "run", spend)

    assert _extract(*_base(tmp_path, make_text), "--json") == code
    payload, _err = _one_line(capsys)
    assert payload["usage"]["extract"]["total_tokens"] == 6
    assert payload["usage"]["total"]["total_tokens"] == 6


def test_a_run_writes_its_chunks_to_the_heartbeat_file(
    tmp_path: Path, make_text: MakeText, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "job.heartbeat.json"
    monkeypatch.setenv(ENV_VAR, str(path))

    async def run(run_plan: RunPlan, events: Any) -> RunResult:
        events.item_started(ItemStarted("prices.txt", "text", 1, 1))
        for done in (1, 2):
            events.chunk_done(ChunkDone("prices.txt", done - 1, "ok", done, 2))
        events.item_finished(ItemFinished("prices.txt", 1, 1, True))
        return RunResult(items=())

    monkeypatch.setattr(extract_cli, "run", run)

    assert _extract(*_base(tmp_path, make_text)) == 0

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["unit_label"] == "chunks processed, item 1/1"
    assert (data["units_done"], data["units_total"], data["eta_seconds"]) == (2, 2, 0)


def test_without_json_stdout_stays_empty(
    tmp_path: Path,
    make_text: MakeText,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(extract_cli, "run", _fake_run([]))

    assert _extract(*_base(tmp_path, make_text)) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Processing complete" in captured.err


def test_relative_paths_resolve_against_the_current_directory(
    tmp_path: Path,
    make_text: MakeText,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_text("work/prices.txt")
    monkeypatch.chdir(tmp_path / "work")
    seen: list[RunPlan] = []

    async def run(run_plan: RunPlan, events: Any) -> RunResult:
        seen.append(run_plan)
        return RunResult(items=())

    monkeypatch.setattr(extract_cli, "run", run)

    argv = ["--schema", SCHEMA, "--input", "prices.txt", "--output", "out"]
    assert _extract(*argv) == 0

    (run_plan,) = seen
    assert run_plan.input_path == (tmp_path / "work" / "prices.txt").resolve()
    assert run_plan.output_dir == (tmp_path / "work" / "out").resolve()


class _Timeline:
    """Records ext.startup calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[CommandSpec] = []
        monkeypatch.setattr(ext, "startup", self.startup)

    def startup(
        self, spec: CommandSpec, *, settings: Settings, args: argparse.Namespace
    ) -> None:
        assert isinstance(settings, Settings)
        assert isinstance(args, argparse.Namespace)
        self.calls.append(spec)


def test_startup_runs_before_processing(
    tmp_path: Path, make_text: MakeText, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeline = _Timeline(monkeypatch)
    order: list[str] = []
    fake = _fake_run([])

    async def process(*args: Any, **kwargs: Any) -> Any:
        order.append(f"process after {len(timeline.calls)} startup call(s)")
        return await fake(*args, **kwargs)

    monkeypatch.setattr(extract_cli, "run", process)

    assert _extract(*_base(tmp_path, make_text)) == 0
    assert order == ["process after 1 startup call(s)"]
