"""batch check and batch cancel through the console entry point.

Run on Windows with:

    uv --directory <repo> run pytest tests/cli/test_batch_command.py -q

The check and cancel cores are replaced by fakes where a test needs a
result; refused databases and missing folders go through the real core.
``--wait`` runs the real check over fake batch APIs on a fake clock whose
sleep can finish a batch. Every exit writes exactly one JSON line.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import chronominer.cli.batch as batch_cli
from chronominer import __main__ as entry
from chronominer.batch import check as check_module
from chronominer.batch.cancel import CancelOutcome, CancelReport
from chronominer.batch.check import (
    ABANDONED,
    CheckResult,
    CheckStats,
    CheckTargetError,
    SubmissionCheck,
)
from chronominer.batch.submit import submit_batch
from chronominer.common.batch import BatchStatus
from chronominer.common.contract import usage_field
from chronominer.common.heartbeat import ENV_VAR
from chronominer.output.store import DATABASE_NAME, open_store
from chronominer.settings import Settings
from tests.batch.conftest import (
    KEYS,
    World,
    install_world,
    leased_document,
    page_images,
    run_spec,
)
from tests.cli.conftest import json_lines

CHECK_ZERO = {
    "dry_run": False,
    "command": "check_batches",
    "finalized": 0,
    "pending": 0,
    "failed": 0,
    "unresolved": 0,
    "submitted": 0,
    "complete": 0,
    "outputs": [],
    "usage": usage_field(),
}
CANCEL_ZERO = {
    "dry_run": False,
    "command": "cancel_batches",
    "cancelled": 0,
    "failed": 0,
}


def _one_line(capsys: pytest.CaptureFixture[str]) -> tuple[dict[str, Any], str]:
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1, captured.out
    (payload,) = json_lines(captured.out)
    return payload, captured.err


def _check(root: Path, *options: str) -> int:
    return entry.main(["batch", "check", "--input", str(root), *options, "--json"])


class _FakeCheck:
    """A check core that returns *result* and records its arguments."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, result: CheckResult | Exception
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result
        monkeypatch.setattr(batch_cli, "check_batches", self.check)

    async def check(
        self, directory: Path, settings: Any, **options: Any
    ) -> CheckResult:
        self.calls.append({"directory": directory, **options})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.parametrize(
    ("stats", "code"),
    [
        (CheckStats(finalized=2, pending=1), 0),
        (CheckStats(failed=2), 1),
        (CheckStats(unresolved=1), 1),
    ],
    ids=["finalized", "failed", "unresolved"],
)
def test_batch_check_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stats: CheckStats,
    code: int,
) -> None:
    fake = _FakeCheck(monkeypatch, CheckResult(stats=stats))

    assert _check(tmp_path, "--formats", "csv") == code

    payload, _err = _one_line(capsys)
    assert payload == {
        **CHECK_ZERO,
        "finalized": stats.finalized,
        "pending": stats.pending,
        "failed": stats.failed,
        "unresolved": stats.unresolved,
    }
    (call,) = fake.calls
    assert call["directory"] == tmp_path.resolve()
    assert tuple(call["formats"]) == ("csv",)
    assert "schema" not in call


def test_the_schema_filter_and_batch_ids_reach_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = _FakeCheck(monkeypatch, CheckResult())

    argv = ("--schema", "wage_records", "--batch-ids", "b1,b2")
    assert _check(tmp_path, *argv) == 0

    (call,) = fake.calls
    assert call["schema"] == "wage_records"
    assert tuple(call["batch_ids"]) == ("b1", "b2")
    assert call["formats"] is None


def test_abandon_reports_the_abandonment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = CheckResult()
    database = tmp_path / DATABASE_NAME
    result.add(SubmissionCheck(database, "s1", "openai", ABANDONED, detail="gone"))
    fake = _FakeCheck(monkeypatch, result)

    assert _check(tmp_path, "--abandon", "s1") == 0

    payload, err = _one_line(capsys)
    assert payload["abandoned"] is True
    assert fake.calls[0]["abandon"] == "s1"
    assert "Abandoned" in err


def test_a_target_that_fits_nothing_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _FakeCheck(monkeypatch, CheckTargetError("No unresolved openai submission"))

    assert _check(tmp_path, "--adopt", "batch_9") == 2

    payload, err = _one_line(capsys)
    assert payload == {**CHECK_ZERO, "adopted": False}
    assert "No unresolved openai submission" in err


def test_a_missing_folder_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _check(tmp_path / "missing") == 2

    payload, err = _one_line(capsys)
    assert payload == CHECK_ZERO
    assert "does not exist" in err


def test_a_versionless_database_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "out"
    root.mkdir()
    database = root / DATABASE_NAME
    connection = sqlite3.connect(database)
    connection.executescript(
        "CREATE TABLE documents (name TEXT PRIMARY KEY);"
        "CREATE TABLE pages (name TEXT, idx INTEGER);"
    )
    connection.close()
    before = database.read_bytes()

    assert _check(tmp_path) == 2

    payload, err = _one_line(capsys)
    assert payload["unresolved"] == 0
    assert "is not a chronominer store" in err
    assert database.read_bytes() == before


def test_batch_cancel_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def cancel(batch_ids: Any, settings: Any, *, args: Any) -> CancelReport:
        return CancelReport(
            outcomes=[
                CancelOutcome(batch_id, "openai", batch_id == "b1", error="gone")
                for batch_id in batch_ids
            ]
        )

    monkeypatch.setattr(batch_cli, "cancel_batches", cancel)

    assert entry.main(["batch", "cancel", "--batch-ids", "b1,b2", "--json"]) == 1
    payload, err = _one_line(capsys)
    assert payload == {**CANCEL_ZERO, "cancelled": 1, "failed": 1}
    assert "Failed to cancel b2: gone" in err


def test_cancel_input_cancels_the_recorded_submissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Path] = []

    async def cancel(directory: Path, settings: Any, *, args: Any) -> CancelReport:
        seen.append(directory)
        return CancelReport(outcomes=[CancelOutcome("b1", "openai", True)])

    monkeypatch.setattr(batch_cli, "cancel_submissions", cancel)

    code = entry.main(["batch", "cancel", "--input", str(tmp_path), "--json"])

    assert code == 0
    payload, _err = _one_line(capsys)
    assert payload == {**CANCEL_ZERO, "cancelled": 1}
    assert seen == [tmp_path.resolve()]


@pytest.mark.parametrize(
    "argv",
    [
        ["batch", "cancel", "--json"],
        ["batch", "cancel", "--all", "--batch-ids", "b1", "--json"],
    ],
    ids=["no target", "all and ids"],
)
def test_batch_cancel_needs_one_target(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert entry.main(argv) == 2
    payload, _err = _one_line(capsys)
    assert payload == CANCEL_ZERO


@pytest.mark.parametrize(
    ("failure", "code"), [(KeyboardInterrupt(), 130), (RuntimeError("boom"), 1)]
)
def test_interrupt_and_crash_write_one_json_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
    code: int,
) -> None:
    async def check(*_args: Any, **_kwargs: Any) -> CheckResult:
        raise failure

    monkeypatch.setattr(batch_cli, "check_batches", check)

    assert _check(tmp_path) == code
    payload, err = _one_line(capsys)
    assert payload == CHECK_ZERO
    assert "[ERROR]" in err


WAIT_ZERO = {**CHECK_ZERO, "polls": 0, "waited_seconds": 0, "stopped": None}


class _Clock:
    """The wait's clock; ``sleep`` advances it and runs *step* on its *at* call."""

    def __init__(self) -> None:
        self.now = 50.0
        self.sleeps: list[float] = []
        self.steps: dict[int, Callable[[], None]] = {}

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        step = self.steps.get(len(self.sleeps))
        if step is not None:
            step()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(check_module, "_clock", fake)
    monkeypatch.setattr(check_module, "_sleep", fake.sleep)
    return fake


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    """Fake batch APIs of every provider, each key variable set."""
    state = World(status=BatchStatus.IN_PROGRESS)
    for variable in KEYS.values():
        monkeypatch.setenv(variable, f"test-{variable.lower()}")
    install_world(state, monkeypatch)
    return state


def _submitted(root: Path) -> str:
    """Submit the two page chunks of document ``doc``; return the batch id."""

    async def submit() -> str:
        root.mkdir(parents=True, exist_ok=True)
        async with open_store(root) as store:
            await store.begin_run("extract")
            handle = await leased_document(store)
            outcome = await submit_batch(
                page_images(2),
                document=handle,
                spec=run_spec(root),
                settings=Settings(),
            )
            (submission_id,) = outcome.submission_ids
            batch_id = store.submission(submission_id).batch_id
        assert batch_id is not None
        return batch_id

    return asyncio.run(submit())


def _complete(world: World, batch_id: str) -> Callable[[], None]:
    def step() -> None:
        world.batches[batch_id].status = BatchStatus.COMPLETED

    return step


def test_a_wait_that_settles_exits_0(
    tmp_path: Path, world: World, clock: _Clock, capsys: pytest.CaptureFixture[str]
) -> None:
    batch_id = _submitted(tmp_path)
    clock.steps[1] = _complete(world, batch_id)

    assert _check(tmp_path, "--wait") == 0

    payload, err = _one_line(capsys)
    outputs = payload.pop("outputs")
    usage = payload.pop("usage")
    assert payload == {
        key: value
        for key, value in {
            **WAIT_ZERO,
            "finalized": 1,
            "complete": 1,
            "polls": 2,
            "waited_seconds": 900,
            "stopped": "settled",
        }.items()
        if key not in ("outputs", "usage")
    }
    assert [Path(path).name for path in outputs] == ["doc_output.json"]
    assert set(usage) == {"extract", "total"}
    assert clock.sleeps == [900]
    assert re.search(
        r"\[INFO\] Waiting for batches: 1 pending; next check at \d\d:\d\d "
        r"\(check 2; 0\.00 of 50 h\)\.",
        err,
    )
    assert "Stopped after 2 checks and 0.25 h: settled." in err
    assert "Waiting until" not in err
    assert err.count("not finished yet") == 1
    assert (tmp_path / "doc_output.json").is_file()


def test_a_wait_writes_the_heartbeat_file_before_it_sleeps(
    tmp_path: Path,
    world: World,
    clock: _Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id = _submitted(tmp_path)
    clock.steps[1] = _complete(world, batch_id)
    path = tmp_path / "job.heartbeat.json"
    monkeypatch.setenv(ENV_VAR, str(path))

    assert _check(tmp_path, "--wait") == 0

    data = json.loads(path.read_text(encoding="utf-8"))
    assert (data["units_done"], data["units_total"]) == (0, 1)
    assert (data["unit_label"], data["eta_seconds"]) == ("batches", None)


def test_a_wait_past_its_deadline_exits_1(
    tmp_path: Path, world: World, clock: _Clock, capsys: pytest.CaptureFixture[str]
) -> None:
    _submitted(tmp_path)

    code = _check(tmp_path, "--wait", "--poll-minutes", "1", "--wait-hours", "0.05")

    assert code == 1
    payload, err = _one_line(capsys)
    assert payload == {
        **WAIT_ZERO,
        "pending": 1,
        "polls": 4,
        "waited_seconds": 180,
        "stopped": "timeout",
    }
    assert clock.sleeps == [60, 60, 60]
    assert err.count("not finished yet") == 2
    assert "Stopped after 4 checks and 0.05 h: timeout." in err


def test_a_refused_database_under_wait_exits_2(
    tmp_path: Path, clock: _Clock, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "out"
    root.mkdir()
    connection = sqlite3.connect(root / DATABASE_NAME)
    connection.executescript("CREATE TABLE documents (name TEXT PRIMARY KEY);")
    connection.close()

    assert _check(tmp_path, "--wait") == 2

    payload, _err = _one_line(capsys)
    assert payload == {**WAIT_ZERO, "polls": 1, "stopped": "settled"}
    assert clock.sleeps == []


def test_ctrl_c_during_a_wait_exits_130_with_the_totals(
    tmp_path: Path, world: World, clock: _Clock, capsys: pytest.CaptureFixture[str]
) -> None:
    _submitted(tmp_path)

    def interrupt() -> None:
        raise KeyboardInterrupt

    clock.steps[1] = interrupt

    assert _check(tmp_path, "--wait") == 130

    payload, err = _one_line(capsys)
    assert payload == {
        **WAIT_ZERO,
        "pending": 1,
        "polls": 1,
        "waited_seconds": 900,
        "stopped": "interrupted",
    }
    assert "Stopped after 1 check and 0.25 h: interrupted." in err


def test_an_error_out_of_a_check_ends_the_wait_with_1(
    tmp_path: Path,
    world: World,
    clock: _Clock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def broken(*_args: Any, **_kwargs: Any) -> CheckResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(check_module, "check_submissions", broken)

    assert _check(tmp_path, "--wait") == 1

    payload, err = _one_line(capsys)
    assert payload == {**WAIT_ZERO, "polls": 1, "stopped": "error"}
    assert "Check 1 failed: boom" in err


def test_a_wait_target_that_fits_nothing_exits_2(
    tmp_path: Path, world: World, clock: _Clock, capsys: pytest.CaptureFixture[str]
) -> None:
    _submitted(tmp_path)

    assert _check(tmp_path, "--wait", "--abandon", "no-such") == 2

    payload, err = _one_line(capsys)
    assert payload == {**WAIT_ZERO, "abandoned": False}
    assert "No submission no-such" in err


@pytest.mark.parametrize(
    "option", [("--poll-minutes", "5"), ("--wait-hours", "2")], ids=lambda o: o[0]
)
def test_the_wait_values_need_wait(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], option: tuple[str, str]
) -> None:
    assert _check(tmp_path, *option) == 2

    payload, err = _one_line(capsys)
    assert payload == CHECK_ZERO
    assert f"{option[0]} needs --wait" in err
