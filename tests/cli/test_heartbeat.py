"""cli/heartbeat.py: the run's events and the batch wait in the progress file."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chronominer.batch.finalize import FAILED, FINALIZED_OUTCOME, PENDING
from chronominer.cli.heartbeat import BatchHeartbeat, HeartbeatEvents, with_heartbeat
from chronominer.common.heartbeat import ENV_VAR, Heartbeat, ItemProgress
from chronominer.common.polling import WaitTick
from chronominer.events import ChunkDone, ItemFinished, ItemStarted, NullEvents


class Recording(NullEvents):
    """Keep the names of the events received."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def item_started(self, event: ItemStarted) -> None:
        self.seen.append("item_started")

    def chunk_done(self, event: ChunkDone) -> None:
        self.seen.append("chunk_done")

    def item_finished(self, event: ItemFinished) -> None:
        self.seen.append("item_finished")


def read(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def test_the_events_reach_the_reporter_and_the_file(tmp_path: Path) -> None:
    path = tmp_path / "hb.json"
    inner = Recording()
    events = HeartbeatEvents(inner, ItemProgress(Heartbeat(path, interval=0), "chunks"))

    events.item_started(ItemStarted("a.txt", "text", 1, 2))
    assert read(path)["unit_label"] == "chunks processed, item 1/2"
    events.chunk_done(ChunkDone("a.txt", 0, "ok", 1, 3, already_complete=2))
    data = read(path)
    assert (data["units_done"], data["units_total"], data["eta_seconds"]) == (
        1,
        3,
        None,
    )
    events.item_finished(ItemFinished("a.txt", 1, 2, True))
    events.item_started(ItemStarted("b.txt", "text", 2, 2))
    events.chunk_done(ChunkDone("b.txt", 0, "ok", 1, 1))
    events.item_finished(ItemFinished("b.txt", 2, 2, True))

    assert inner.seen == [
        "item_started",
        "chunk_done",
        "item_finished",
        "item_started",
        "chunk_done",
        "item_finished",
    ]
    data = read(path)
    assert data["unit_label"] == "chunks processed, item 2/2"
    assert (data["units_done"], data["units_total"], data["eta_seconds"]) == (1, 1, 0)


def test_without_the_variable_the_reporter_runs_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    reporter = Recording()
    assert with_heartbeat(reporter) == (reporter, None)


def test_with_the_variable_the_reporter_gets_a_heartbeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "hb.json"))
    reporter = Recording()
    events, progress = with_heartbeat(reporter)
    assert isinstance(events, HeartbeatEvents)
    assert events.inner is reporter
    assert progress is not None


def _result(*outcomes: str) -> Any:
    return SimpleNamespace(
        submissions=[
            SimpleNamespace(database=Path("db"), submission_id=f"s{i}", outcome=outcome)
            for i, outcome in enumerate(outcomes)
        ]
    )


def test_the_wait_writes_the_finished_submissions_before_each_sleep(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hb.json"
    beat = BatchHeartbeat(Heartbeat(path))
    tick = WaitTick(passes=1, pending=2, elapsed=0, next_in=900, timeout=3600)

    beat.on_pass(_result(PENDING, PENDING))
    beat.on_wait(tick)
    assert (read(path)["units_done"], read(path)["units_total"]) == (0, 2)
    beat.on_pass(_result(FINALIZED_OUTCOME, FAILED, PENDING))
    beat.on_wait(tick)

    data = read(path)
    assert (data["units_done"], data["units_total"]) == (2, 3)
    assert (data["unit_label"], data["eta_seconds"]) == ("batches", None)
