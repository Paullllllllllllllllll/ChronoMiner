"""common/heartbeat.py: the launcher's progress file and the item progress."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from chronominer.common import heartbeat as heartbeat_module
from chronominer.common.heartbeat import ENV_VAR, Heartbeat, ItemProgress, RateEta

STAMP = "2026-10-13T08:00:00+00:00"


class Clock:
    """A settable monotonic clock."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def beat(path: Path, clock: Clock | None = None, interval: float = 15.0) -> Heartbeat:
    return Heartbeat(path, interval=interval, clock=clock or Clock(), now=lambda: STAMP)


def read(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def test_an_update_replaces_the_file_with_every_field(tmp_path: Path) -> None:
    path = tmp_path / "job.heartbeat.json"
    path.write_text("old", encoding="utf-8")

    assert beat(path).update(3, 40, "pages processed, item 2/5", 61.6)

    assert read(path) == {
        "units_done": 3,
        "units_total": 40,
        "unit_label": "pages processed, item 2/5",
        "eta_seconds": 62,
        "updated_at": STAMP,
    }
    assert list(tmp_path.glob("*.tmp")) == []


def test_an_update_without_eta_writes_null(tmp_path: Path) -> None:
    path = tmp_path / "hb.json"
    beat(path).update(0, 0, "batches")
    assert read(path)["eta_seconds"] is None


def test_a_locked_file_skips_the_update_and_the_next_one_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "hb.json"
    path.write_text("{}", encoding="utf-8")
    real = os.replace
    calls: list[int] = []

    def locked(src: Any, dst: Any) -> None:
        calls.append(1)
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(os, "replace", locked)
    writer = beat(path, interval=0)
    caplog.set_level(logging.DEBUG, logger=heartbeat_module.__name__)

    assert not writer.update(1, 2, "pages")
    assert not writer.update(1, 2, "pages")

    assert calls == [1, 1]
    assert path.read_text(encoding="utf-8") == "{}"
    assert list(tmp_path.glob("*.tmp")) == []
    levels = [record.levelno for record in caplog.records]
    assert levels == [logging.WARNING, logging.DEBUG]
    monkeypatch.setattr(os, "replace", real)
    assert writer.update(2, 2, "pages")
    assert read(path)["units_done"] == 2


def test_a_missing_folder_never_raises(tmp_path: Path) -> None:
    assert not beat(tmp_path / "gone" / "hb.json").update(1, 1, "pages")


def test_updates_are_throttled_unless_forced(tmp_path: Path) -> None:
    path = tmp_path / "hb.json"
    clock = Clock()
    writer = beat(path, clock)

    assert writer.update(1, 9, "pages")
    clock.now += 14
    assert not writer.update(2, 9, "pages")
    assert read(path)["units_done"] == 1
    assert writer.update(3, 9, "pages", force=True)
    clock.now += 15
    assert writer.update(4, 9, "pages")
    assert read(path)["units_done"] == 4


@pytest.mark.parametrize("value", [None, "", "   "])
def test_from_env_is_none_without_a_file(value: str | None) -> None:
    environ = {} if value is None else {ENV_VAR: value}
    assert Heartbeat.from_env(environ) is None


def test_from_env_names_the_file(tmp_path: Path) -> None:
    writer = Heartbeat.from_env({ENV_VAR: str(tmp_path / "hb.json")})
    assert writer is not None
    assert writer.path == tmp_path / "hb.json"


def test_the_rate_eta_needs_a_unit_after_the_start() -> None:
    clock = Clock()
    eta = RateEta(clock)
    eta.start()
    assert eta.seconds(0, 10) is None
    clock.now += 30
    assert eta.seconds(2, 10) == pytest.approx(120)
    clock.now += 30
    assert eta.seconds(10, 10) == 0


def test_the_item_progress_labels_items_and_estimates_the_last_one(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hb.json"
    clock = Clock()
    progress = ItemProgress(beat(path, clock, interval=0), "pages", clock)

    progress.item_started(1, 2)
    assert read(path) | {"updated_at": None} == {
        "units_done": 0,
        "units_total": 0,
        "unit_label": "pages processed, item 1/2",
        "eta_seconds": None,
        "updated_at": None,
    }
    clock.now += 10
    progress.units(1, 4)
    assert (read(path)["units_done"], read(path)["eta_seconds"]) == (1, None)
    progress.item_finished()
    assert read(path)["eta_seconds"] is None

    progress.item_started(2, 2)
    assert read(path)["eta_seconds"] is None
    clock.now += 20
    progress.units(2, 6, "images")
    data = read(path)
    assert data["unit_label"] == "images processed, item 2/2"
    assert (data["units_done"], data["units_total"], data["eta_seconds"]) == (2, 6, 40)
    progress.item_finished()
    assert read(path)["eta_seconds"] == 0
    progress.run_finished()
    assert read(path)["eta_seconds"] == 0


def test_the_item_and_run_ends_pass_the_throttle(tmp_path: Path) -> None:
    path = tmp_path / "hb.json"
    clock = Clock()
    progress = ItemProgress(beat(path, clock), "chunks", clock)

    progress.item_started(1, 1)
    progress.units(5, 5)
    assert read(path)["units_done"] == 0
    progress.item_finished()
    assert read(path)["units_done"] == 5


def test_a_run_without_items_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "hb.json"
    ItemProgress(beat(path), "pages").run_finished()
    assert not path.exists()


def test_a_failing_writer_never_reaches_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = beat(tmp_path / "hb.json", interval=0)

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("broken")

    monkeypatch.setattr(writer, "_write", broken)
    progress = ItemProgress(writer, "pages")

    progress.item_started(1, 1)
    progress.units(1, 1)
    progress.item_finished()
    progress.run_finished()
