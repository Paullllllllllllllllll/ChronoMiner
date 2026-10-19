"""A run whose output file cannot be read or replaced finishes every item.

Run on Windows with (``<repo>`` is the repository root):

    uv --directory <repo> run pytest tests/core/test_run_render_errors.py -q

Two text files, ``a.txt`` and ``b.txt``, are extracted through the CLI by a
scripted chat model (``tests.llm.helpers``), so no request leaves the
machine. ``a_output.json`` is either held open with an exclusive Windows
handle (share mode 0, skipped elsewhere) or its render is patched to raise
``PermissionError``. Oracle: the run exits 0 without "Unexpected error",
renders ``b_output.json``, warns with a message naming ``a_output.json`` and
keeps the units of ``a.txt`` stored as done.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import chronominer.run as core
from chronominer import __main__ as entry
from chronominer.common.sqlite import read
from chronominer.output.store import DATABASE_NAME
from chronominer.spec import to_argv
from tests.conftest import make_settings, make_spec
from tests.llm import helpers
from tests.llm.helpers import Script

ANSWER: dict[str, Any] = {
    "contains_no_content_of_requested_type": False,
    "entries": [],
}
UNSTORED = '{"records": []}\n'

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="exclusive file handles are Windows-only"
)


def _text(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"1. {path.stem}, mason, 2s 6d\n", encoding="utf-8")
    return path


@contextlib.contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    """Hold *path* open for reading with share mode 0."""
    import _winapi

    handle = _winapi.CreateFile(
        str(path), _winapi.GENERIC_READ, 0, 0, _winapi.OPEN_EXISTING, 0, 0
    )
    try:
        yield
    finally:
        _winapi.CloseHandle(handle)


class Scenario:
    """Two text files, an output folder and the CLI that extracts them."""

    def __init__(self, tmp_path: Path, script: Script) -> None:
        self.input = tmp_path / "in"
        for name in ("a", "b"):
            _text(self.input / f"{name}.txt")
        self.out = tmp_path / "out"
        self.script = script
        spec = make_spec(self.input, output=self.out)
        planned = core.plan(spec, make_settings()).to_process
        self.targets = {item.path.stem: item.output for item in planned}

    @property
    def db(self) -> Path:
        return self.out / DATABASE_NAME

    def target(self, name: str) -> Path:
        return self.targets[name]

    def main(self, capsys: pytest.CaptureFixture[str], **flags: Any) -> Outcome:
        flags.setdefault("concurrency", 1)
        spec = make_spec(self.input, output=self.out, **flags)
        code = entry.main(["extract", *to_argv(spec), "--json"])
        captured = capsys.readouterr()
        return Outcome(code, captured.out, captured.err)

    def statuses(self, name: str) -> list[str]:
        rows = read(
            self.db,
            lambda connection: connection.execute(
                "SELECT u.status FROM units u JOIN documents d USING (doc_id) "
                "WHERE d.input_path LIKE ? ORDER BY u.idx",
                (f"%{name}.txt",),
            ).fetchall(),
        )
        return [str(status) for (status,) in rows]

    def stale(self, path: Path) -> None:
        connection = sqlite3.connect(self.db)
        try:
            connection.execute(
                "UPDATE outputs SET stale = 1 WHERE path LIKE ?", (f"%{path.name}",)
            )
            connection.commit()
        finally:
            connection.close()


class Outcome:
    """The exit code and console output of one CLI call."""

    def __init__(self, code: int, out: str, err: str) -> None:
        self.code = code
        self.out = out
        self.err = err

    @property
    def summary(self) -> dict[str, Any]:
        line = self.out.strip().splitlines()[-1]
        return dict(json.loads(line))


@pytest.fixture(autouse=True)
def openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variable of the default provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scenario:
    return Scenario(tmp_path, helpers.install(monkeypatch, [ANSWER]))


def _patch_render(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    """Make the render of *target*'s item raise PermissionError."""
    original: Callable[..., Any] = core.render_item

    async def render_item(
        store: Any, doc_id: Any, formats: Any, output: Path, *a: Any, **k: Any
    ) -> Any:
        if output == target:
            raise PermissionError(13, "Permission denied", str(target))
        return await original(store, doc_id, formats, output, *a, **k)

    monkeypatch.setattr(core, "render_item", render_item)


def _assert_survived(scenario: Scenario, outcome: Outcome, held: Path) -> None:
    assert "Unexpected error" not in outcome.out + outcome.err
    assert outcome.code == 0, outcome.err
    records = json.loads(scenario.target("b").read_text(encoding="utf-8"))["records"]
    assert [record["chunk_index"] for record in records] == [1]
    assert held.name in outcome.err
    assert scenario.statuses("a") == ["done"]
    assert scenario.statuses("b") == ["done"]


def _unstored_first_file(scenario: Scenario) -> Path:
    held = scenario.target("a")
    held.parent.mkdir(parents=True, exist_ok=True)
    held.write_text(UNSTORED, encoding="utf-8")
    return held


@windows_only
def test_a_force_run_survives_an_exclusively_held_output(
    scenario: Scenario, capsys: pytest.CaptureFixture[str]
) -> None:
    held = _unstored_first_file(scenario)
    target_b = scenario.target("b")

    with _exclusive(held):
        outcome = scenario.main(capsys, force=True)

    _assert_survived(scenario, outcome, held)
    assert outcome.summary["items_failed"] == 0
    assert held.read_text(encoding="utf-8") == UNSTORED
    assert target_b.is_file()


def test_a_force_run_survives_a_render_that_raises_permission_error(
    scenario: Scenario,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = _unstored_first_file(scenario)
    _patch_render(monkeypatch, held)

    outcome = scenario.main(capsys, force=True)

    _assert_survived(scenario, outcome, held)
    assert len(scenario.script.requests) == 2
    notes = read(
        scenario.db,
        lambda connection: connection.execute("SELECT message FROM notes").fetchall(),
    )
    assert any(held.name in str(message) for (message,) in notes)


def _complete_with_stale_and_missing(
    scenario: Scenario, capsys: pytest.CaptureFixture[str]
) -> Path:
    first = scenario.main(capsys)
    assert first.code == 0, first.err
    held = scenario.target("a")
    scenario.stale(held)
    scenario.target("b").unlink()
    return held


@windows_only
def test_render_only_survives_an_exclusively_held_output(
    scenario: Scenario, capsys: pytest.CaptureFixture[str]
) -> None:
    held = _complete_with_stale_and_missing(scenario, capsys)
    calls = len(scenario.script.requests)

    with _exclusive(held):
        outcome = scenario.main(capsys)

    _assert_survived(scenario, outcome, held)
    assert len(scenario.script.requests) == calls


def test_render_only_survives_a_render_that_raises_permission_error(
    scenario: Scenario,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = _complete_with_stale_and_missing(scenario, capsys)
    calls = len(scenario.script.requests)
    _patch_render(monkeypatch, held)

    outcome = scenario.main(capsys)

    _assert_survived(scenario, outcome, held)
    assert len(scenario.script.requests) == calls
