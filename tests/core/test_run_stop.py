"""A run stop: a provider that serves no further calls ends the run with exit 3.

An attempt whose ``on_error`` returns ``Decision.STOP`` stands in for the
extension that makes this decision; the scripted chat model of
``tests.llm.helpers`` or the fake decision API answers. :class:`Stopper`
makes the order of concurrent calls deterministic: the call of the stopping
unit fails only once the units meant to be in flight have entered their
attempts, and those send their requests after the stop.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import httpx
import pymupdf
import pytest

import chronominer.run as core
from chronominer import __main__ as entry
from chronominer import ext
from chronominer.common.contract import ExitCode
from chronominer.common.retry import Decision
from chronominer.common.store import current_unit
from chronominer.common.testing import FakeRequest
from chronominer.llm import decision_model
from chronominer.output.store import database_path
from tests.batch.test_repair_store import store_document
from tests.characterization.inputs import make_text_file, write_ranges
from tests.classify.conftest import DecisionServer, write_text
from tests.cli.conftest import json_lines
from tests.conftest import DEFAULT_TEST_SCHEMA, make_settings, make_spec
from tests.core.test_run import ANSWER
from tests.llm import helpers
from tests.llm.helpers import StatusError
from tests.ranges.fakes import default_answer
from tests.recording_events import RecordingEvents

REASON = "StatusError: status 429"
MID_RECORD = [(1, 20), (21, 38), (39, 56), (57, 71)]
"""Line ranges that start inside the records of a twelve-record ledger."""


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variables of the providers the tests call."""
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(name, "test-key")


class Stopper:
    """Fail the call of unit *stop* with a 429 that its attempt answers with STOP.

    The units in *held* enter their attempts before that call fails and send
    their requests after it. *requests* records the unit index of every
    request (None outside a unit); *decisions* counts the STOP decisions.
    Set *active* to False for a run without a stop.
    """

    def __init__(self, stop: int | None = 0, held: Iterable[int] = ()) -> None:
        self.stop = stop
        self.held = set(held)
        self.active = True
        self.requests: list[int | None] = []
        self.decisions = 0
        self._arrived: set[int] = set()
        self._events: tuple[asyncio.Event, asyncio.Event] | None = None

    def _pair(self) -> tuple[asyncio.Event, asyncio.Event]:
        if self._events is None:
            self._events = (asyncio.Event(), asyncio.Event())
            if not self.held:
                self._events[0].set()
        return self._events

    async def _enter(self) -> None:
        unit = current_unit()
        idx = unit.idx if unit is not None else None
        all_held, stopped = self._pair()
        if idx in self.held:
            self._arrived.add(idx)
            if self._arrived >= self.held:
                all_held.set()
            await stopped.wait()
        elif idx == self.stop:
            await all_held.wait()

    def install(self, monkeypatch: pytest.MonkeyPatch) -> Stopper:
        """Wrap ``ext.attempt`` so every attempt follows the choreography."""
        real = ext.attempt

        def attempt(
            role: str, provider: str, estimate: int | None = None, **kwargs: Any
        ) -> ext.Attempt[Any]:
            opened: ext.Attempt[Any] = real(role, provider, estimate, **kwargs)
            on_error, refresh = opened.on_error, opened.refresh

            async def entering() -> Any:
                if self.active:
                    await self._enter()
                return await refresh()

            def deciding(exc: BaseException) -> Decision:
                decision = on_error(exc)
                if not self.active:
                    return decision
                self.decisions += 1
                self._pair()[1].set()
                return Decision.STOP

            opened.refresh = entering  # type: ignore[method-assign]
            opened.on_error = deciding  # type: ignore[method-assign]
            return opened

        monkeypatch.setattr(ext, "attempt", attempt)
        return self

    def answer(self, reply: Callable[[FakeRequest], Any]) -> Callable[..., Any]:
        """Return a responder: a 429 for the stopping unit, else *reply*."""

        def respond(request: FakeRequest) -> Any:
            unit = current_unit()
            idx = unit.idx if unit is not None else None
            self.requests.append(idx)
            if self.active and idx == self.stop:
                return StatusError(429)
            return reply(request)

        return respond


def _pdf(path: Path, pages: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    for index in range(pages):
        document.new_page(width=200, height=300).insert_text((50, 100), f"P{index}")
    document.save(path)
    document.close()
    return path


def _query(root: Path, sql: str) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(database_path(root))
    try:
        return [tuple(row) for row in connection.execute(sql)]
    finally:
        connection.close()


def _units(root: Path, kind: str = "chunk") -> list[tuple[str, int, str]]:
    return [
        (str(name), int(idx), str(status))
        for name, idx, status in _query(
            root,
            "SELECT d.name, u.idx, u.status FROM units AS u JOIN documents AS d "
            f"USING (doc_id) WHERE u.kind = '{kind}' ORDER BY d.name, u.idx",
        )
    ]


def _run_codes(root: Path, command: str) -> list[int]:
    return [
        int(code)
        for (code,) in _query(
            root, f"SELECT exit_code FROM runs WHERE command = '{command}'"
        )
    ]


def _one_line(capsys: pytest.CaptureFixture[str]) -> tuple[dict[str, Any], str]:
    captured = capsys.readouterr()
    (payload,) = json_lines(captured.out)
    return payload, captured.err


def _answer(request: FakeRequest) -> Any:
    return ANSWER


def _boundary(request: FakeRequest) -> Any:
    return default_answer(str(request.messages[-1].content), "")


@pytest.mark.usefixtures("keys")
def test_a_stop_ends_extraction_and_a_rerun_resumes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pdf(tmp_path / "in" / "a.pdf", 4)
    _pdf(tmp_path / "in" / "b.pdf", 2)
    out = tmp_path / "out"
    stopper = Stopper(stop=1, held=(0,)).install(monkeypatch)
    helpers.install(monkeypatch, [stopper.answer(_answer)])
    spec = make_spec(tmp_path / "in", output=out, concurrency=2)
    settings = make_settings(rate_limits=((100_000, 1),))
    events = RecordingEvents()

    result = asyncio.run(core.run(core.plan(spec, settings), events))

    assert result.exit_code == ExitCode.STOPPED
    assert result.stop is not None and result.stopped == REASON
    assert (result.stop.name, result.stop.units, result.stop.further) == (
        "a.pdf",
        3,
        1,
    )
    assert (result.items, result.failed) == ((), 0)
    assert sorted(stopper.requests) == [0, 1]
    assert stopper.decisions == 1
    assert _units(out) == [
        ("a", 0, "done"),
        ("a", 1, "pending"),
        ("a", 2, "pending"),
        ("a", 3, "pending"),
    ]
    assert _query(out, "SELECT count(*) FROM responses WHERE idx = 1") == [(1,)]
    assert (out / "a_output.json").exists()
    assert [event.name for event in events.of("item_started")] == ["a.pdf"]
    assert events.of("item_finished") == []
    assert _run_codes(out, "extract") == [ExitCode.STOPPED]

    stopper.active = False
    again = core.plan(spec, settings)
    assert again.blocked == ()
    second = asyncio.run(core.run(again))

    assert (second.exit_code, second.complete, second.stop) == (ExitCode.OK, 2, None)
    assert len(stopper.requests) == 2 + 3 + 2
    assert {status for _, _, status in _units(out)} == {"done"}


@pytest.mark.usefixtures("keys")
def test_a_stop_outlasts_an_item_that_also_lost_its_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pdf(tmp_path / "in" / "a.pdf", 2)
    _pdf(tmp_path / "in" / "b.pdf", 2)
    helpers.install(monkeypatch, [_answer])

    async def extract(self: Any, *_args: Any) -> tuple[tuple[str, ...], str | None]:
        self.job.env.stop.set(REASON)
        return (), "its lease was lost: another run took the document"

    monkeypatch.setattr(core._ItemRun, "extract", extract)
    spec = make_spec(tmp_path / "in", output=tmp_path / "out", concurrency=1)
    events = RecordingEvents()

    result = asyncio.run(core.run(core.plan(spec, make_settings()), events))

    assert result.exit_code == ExitCode.STOPPED
    assert result.stop is not None
    assert (result.stopped, result.stop.name, result.stop.further) == (REASON, None, 1)
    assert result.failed == 1
    assert [event.name for event in events.of("item_started")] == ["a.pdf"]


@pytest.mark.usefixtures("keys")
def test_the_extract_command_exits_3_and_names_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _pdf(tmp_path / "in" / "a.pdf", 2)
    _pdf(tmp_path / "in" / "b.pdf", 1)
    stopper = Stopper(stop=0).install(monkeypatch)
    helpers.install(monkeypatch, [stopper.answer(_answer)])
    argv = ["extract", "--input", str(tmp_path / "in"), "--all"]
    argv += ["--schema", DEFAULT_TEST_SCHEMA, "--output", str(tmp_path / "out")]

    code = entry.main([*argv, "--concurrency", "1", "--json"])

    payload, err = _one_line(capsys)
    assert code == ExitCode.STOPPED
    assert payload["stopped"] == REASON
    assert (payload["items_failed"], payload["items_complete"]) == (0, 0)
    assert stopper.requests == [0]
    assert (
        f"Run stopped ({REASON}): 2 chunk(s) of 'a.pdf' and 1 further item(s) were "
        "left unprocessed; a rerun resumes them." in " ".join(err.split())
    )


@pytest.mark.usefixtures("keys")
def test_a_stop_leaves_the_range_undecided_and_starts_no_further_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "in"
    write_ranges(make_text_file(folder / "a.txt", 12), MID_RECORD)
    write_ranges(make_text_file(folder / "b.txt", 12), MID_RECORD[:2])
    stopper = Stopper(stop=1).install(monkeypatch)
    helpers.install(monkeypatch, [stopper.answer(_boundary)])
    argv = ["ranges", "adjust", "--input", str(folder), "--schema", DEFAULT_TEST_SCHEMA]

    code = entry.main([*argv, "--concurrency", "1", "--json"])

    payload, err = _one_line(capsys)
    assert code == ExitCode.STOPPED
    assert payload["stopped"] == REASON
    assert (payload["items_failed"], payload["items_total"]) == (0, 0)
    assert set(stopper.requests) <= {0, 1}
    assert stopper.requests[-1] == 1
    assert _units(folder, "range") == [
        ("a", 0, "done"),
        ("a", 1, "pending"),
        ("a", 2, "pending"),
        ("a", 3, "pending"),
    ]
    assert "3 line range(s) of 'a.txt' and 1 further item(s)" in " ".join(err.split())
    assert _run_codes(folder, "ranges adjust") == [ExitCode.STOPPED]

    stopper.active = False
    assert entry.main([*argv, "--json"]) == ExitCode.OK
    payload, _err = _one_line(capsys)
    assert (payload["items_complete"], payload["stopped"]) == (2, None)
    assert {status for _, _, status in _units(folder, "range")} == {"done"}


@pytest.mark.usefixtures("keys")
def test_a_stopped_repair_keeps_its_units_and_starts_no_further_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    failed = ["failed", "failed"]
    first = asyncio.run(store_document(tmp_path, failed, kind="text"))
    second = asyncio.run(store_document(tmp_path, failed, kind="text", folder="out2"))
    stopper = Stopper(stop=0, held=(1,)).install(monkeypatch)
    helpers.install(monkeypatch, [stopper.answer(_answer)])
    argv = ["repair", "--input", str(tmp_path), "--json"]

    code = entry.main(argv)

    payload, err = _one_line(capsys)
    assert code == ExitCode.STOPPED
    assert (payload["stopped"], payload["repaired"], payload["failed"]) == (
        REASON,
        1,
        0,
    )
    assert sorted(stopper.requests) == [0, 1]
    assert _units(first.root) == [("doc", 0, "pending"), ("doc", 1, "done")]
    assert _units(second.root) == [("doc", 0, "failed"), ("doc", 1, "failed")]
    assert _run_codes(first.root, "repair") == [ExitCode.STOPPED]
    assert _run_codes(second.root, "repair") == []
    assert "1 chunk(s) of 'doc' and 1 further item(s)" in " ".join(err.split())
    output = json.loads(first.output.read_text(encoding="utf-8"))
    assert len(output["records"]) == 1

    stopper.active = False
    assert entry.main(argv) == ExitCode.OK
    payload, _err = _one_line(capsys)
    assert (payload["repaired"], payload["stopped"]) == (2, None)


@pytest.mark.usefixtures("keys")
def test_a_stopped_classify_run_records_nothing_and_a_rerun_classifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("a.txt", "b.txt"):
        write_text(tmp_path / "in" / name)
    stopper = Stopper(stop=None).install(monkeypatch)
    server = DecisionServer()

    def handler(request: httpx.Request) -> httpx.Response:
        if not stopper.active:
            return server.handler(request)
        server.requests.append(json.loads(request.content))
        return httpx.Response(429, json={"error": {"message": "too many requests"}})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        decision_model.httpx,
        "AsyncClient",
        lambda *a, **k: real(transport=httpx.MockTransport(handler)),
    )
    argv = ["classify", "chunks", "--classifier", "language", "--input"]
    argv += [str(tmp_path / "in"), "--all", "--output", str(tmp_path / "out")]
    argv += ["--chunk-tokens", "200", "--json"]

    code = entry.main(argv)

    payload, err = _one_line(capsys)
    assert code == ExitCode.STOPPED
    assert payload["stopped"] is not None and "429" in payload["stopped"]
    assert (payload["items_failed"], payload["items_classified"]) == (0, 0)
    assert (len(server.requests), stopper.decisions) == (1, 1)
    assert "'a.txt' and 1 further item(s) were left unprocessed" in err
    assert _query(tmp_path / "out", "SELECT count(*) FROM classifications") == [(0,)]
    assert _run_codes(tmp_path / "out", "classify chunks") == [ExitCode.STOPPED]

    stopper.active = False
    assert entry.main(argv) == ExitCode.OK
    payload, _err = _one_line(capsys)
    assert (payload["items_classified"], payload["stopped"]) == (2, None)
