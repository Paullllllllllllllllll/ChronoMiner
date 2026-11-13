"""Several items open at once under one request limit (``--open-items``).

The scripted chat model of ``tests.llm.helpers`` answers every chunk; a
:class:`Flight` makes each call take a fixed delay and records how many calls,
and how many documents, are in flight at once. A call may also wait on a
gate, which fixes the order of concurrent calls where a test needs it.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pymupdf
import pytest

import chronominer.run as core
from chronominer import ext
from chronominer.common.contract import ExitCode
from chronominer.common.retry import Decision
from chronominer.common.store import Store, UnitKey, current_unit
from chronominer.common.testing import FakeRequest, ScriptedChatModel
from chronominer.events import ItemFinished
from chronominer.output.store import database_path
from chronominer.stopping import RunStop
from tests.conftest import make_settings, make_spec
from tests.core.test_run import ANSWER
from tests.llm import helpers
from tests.llm.helpers import StatusError
from tests.recording_events import RecordingEvents

DELAY = 0.05
"""Seconds every scripted call takes."""

WAIT = 5.0
"""Seconds a gated call waits before the test fails instead of hanging."""

Gate = Callable[[UnitKey | None], Awaitable[None]]


@pytest.fixture(autouse=True)
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variable of the provider the tests call."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


class Flight:
    """Make every scripted call take *delay* seconds and record what flies.

    *peak* is the most calls in flight at once, *documents* the most
    documents with a call in flight at once. *log* records ``("request",
    doc_id)`` as each call starts; *gate*, when set, is awaited by each call
    before it answers.
    """

    def __init__(self, delay: float = DELAY) -> None:
        self.delay = delay
        self.gate: Gate | None = None
        self.log: list[tuple[str, Any]] = []
        self.peak = 0
        self.documents = 0
        self._flying: Counter[int | None] = Counter()

    def install(self, monkeypatch: pytest.MonkeyPatch) -> Flight:
        """Patch the scripted model's asynchronous call."""
        respond = ScriptedChatModel._respond

        async def agenerate(
            model: ScriptedChatModel,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> Any:
            unit = current_unit()
            doc = unit.doc_id if unit is not None else None
            self.log.append(("request", doc))
            self._flying[doc] += 1
            self.peak = max(self.peak, sum(self._flying.values()))
            self.documents = max(
                self.documents, sum(1 for count in self._flying.values() if count)
            )
            try:
                await asyncio.sleep(self.delay)
                if self.gate is not None:
                    await asyncio.wait_for(self.gate(unit), WAIT)
                return respond(model, messages, kwargs)
            finally:
                self._flying[doc] -= 1

        monkeypatch.setattr(ScriptedChatModel, "_agenerate", agenerate)
        return self


class LoggedEvents(RecordingEvents):
    """Record the events and log each finished item in *flight*."""

    def __init__(self, flight: Flight) -> None:
        super().__init__()
        self.flight = flight

    def item_finished(self, event: ItemFinished) -> None:
        """Log the finished item, then record it."""
        self.flight.log.append(("finished", event.name))
        super().item_finished(event)


def _pdf(path: Path, pages: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    for index in range(pages):
        document.new_page(width=200, height=300).insert_text((50, 100), f"P{index}")
    document.save(path)
    document.close()
    return path


def _items(folder: Path, names: str, pages: int = 2) -> None:
    for name in names:
        _pdf(folder / f"{name}.pdf", pages)


def _query(root: Path, sql: str) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(database_path(root))
    try:
        return [tuple(row) for row in connection.execute(sql)]
    finally:
        connection.close()


def _statuses(root: Path) -> dict[str, list[str]]:
    rows = _query(
        root,
        "SELECT d.name, u.status FROM units AS u JOIN documents AS d USING (doc_id) "
        "WHERE u.kind = 'chunk' ORDER BY d.name, u.idx",
    )
    found: dict[str, list[str]] = {}
    for name, status in rows:
        found.setdefault(str(name), []).append(str(status))
    return found


def _held_leases(root: Path) -> int:
    ((count,),) = _query(
        root, "SELECT count(*) FROM documents WHERE lease_run_id IS NOT NULL"
    )
    return int(count)


def _names(root: Path) -> dict[int, str]:
    return {
        int(doc_id): str(name)
        for doc_id, name in _query(root, "SELECT doc_id, name FROM documents")
    }


def _run(tmp_path: Path, events: Any = None, **flags: Any) -> core.RunResult:
    spec = make_spec(tmp_path / "in", output=tmp_path / "out", **flags)
    settings = make_settings(rate_limits=((100_000, 1),))
    return asyncio.run(core.run(core.plan(spec, settings), events))


def _answer(request: FakeRequest) -> Any:
    return ANSWER


def _barrier(calls: int) -> Gate:
    """Hold the first *calls* calls until all of them are in flight."""
    arrived = [0]
    full = asyncio.Event()

    async def gate(unit: UnitKey | None) -> None:
        arrived[0] += 1
        if arrived[0] >= calls:
            full.set()
        if arrived[0] <= calls:
            await full.wait()

    return gate


def _pairs() -> Gate:
    """Hold each call until the other chunk of its document is in flight."""
    arrived: Counter[int | None] = Counter()
    events: dict[int | None, asyncio.Event] = {}

    async def gate(unit: UnitKey | None) -> None:
        doc = unit.doc_id if unit is not None else None
        arrived[doc] += 1
        event = events.setdefault(doc, asyncio.Event())
        if arrived[doc] >= 2:
            event.set()
        await event.wait()

    return gate


def test_open_items_share_the_limit_and_report_in_item_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _items(tmp_path / "in", "abcdef")
    helpers.install(monkeypatch, [_answer])
    flight = Flight().install(monkeypatch)
    flight.gate = _barrier(4)

    result = _run(tmp_path, concurrency=4, open_items=3)

    assert result.exit_code == ExitCode.OK
    assert flight.peak == 4
    assert flight.documents >= 2
    assert [item.name for item in result.items] == [f"{n}.pdf" for n in "abcdef"]
    statuses = _statuses(tmp_path / "out")
    assert statuses == {name: ["done", "done"] for name in "abcdef"}


def test_one_open_item_runs_the_items_one_after_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _items(tmp_path / "in", "abcd")
    helpers.install(monkeypatch, [_answer])
    flight = Flight().install(monkeypatch)
    flight.gate = _pairs()
    events = LoggedEvents(flight)

    result = _run(tmp_path, events, concurrency=4, open_items=1)

    assert (result.exit_code, flight.peak, flight.documents) == (ExitCode.OK, 2, 1)
    names = _names(tmp_path / "out")
    log = [
        f"{kind}:{names[value] if kind == 'request' else value.removesuffix('.pdf')}"
        for kind, value in flight.log
    ]
    for earlier, later in zip("abc", "bcd", strict=True):
        assert log.index(f"request:{later}") > log.index(f"finished:{earlier}")


class StopAt:
    """Stop the run at the second chunk of the first item, the second item open.

    That call fails once both chunks of the second item are in flight, and
    its attempt answers with STOP; the calls held until then answer after
    the stop.
    """

    def __init__(self) -> None:
        self.active = True
        self.stopped = asyncio.Event()
        self.entered: set[int] = set()
        self.second_open = asyncio.Event()
        self.decisions = 0

    async def gate(self, unit: UnitKey | None) -> None:
        if not self.active or unit is None:
            return
        if unit.doc_id == 2:
            self.entered.add(unit.idx)
            if self.entered == {0, 1}:
                self.second_open.set()
            await self.stopped.wait()
        elif (unit.doc_id, unit.idx) == (1, 1):
            await self.second_open.wait()
        elif unit.doc_id == 1:
            await self.stopped.wait()

    def respond(self, request: FakeRequest) -> Any:
        unit = current_unit()
        if self.active and unit is not None and (unit.doc_id, unit.idx) == (1, 1):
            return StatusError(429)
        return ANSWER

    def install(self, monkeypatch: pytest.MonkeyPatch) -> StopAt:
        real = ext.attempt

        def attempt(
            role: str, provider: str, estimate: int | None = None, **kwargs: Any
        ) -> ext.Attempt[Any]:
            opened: ext.Attempt[Any] = real(role, provider, estimate, **kwargs)
            on_error = opened.on_error

            def deciding(exc: BaseException) -> Decision:
                decision = on_error(exc)
                if not self.active:
                    return decision
                self.decisions += 1
                self.stopped.set()
                return Decision.STOP

            opened.on_error = deciding  # type: ignore[method-assign]
            return opened

        monkeypatch.setattr(ext, "attempt", attempt)
        return self


def test_a_stop_in_one_open_item_opens_no_further_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _items(tmp_path / "in", "abc")
    out = tmp_path / "out"
    stop = StopAt().install(monkeypatch)
    helpers.install(monkeypatch, [stop.respond])
    flight = Flight().install(monkeypatch)
    flight.gate = stop.gate

    result = _run(tmp_path, concurrency=4, open_items=2)

    assert result.exit_code == ExitCode.STOPPED
    assert stop.decisions == 1
    assert _names(out) == {1: "a", 2: "b"}
    assert {doc for kind, doc in flight.log if kind == "request"} == {1, 2}
    assert result.failed == 0
    assert result.stop is not None
    assert (result.stop.name, result.stop.units, result.stop.further) == (
        "a.pdf",
        1,
        1,
    )
    assert [item.name for item in result.items] == ["b.pdf"]
    assert _statuses(out) == {"a": ["done", "pending"], "b": ["done", "done"]}
    assert _held_leases(out) == 0

    stop.active = False
    second = _run(tmp_path, concurrency=4, open_items=2)

    assert (second.exit_code, second.stop) == (ExitCode.OK, None)
    assert _statuses(out) == {name: ["done", "done"] for name in "abc"}


def test_an_item_that_opens_as_the_run_stops_keeps_its_stored_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _items(tmp_path / "in", "ab")
    out = tmp_path / "out"
    helpers.install(monkeypatch, [_answer])
    Flight().install(monkeypatch)
    assert _run(tmp_path, concurrency=4, open_items=2).exit_code == ExitCode.OK
    stopped = asyncio.Event()
    held: list[str] = []
    real, lease = ext.attempt, core._ItemRun.lease

    def attempt(
        role: str, provider: str, estimate: int | None = None, **kwargs: Any
    ) -> ext.Attempt[Any]:
        opened: ext.Attempt[Any] = real(role, provider, estimate, **kwargs)

        def deciding(exc: BaseException) -> Decision:
            stopped.set()
            return Decision.STOP

        opened.on_error = deciding  # type: ignore[method-assign]
        return opened

    async def late(self: Any) -> Any:
        # The second item has its lease and waits here while the first stops
        # the run.
        leased = await lease(self)
        if self.planned.name == "b.pdf":
            held.append(self.planned.name)
            await asyncio.wait_for(stopped.wait(), WAIT)
        return leased

    monkeypatch.setattr(ext, "attempt", attempt)
    monkeypatch.setattr(core._ItemRun, "lease", late)
    helpers.install(monkeypatch, [lambda request: StatusError(429)])

    result = _run(tmp_path, concurrency=4, open_items=2, force=True)

    assert result.exit_code == ExitCode.STOPPED
    assert held == ["b.pdf"]
    assert _statuses(out)["b"] == ["done", "done"]
    assert _held_leases(out) == 0


def test_an_item_that_fails_leaves_the_others_to_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _items(tmp_path / "in", "abcd")

    def respond(request: FakeRequest) -> Any:
        unit = current_unit()
        return StatusError(400) if unit is not None and unit.doc_id == 2 else ANSWER

    helpers.install(monkeypatch, [respond])
    Flight().install(monkeypatch)

    result = _run(tmp_path, concurrency=4, open_items=3)

    assert result.exit_code == ExitCode.FAILURE
    assert _names(tmp_path / "out")[2] == "b"
    assert [(item.name, item.success) for item in result.items] == [
        ("a.pdf", True),
        ("b.pdf", False),
        ("c.pdf", True),
        ("d.pdf", True),
    ]
    assert result.complete == 3


def test_open_items_of_one_output_root_share_one_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _items(tmp_path / "in", "ab")
    helpers.install(monkeypatch, [_answer])
    flight = Flight().install(monkeypatch)
    flight.gate = _barrier(4)
    opened: list[Path] = []
    real = core.open_store

    def counting(root: Path, **kwargs: Any) -> Store:
        opened.append(root)
        return real(root, **kwargs)

    monkeypatch.setattr(core, "open_store", counting)

    result = _run(tmp_path, concurrency=4, open_items=2)

    assert (result.exit_code, result.complete) == (ExitCode.OK, 2)
    assert flight.documents == 2
    assert len(opened) == 1
    assert len(result.databases) == 1
    assert _statuses(tmp_path / "out") == {"a": ["done", "done"], "b": ["done", "done"]}


def test_cancelling_the_run_releases_the_leases_of_the_open_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _items(tmp_path / "in", "abc")
    out = tmp_path / "out"
    helpers.install(monkeypatch, [_answer])
    flight = Flight().install(monkeypatch)
    both = asyncio.Event()
    entered: set[int] = set()

    async def hold(unit: UnitKey | None) -> None:
        if unit is not None:
            entered.add(unit.doc_id)
            if len(entered) == 2:
                both.set()
        await asyncio.Event().wait()

    flight.gate = hold
    released: list[int] = []
    real_release = Store.release

    async def release(self: Store, lease: Any) -> None:
        released.append(lease.doc_id)
        await real_release(self, lease)

    monkeypatch.setattr(Store, "release", release)
    spec = make_spec(tmp_path / "in", output=out, concurrency=4, open_items=2)
    run_plan = core.plan(spec, make_settings(rate_limits=((100_000, 1),)))

    async def cancel_when_two_are_open() -> None:
        task = asyncio.ensure_future(core.run(run_plan))
        await asyncio.wait_for(both.wait(), WAIT)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_when_two_are_open())

    assert sorted(released) == [1, 2]
    assert _held_leases(out) == 0
    assert _query(out, "SELECT exit_code FROM runs") == [(int(ExitCode.INTERRUPTED),)]


def test_a_stop_counts_the_units_left_over_every_item_in_work() -> None:
    stop = RunStop.of("r", [("a.pdf", 2), ("b.pdf", 1)], further=3, noun="chunk")

    assert stop.items == 5
    assert stop.message() == (
        "Run stopped (r): 3 chunk(s) of 2 item(s) in work and 3 further item(s) "
        "were left unprocessed; a rerun resumes them."
    )


def test_open_items_have_no_effect_on_a_single_item(tmp_path: Path) -> None:
    _items(tmp_path / "in", "a")
    spec = make_spec(tmp_path / "in", output=tmp_path / "out", open_items=4)

    run_plan = core.plan(spec, make_settings(), require_keys=False)

    assert any(
        "--open-items applies only to a run of several items" in warning
        and "the run processes 1 item(s)" in warning
        for warning in run_plan.warnings
    )
