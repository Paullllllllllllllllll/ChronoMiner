"""The requests about one document go through the worker pool.

An asynchronous fake decision API holds chosen requests, so the tests fix
the order in which concurrent requests are answered and recorded. The
stored answers must not depend on that order.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from chronominer import __main__ as entry
from chronominer.classify import run as classify_run
from chronominer.common.contract import ExitCode
from chronominer.llm import decision_model
from tests.classify.conftest import DecisionServer, classifier_file, view, write_text
from tests.cli.conftest import json_lines
from tests.core.test_run_stop import Stopper

WAIT = 5.0
"""Seconds a held request waits before the test fails instead of hanging."""

_VOLATILE = ("run_id", "created_at")


class Gate:
    """An asynchronous decision API that answers through :class:`DecisionServer`.

    *hold* maps the 0-based arrival number of a request to an awaitable
    factory; the request is answered once it completes. *peak* is the most
    requests in flight at once.
    """

    def __init__(self) -> None:
        self.server = DecisionServer()
        self.hold: dict[int, Callable[[], Awaitable[Any]]] = {}
        self.fail: set[int] = set()
        self.arrived = 0
        self.flying = 0
        self.peak = 0

    async def handler(self, request: httpx.Request) -> httpx.Response:
        number = self.arrived
        self.arrived += 1
        self.flying += 1
        self.peak = max(self.peak, self.flying)
        try:
            if number in self.hold:
                await asyncio.wait_for(self.hold[number](), WAIT)
            if number in self.fail:
                self.server.requests.append(json.loads(request.content))
                return httpx.Response(429, json={"error": {"message": "slow down"}})
            return self.server.handler(request)
        finally:
            self.flying -= 1

    def install(self, monkeypatch: pytest.MonkeyPatch) -> Gate:
        """Route every decision request of the test to :meth:`handler`."""
        real = httpx.AsyncClient
        monkeypatch.setattr(
            decision_model.httpx,
            "AsyncClient",
            lambda *a, **k: real(transport=httpx.MockTransport(self.handler)),
        )
        return self


def _classifier(folder: Path) -> Path:
    """A chunk classifier that asks about one chunk per request."""
    return classifier_file(
        folder,
        {
            "name": "is_list",
            "version": 1,
            "unit": "chunk",
            "questions": [
                {"name": "list", "type": "predicate", "instructions": "A list."}
            ],
        },
    )


def _classify(
    capsys: pytest.CaptureFixture[str], text: Path, out: Path, *flags: str
) -> tuple[int, dict[str, Any]]:
    argv = ["classify", "chunks", "--classifier", str(_classifier(text.parent))]
    argv += ["--input", str(text), "--output", str(out), "--chunk-tokens", "40"]
    code = entry.main([*argv, *flags, "--json"])
    (summary,) = json_lines(capsys.readouterr().out)
    return code, summary


def _stable(root: Path) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key not in _VOLATILE}
        for row in view(root)
    ]


@pytest.fixture
def text(tmp_path: Path) -> Path:
    return write_text(tmp_path / "in" / "prices.txt")


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> Gate:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return Gate().install(monkeypatch)


def test_requests_fly_together_and_store_what_one_at_a_time_stores(
    tmp_path: Path, text: Path, gate: Gate, capsys: pytest.CaptureFixture[str]
) -> None:
    code, serial = _classify(capsys, text, tmp_path / "one", "--concurrency", "1")
    assert (code, gate.peak) == (0, 1)
    asked = serial["units_asked"]
    assert asked >= 4
    four = asyncio.Event()
    for number in range(4):
        gate.hold[asked + number] = four.wait
    gate.hold[asked + 3] = _set_then(four)

    code, pooled = _classify(capsys, text, tmp_path / "four", "--concurrency", "4")

    assert code == 0
    assert gate.peak == 4
    assert pooled["units_asked"] == asked
    assert _stable(tmp_path / "four") == _stable(tmp_path / "one")


def _set_then(event: asyncio.Event) -> Callable[[], Awaitable[None]]:
    async def release() -> None:
        event.set()

    return release


def _recorded_after(
    monkeypatch: pytest.MonkeyPatch, name: str, count: int
) -> asyncio.Event:
    """Return an event set once *count* calls of the run's writer *name* returned."""
    real = getattr(classify_run, name)
    done = asyncio.Event()
    calls = [0]

    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = await real(*args, **kwargs)
        calls[0] += 1
        if calls[0] >= count:
            done.set()
        return result

    monkeypatch.setattr(classify_run, name, wrapped)
    return done


def test_a_request_recorded_after_a_later_one_stores_the_same_answers(
    tmp_path: Path,
    text: Path,
    gate: Gate,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, serial = _classify(capsys, text, tmp_path / "one", "--concurrency", "1")
    assert code == 0
    later = _recorded_after(monkeypatch, "record_answers", 1)
    gate.hold[serial["units_asked"]] = later.wait

    code, pooled = _classify(capsys, text, tmp_path / "two", "--concurrency", "2")

    assert code == 0
    assert _stable(tmp_path / "two") == _stable(tmp_path / "one")
    assert pooled["units_asked"] == serial["units_asked"]


def test_a_request_asked_while_later_ones_are_reused_stores_the_same_answers(
    tmp_path: Path,
    text: Path,
    gate: Gate,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for out in ("one", "two"):
        code, _ = _classify(capsys, text, tmp_path / out, "--pages", "2-")
        assert code == 0
    code, serial = _classify(capsys, text, tmp_path / "one", "--concurrency", "1")
    assert (code, serial["units_asked"]) == (0, 1)
    reused = serial["units_reused"]
    assert reused >= 2
    republished = _recorded_after(monkeypatch, "republish_answers", reused)
    gate.hold[gate.arrived] = republished.wait

    code, pooled = _classify(capsys, text, tmp_path / "two", "--concurrency", "4")

    assert code == 0
    assert (pooled["units_asked"], pooled["units_reused"]) == (1, reused)
    assert _stable(tmp_path / "two") == _stable(tmp_path / "one")


def test_a_stop_lets_the_request_in_flight_finish_and_sends_no_further_one(
    tmp_path: Path,
    text: Path,
    gate: Gate,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopper = Stopper(stop=None).install(monkeypatch)
    gate.hold[0] = stopper._pair()[1].wait
    gate.fail.add(1)
    out = tmp_path / "out"

    code, summary = _classify(capsys, text, out, "--concurrency", "2")

    assert code == ExitCode.STOPPED
    assert summary["stopped"] is not None and "429" in summary["stopped"]
    assert stopper.decisions == 1
    assert gate.arrived == 2
    assert [row["ordinal"] for row in view(out)] == [1]

    stopper.active = False
    gate.fail.clear()
    gate.hold.clear()
    code, again = _classify(capsys, text, out, "--concurrency", "2")
    assert (code, again["stopped"], again["units_reused"]) == (0, None, 1)
    assert [row["ordinal"] for row in view(out)] == list(
        range(1, again["units_asked"] + 2)
    )
