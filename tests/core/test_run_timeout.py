"""A chunk whose call times out on every attempt ends the run with exit 4.

The scripted chat model of ``tests.llm.helpers`` raises a read timeout for
page 1 of ``a.pdf`` and answers every other request; the retry waits are
zero. The other chunks and documents are extracted and stored, and a rerun
asks only for the chunk that timed out.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pymupdf
import pytest

import chronominer.run as core
from chronominer import __main__ as entry
from chronominer import ext
from chronominer.common.contract import ExitCode
from chronominer.common.retry import Decision, RetryPolicy
from chronominer.common.store import current_unit
from chronominer.common.testing import FakeRequest
from chronominer.output.store import database_path
from chronominer.settings import RetrySettings
from tests.cli.conftest import json_lines
from tests.conftest import DEFAULT_TEST_SCHEMA, make_settings, make_spec
from tests.core.test_run import ANSWER
from tests.llm import helpers
from tests.llm.helpers import StatusError


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variable and make every retry wait zero seconds."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(RetryPolicy, "delay", lambda *_args, **_kwargs: 0.0)


class SlowPage:
    """Time out every request for unit *idx*; answer the others."""

    def __init__(self, idx: int = 1) -> None:
        self.idx = idx
        self.active = True
        self.requests: list[int | None] = []

    def __call__(self, request: FakeRequest) -> Any:
        unit = current_unit()
        idx = unit.idx if unit is not None else None
        self.requests.append(idx)
        if self.active and idx == self.idx:
            return httpx.ReadTimeout("Request timed out.")
        return ANSWER


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


def _units(root: Path) -> list[tuple[str, int, str, str | None]]:
    return [
        (str(name), int(idx), str(status), error_type)
        for name, idx, status, error_type in _query(
            root,
            "SELECT d.name, u.idx, u.status, u.error_type FROM units AS u "
            "JOIN documents AS d USING (doc_id) ORDER BY d.name, u.idx",
        )
    ]


def _inputs(tmp_path: Path) -> Path:
    _pdf(tmp_path / "in" / "a.pdf", 3)
    _pdf(tmp_path / "in" / "b.pdf", 1)
    return tmp_path / "in"


@pytest.mark.usefixtures("keys")
def test_a_chunk_that_times_out_throughout_exits_4_and_keeps_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    slow = SlowPage(idx=1)
    helpers.install(monkeypatch, [slow])
    spec = make_spec(_inputs(tmp_path), output=out, concurrency=1)
    settings = make_settings(
        rate_limits=((100_000, 1),), retry=RetrySettings(timeout_attempts=2)
    )

    result = asyncio.run(core.run(core.plan(spec, settings)))

    assert result.exit_code == core.TIMED_OUT
    assert [name for name, _label in result.timed_out] == ["a.pdf"]
    assert slow.requests.count(1) == 2
    assert _units(out) == [
        ("a", 0, "done", None),
        ("a", 1, "failed", "call_timeout"),
        ("a", 2, "done", None),
        ("b", 0, "done", None),
    ]
    assert _query(out, "SELECT exit_code FROM runs") == [(core.TIMED_OUT,)]
    (failed,) = [item for item in result.items if not item.success]
    assert failed.error is not None
    assert "the call timed out on every attempt for chunk(s)" in failed.error

    slow.active, sent = False, len(slow.requests)
    second = asyncio.run(core.run(core.plan(spec, settings)))

    assert (second.exit_code, second.timed_out) == (ExitCode.OK, ())
    assert slow.requests[sent:] == [1]


@pytest.mark.usefixtures("keys")
def test_a_provider_stop_wins_over_a_timed_out_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    slow = SlowPage(idx=1)

    def respond(request: FakeRequest) -> Any:
        # a.pdf sends four requests (page 1 twice); b.pdf's page meets the stop.
        return StatusError(429) if len(slow.requests) == 4 else slow(request)

    real = ext.attempt

    def attempt(*args: Any, **kwargs: Any) -> ext.Attempt[Any]:
        opened: ext.Attempt[Any] = real(*args, **kwargs)
        on_error = opened.on_error

        def deciding(exc: BaseException) -> Decision:
            return Decision.STOP if isinstance(exc, StatusError) else on_error(exc)

        opened.on_error = deciding  # type: ignore[method-assign]
        return opened

    monkeypatch.setattr(ext, "attempt", attempt)
    helpers.install(monkeypatch, [respond])
    spec = make_spec(_inputs(tmp_path), output=out, concurrency=1)
    settings = make_settings(
        rate_limits=((100_000, 1),), retry=RetrySettings(timeout_attempts=2)
    )

    result = asyncio.run(core.run(core.plan(spec, settings)))

    assert result.exit_code == ExitCode.STOPPED
    assert [name for name, _label in result.timed_out] == ["a.pdf"]
    assert _units(out) == [
        ("a", 0, "done", None),
        ("a", 1, "failed", "call_timeout"),
        ("a", 2, "done", None),
        ("b", 0, "pending", None),
    ]
    assert _query(out, "SELECT exit_code FROM runs") == [(ExitCode.STOPPED,)]


@pytest.mark.usefixtures("keys")
def test_a_timeout_that_a_retry_cures_is_no_timeout_of_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = iter([httpx.ReadTimeout("Request timed out."), ANSWER])

    def respond(request: FakeRequest) -> Any:
        return next(answers, ANSWER)

    helpers.install(monkeypatch, [respond])
    spec = make_spec(_inputs(tmp_path), output=tmp_path / "out", concurrency=1)

    result = asyncio.run(core.run(core.plan(spec, make_settings())))

    assert (result.exit_code, result.timed_out) == (ExitCode.OK, ())


@pytest.mark.usefixtures("keys")
def test_the_extract_command_exits_4_and_names_the_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    helpers.install(monkeypatch, [SlowPage(idx=1)])
    argv = ["extract", "--input", str(_inputs(tmp_path)), "--all"]
    argv += ["--schema", DEFAULT_TEST_SCHEMA, "--output", str(tmp_path / "out")]

    code = entry.main([*argv, "--concurrency", "1", "--json"])

    captured = capsys.readouterr()
    (payload,) = json_lines(captured.out)
    err = " ".join(captured.err.split())
    assert code == core.TIMED_OUT
    assert (payload["items_failed"], payload["items_complete"]) == (1, 1)
    assert "'a.pdf': the call of chunk " in err
    assert "timed out on every attempt; a rerun would wait as long again" in err
