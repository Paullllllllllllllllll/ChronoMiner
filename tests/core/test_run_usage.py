"""Model calls of a whole run: attempt contexts and per-role token totals."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import chronominer.run as core
from chronominer.common.testing import FakeRequest
from chronominer.common.usage import Usage
from tests.conftest import make_settings, make_spec
from tests.llm import helpers
from tests.llm.helpers import RecordingExt, StatusError

RECOVERED = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
ANSWER_USAGE = Usage(input_tokens=100, output_tokens=20, total_tokens=120)
ANSWER: dict[str, Any] = {
    "contains_no_content_of_requested_type": False,
    "entries": [],
}
_REAL_SLEEP = asyncio.sleep
HEARTBEAT_TASK = "lease-heartbeat"


def _text(path: Path, chunks: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{n}. John Carter, mason, 2s 6d\n" for n in range(chunks)),
        encoding="utf-8",
    )
    path.with_name(f"{path.stem}_line_ranges.txt").write_text(
        "".join(f"({n}, {n})\n" for n in range(1, chunks + 1)), encoding="utf-8"
    )
    return path


def _responder() -> Callable[[FakeRequest], Any]:
    failed: list[bool] = []

    def answer(request: FakeRequest) -> Any:
        if not failed:
            failed.append(True)
            return StatusError(408, RECOVERED)
        return ANSWER

    return answer


async def _yield_only(delay: float, result: Any = None) -> Any:
    """Only yield, except in the lease heartbeat, which keeps its interval."""
    task = asyncio.current_task()
    if task is not None and task.get_name() == HEARTBEAT_TASK:
        return await _REAL_SLEEP(delay, result)
    await _REAL_SLEEP(0)
    return result


@pytest.fixture
def scripted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[core.RunResult, RecordingExt]:
    """Run a text of two chunks; the first extraction attempt fails."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(asyncio, "sleep", _yield_only)
    recorder = RecordingExt().install(monkeypatch)
    helpers.install(monkeypatch, [_responder()])
    settings = make_settings(rate_limits=((100_000, 1),))
    spec = make_spec(
        _text(tmp_path / "in" / "ledger.txt", chunks=2),
        output=tmp_path / "out",
        chunking="line-ranges",
        concurrency=1,
    )
    result = asyncio.run(core.run(core.plan(spec, settings)))
    return result, recorder


def test_every_model_call_runs_in_an_attempt_context(
    scripted_run: tuple[core.RunResult, RecordingExt],
) -> None:
    result, recorder = scripted_run

    assert result.complete == 1
    assert recorder.roles == ["extract"] * 3
    assert all(attempt.released for attempt in recorder.attempts)
    assert [type(error) for error in recorder.errors] == [StatusError]
    assert recorder.commits == [ANSWER_USAGE] * 2
    failed = next(a for a in recorder.attempts if a.errors)
    assert failed.identity.role == "extract"
    assert failed.recovered == Usage(8, 2, 10)


def test_the_run_result_holds_token_totals_per_role(
    scripted_run: tuple[core.RunResult, RecordingExt],
) -> None:
    result, _ = scripted_run

    assert result.usage == {"extract": ANSWER_USAGE + ANSWER_USAGE + Usage(8, 2, 10)}
    assert result.total_usage.total_tokens == 250
