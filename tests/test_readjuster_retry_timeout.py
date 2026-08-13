"""Retry, timeout-budget and ceiling behaviour of the boundary LLM call.

Covers ``LineRangeReadjuster._run_model``: a hung or transiently failing
provider call must not stall or abort a whole file's readjustment, while a
non-transient error (bad key, schema violation) must still fail loudly.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from modules.line_ranges import readjuster as readjuster_mod
from modules.line_ranges.readjuster import LineRangeReadjuster

pytestmark = pytest.mark.unit


_RAW_LINES = ["Line one", "Line two", "Line three", "Line four"]

_DECISION = {
    "contains_no_semantic_boundary": False,
    "needs_more_context": False,
    "boundary_already_on_target": True,
    "certainty": 95,
    "semantic_marker": "",
}


def _concurrency_config(
    *,
    chunk_timeout: Any = "off",
    attempts: int = 8,
    timeout_attempts: int = 3,
) -> dict[str, Any]:
    """Build a concurrency config with near-instant backoff waits."""
    return {
        "concurrency": {
            "extraction": {
                "timeouts": {"total": 900, "chunk_timeout": chunk_timeout},
                "retry": {
                    "attempts": attempts,
                    "timeout_attempts": timeout_attempts,
                    "wait_min_seconds": 0.001,
                    "wait_max_seconds": 0.001,
                    "jitter_max_seconds": 0.0,
                },
            }
        }
    }


def _make_readjuster(**kwargs: Any) -> LineRangeReadjuster:
    return LineRangeReadjuster(
        {"extraction_model": {"name": "gpt-4o-mini"}},
        context_window=2,
        **kwargs,
    )


async def _run(adjuster: LineRangeReadjuster) -> dict[str, Any]:
    return await adjuster._run_model(
        extractor=object(),  # type: ignore[arg-type]
        raw_lines=_RAW_LINES,
        original_range=(1, 4),
        context_window=(1, 4),
        window_index=0,
        boundary_type="BibliographicEntries",
        context=None,
    )


def _ok_payload() -> dict[str, Any]:
    return {"output_text": json.dumps(_DECISION)}


def _assert_unsure(payload: dict[str, Any]) -> None:
    assert payload["certainty"] == 0
    assert payload["contains_no_semantic_boundary"] is False
    assert payload["needs_more_context"] is False
    assert payload["boundary_already_on_target"] is False
    assert payload["semantic_marker"] == ""


@pytest.mark.asyncio
async def test_transient_connection_error_then_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One transport failure is retried; the second call's decision wins."""
    calls = 0

    async def _stub(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("connection refused")
        return _ok_payload()

    monkeypatch.setattr(readjuster_mod, "process_text_chunk", _stub)
    adjuster = _make_readjuster(concurrency_config=_concurrency_config())

    payload = await _run(adjuster)

    assert calls == 2
    assert payload["boundary_already_on_target"] is True
    assert payload["certainty"] == 95


@pytest.mark.asyncio
async def test_timeout_budget_exhausts_before_general_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pure timeouts are charged to the smaller timeout budget (3 of 8)."""
    calls = 0

    async def _stub(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("read timed out")

    monkeypatch.setattr(readjuster_mod, "process_text_chunk", _stub)
    adjuster = _make_readjuster(
        concurrency_config=_concurrency_config(attempts=8, timeout_attempts=3)
    )

    payload = await _run(adjuster)

    assert calls == 3
    _assert_unsure(payload)


@pytest.mark.asyncio
async def test_ceiling_overrun_returns_unsure_quickly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung call is cut off by the wall-clock ceiling, not by the SDK."""

    async def _stub(**_kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(10)
        return _ok_payload()

    monkeypatch.setattr(readjuster_mod, "process_text_chunk", _stub)
    adjuster = _make_readjuster(
        concurrency_config=_concurrency_config(chunk_timeout=0.2)
    )

    loop = asyncio.get_running_loop()
    started = loop.time()
    payload = await _run(adjuster)
    elapsed = loop.time() - started

    _assert_unsure(payload)
    assert elapsed < 5.0


@pytest.mark.asyncio
async def test_non_retryable_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An auth failure must surface, not degrade into an unsure window."""
    calls = 0

    async def _stub(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise Exception("Error code: 401 - invalid API key")

    monkeypatch.setattr(readjuster_mod, "process_text_chunk", _stub)
    adjuster = _make_readjuster(concurrency_config=_concurrency_config())

    with pytest.raises(Exception, match="401"):
        await _run(adjuster)

    assert calls == 1


@pytest.mark.asyncio
async def test_cancellation_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling the run tears the call down instead of retrying it."""
    started = asyncio.Event()

    async def _stub(**_kwargs: Any) -> dict[str, Any]:
        started.set()
        await asyncio.sleep(30)
        return _ok_payload()

    monkeypatch.setattr(readjuster_mod, "process_text_chunk", _stub)
    adjuster = _make_readjuster(concurrency_config=_concurrency_config())

    task = asyncio.create_task(_run(adjuster))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_constructor_without_concurrency_config_uses_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting the config falls back to the loader, keeping retries active."""

    class _StubLoader:
        def get_concurrency_config(self) -> dict[str, Any]:
            return _concurrency_config(attempts=8, timeout_attempts=2)

    monkeypatch.setattr(readjuster_mod, "get_config_loader", lambda: _StubLoader())

    calls = 0

    async def _stub(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("read timed out")

    monkeypatch.setattr(readjuster_mod, "process_text_chunk", _stub)
    adjuster = _make_readjuster()

    payload = await _run(adjuster)

    assert calls == 2
    _assert_unsure(payload)
