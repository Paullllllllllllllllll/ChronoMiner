"""Tests for the per-chunk wall-clock watchdog and the timeout retry budget.

Covers the deadline that bounds all attempts and backoff sleeps of one
unit, the separate (smaller) budget charged to type-level timeouts, and
the classifier hole that made a bare ``httpx.ReadTimeout`` look
non-retryable. All offline (no API calls).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

import modules.extract.processing_strategy as ps


class _AsyncExtractorCM:
    def __init__(self, extractor: object) -> None:
        self._extractor = extractor

    async def __aenter__(self) -> object:
        return self._extractor

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _StubHandler:
    schema_name = "TestSchema"


class _SpyLimiter:
    """Stand-in for the shared adaptive rate limiter."""

    def __init__(self) -> None:
        self.successes = 0
        self.errors: list[bool] = []

    def report_success(self) -> None:
        self.successes += 1

    def report_error(self, is_rate_limit: bool = False) -> None:
        self.errors.append(is_rate_limit)


def _install_stubs(
    monkeypatch: pytest.MonkeyPatch, call: Any, limiter: _SpyLimiter | None = None
) -> _SpyLimiter:
    """Wire the strategy's collaborators to offline stubs.

    ``await_capacity`` is patched in the strategy's own namespace (it is
    imported there): the real one hops to a thread, which alone can
    outlast a sub-second ceiling and make the watchdog look flaky.
    """
    spy = limiter or _SpyLimiter()
    monkeypatch.setattr(
        ps.ProviderConfig, "_detect_provider", staticmethod(lambda model: "openai")
    )
    monkeypatch.setattr(
        ps.ProviderConfig, "_get_api_key", staticmethod(lambda provider: "key")
    )
    monkeypatch.setattr(
        ps, "open_extractor", lambda **_kwargs: _AsyncExtractorCM(object())
    )

    async def _no_capacity(_limiter: Any) -> None:
        return None

    monkeypatch.setattr(ps, "await_capacity", _no_capacity)
    monkeypatch.setattr(ps, "get_shared_rate_limiter", lambda _provider: spy)
    monkeypatch.setattr(ps, "process_text_chunk", call)
    return spy


def _strategy(
    *, retry: dict[str, Any], timeouts: dict[str, Any]
) -> ps.SynchronousProcessingStrategy:
    return ps.SynchronousProcessingStrategy(
        concurrency_config={
            "concurrency": {
                "extraction": {
                    "concurrency_limit": 1,
                    "retry": retry,
                    "timeouts": timeouts,
                }
            }
        }
    )


async def _run(
    strat: ps.SynchronousProcessingStrategy, tmp_path: Path
) -> list[dict[str, Any]]:
    return await strat.process_chunks(
        chunks=["c1"],
        handler=_StubHandler(),
        dev_message="dev",
        model_config={"extraction_model": {"name": "gpt-5-mini"}},
        schema={"type": "object"},
        file_path=tmp_path / "input.txt",
        temp_jsonl_path=tmp_path / "temp.jsonl",
        console_print=lambda *_a, **_k: None,
    )


_FAST_RETRY: dict[str, Any] = {
    "attempts": 8,
    "wait_min_seconds": 0.001,
    "wait_max_seconds": 0.01,
    "jitter_max_seconds": 0.0,
}


# ---------------------------------------------------------------------------
# Wall-clock ceiling
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_watchdog_aborts_a_hung_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A call that outlives the ceiling becomes a terminal chunk failure."""

    async def _hang(**_kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(10)
        return {"output_text": "never"}

    spy = _install_stubs(monkeypatch, _hang)
    strat = _strategy(
        retry={"attempts": 3, "wait_min_seconds": 0.001, "wait_max_seconds": 0.01},
        timeouts={"total": 60, "chunk_timeout": 0.2},
    )

    started = time.monotonic()
    results = await _run(strat, tmp_path)
    elapsed = time.monotonic() - started

    assert elapsed < 3.0
    assert len(results) == 1
    assert results[0]["chunk_index"] == 1
    assert "wall-clock ceiling" in results[0]["error"]
    assert "input chunk 1" in results[0]["error"]
    # The limiter is told about the unit that never completed.
    assert spy.errors == [False]
    assert spy.successes == 0


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", [False, 0, "off"])
async def test_watchdog_disabled_constructs_no_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, disabled: Any
) -> None:
    """Disabling values leave the call unbounded (no timeout scope at all)."""

    async def _ok(**_kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(0.05)
        return {"output_text": "done"}

    _install_stubs(monkeypatch, _ok)

    scopes = {"n": 0}
    real_timeout_at = asyncio.timeout_at

    def _counting_timeout_at(when: float | None) -> Any:
        scopes["n"] += 1
        return real_timeout_at(when)

    monkeypatch.setattr(ps.asyncio, "timeout_at", _counting_timeout_at)

    strat = _strategy(
        retry={"attempts": 2, "wait_min_seconds": 0.001},
        timeouts={"total": 60, "chunk_timeout": disabled},
    )
    results = await _run(strat, tmp_path)

    assert results[0]["output_text"] == "done"
    assert scopes["n"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_watchdog_fires_during_backoff_sleep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ceiling bounds backoff too, so one slow retry wait cannot outlast it."""
    calls = {"n": 0}

    async def _boom(**_kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        raise RuntimeError("Error code: 503 - service unavailable")

    _install_stubs(monkeypatch, _boom)
    strat = _strategy(
        retry={
            "attempts": 5,
            "wait_min_seconds": 30.0,
            "wait_max_seconds": 120.0,
            "jitter_max_seconds": 0.0,
        },
        timeouts={"total": 60, "chunk_timeout": 0.3},
    )

    started = time.monotonic()
    results = await _run(strat, tmp_path)
    elapsed = time.monotonic() - started

    assert elapsed < 3.0
    assert calls["n"] == 1
    assert "wall-clock ceiling" in results[0]["error"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_external_cancel_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cancel from outside propagates instead of becoming a failure dict."""
    entered = asyncio.Event()
    blocker = asyncio.Event()

    async def _block(**_kwargs: Any) -> dict[str, Any]:
        entered.set()
        await blocker.wait()
        return {"output_text": "never"}

    _install_stubs(monkeypatch, _block)
    strat = _strategy(
        retry={"attempts": 2, "wait_min_seconds": 0.001},
        timeouts={"total": 60, "chunk_timeout": 30},
    )

    task = asyncio.create_task(_run(strat, tmp_path))
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_persistence_is_not_cancellable_by_the_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The temp-JSONL write finishes even when the ceiling has already passed.

    Cancelling the write would release ``write_lock`` while the worker
    thread is still writing and interleave bytes in the resume file, so
    the success path must sit outside every timeout scope.
    """

    async def _ok(**_kwargs: Any) -> dict[str, Any]:
        return {"output_text": "done"}

    spy = _install_stubs(monkeypatch, _ok)

    real_append = ps._append_jsonl_line

    def _slow_append(handle: Any, line: str) -> None:
        # Outlast the 0.2 s ceiling from inside the persistence path.
        time.sleep(0.5)
        real_append(handle, line)

    monkeypatch.setattr(ps, "_append_jsonl_line", _slow_append)
    strat = _strategy(
        retry={"attempts": 2, "wait_min_seconds": 0.001},
        timeouts={"total": 60, "chunk_timeout": 0.2},
    )
    results = await _run(strat, tmp_path)

    assert results[0]["output_text"] == "done"
    assert "error" not in results[0]
    assert spy.successes == 1
    assert (tmp_path / "temp.jsonl").exists()


# ---------------------------------------------------------------------------
# Timeout retry budget
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bare_read_timeout_retries_then_stops_at_timeout_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bare ReadTimeout is retryable, but only for ``timeout_attempts`` calls.

    ``httpx.ReadTimeout("read timeout")`` used to stringify past the
    message-only classifier, so the chunk failed after a single call.
    """
    calls = {"n": 0}

    async def _timeout(**_kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        raise httpx.ReadTimeout("read timeout")

    _install_stubs(monkeypatch, _timeout)
    strat = _strategy(
        retry={**_FAST_RETRY, "timeout_attempts": 3},
        timeouts={"total": 60, "chunk_timeout": "off"},
    )
    results = await _run(strat, tmp_path)

    assert calls["n"] == 3
    assert results[0]["chunk_index"] == 1
    assert "error" in results[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bare_read_timeout_with_empty_message_still_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The empty-string stringification is the exact case that used to slip."""
    calls = {"n": 0}

    async def _timeout(**_kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        raise httpx.ReadTimeout("")

    _install_stubs(monkeypatch, _timeout)
    strat = _strategy(
        retry={**_FAST_RETRY, "timeout_attempts": 2},
        timeouts={"total": 60, "chunk_timeout": "off"},
    )
    results = await _run(strat, tmp_path)

    assert calls["n"] == 2
    assert "error" in results[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_connection_error_keeps_the_full_attempt_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A refused connection is cheap and unbilled, so it spends all attempts."""
    calls = {"n": 0}

    async def _refused(**_kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        raise httpx.ConnectError("connection refused")

    _install_stubs(monkeypatch, _refused)
    strat = _strategy(
        retry={**_FAST_RETRY, "timeout_attempts": 3},
        timeouts={"total": 60, "chunk_timeout": "off"},
    )
    results = await _run(strat, tmp_path)

    assert calls["n"] == 8
    assert "error" in results[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_timeout_attempts_is_clamped_to_the_attempt_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``timeout_attempts`` can never exceed the general attempt budget."""
    calls = {"n": 0}

    async def _timeout(**_kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        raise httpx.ReadTimeout("read timeout")

    _install_stubs(monkeypatch, _timeout)
    strat = _strategy(
        retry={**_FAST_RETRY, "attempts": 2, "timeout_attempts": 9},
        timeouts={"total": 60, "chunk_timeout": "off"},
    )
    results = await _run(strat, tmp_path)

    assert calls["n"] == 2
    assert "error" in results[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gateway_timeout_status_keeps_the_full_attempt_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 504 body reads as "timeout" but is a server error, not a read timeout."""
    calls = {"n": 0}

    async def _gateway(**_kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        raise Exception("Error code: 504 - Gateway Timeout")

    _install_stubs(monkeypatch, _gateway)
    strat = _strategy(
        retry={**_FAST_RETRY, "attempts": 5, "timeout_attempts": 2},
        timeouts={"total": 60, "chunk_timeout": "off"},
    )
    results = await _run(strat, tmp_path)

    assert calls["n"] == 5
    assert "error" in results[0]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_classify_transient_error_uses_the_exception_type() -> None:
    from modules.extract.processing_strategy import classify_transient_error

    # Type-level: the message carries nothing at all.
    _, is_timeout, _ = classify_transient_error("", httpx.ReadTimeout(""))
    assert is_timeout

    # Message-level classification still stands on its own.
    _, is_timeout, _ = classify_transient_error("Request timed out.")
    assert is_timeout
    _, is_timeout, _ = classify_transient_error("read timeout")
    assert is_timeout

    # A connection failure is not a timeout.
    _, is_timeout, _ = classify_transient_error(
        "connection refused", httpx.ConnectError("connection refused")
    )
    assert not is_timeout

    # Non-retryable errors stay non-retryable when an exception is supplied.
    exc = Exception("Error code: 400 - invalid request body")
    is_429, is_timeout, server = classify_transient_error(str(exc), exc)
    assert not (is_429 or is_timeout or server)
