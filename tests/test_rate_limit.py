"""Tests for the client-side rate limiter (modules/infra/rate_limit.py)."""

from __future__ import annotations

import time

import pytest

from modules.infra import rate_limit
from modules.infra.rate_limit import (
    DEFAULT_RATE_LIMITS,
    MAX_ERROR_MULTIPLIER,
    RateLimiter,
    await_capacity,
    get_rate_limits,
    get_shared_rate_limiter,
    reset_shared_rate_limiters,
)


@pytest.mark.unit
def test_wait_for_capacity_records_request_when_under_limit():
    limiter = RateLimiter([(1000, 1)])
    waited = limiter.wait_for_capacity()
    assert waited >= 0.0
    assert limiter.total_requests == 1
    # One timestamp recorded in each window.
    assert [len(ts) for ts in limiter.request_timestamps] == [1]


@pytest.mark.unit
def test_wait_for_capacity_enforces_window():
    # 2 requests per 1 second; the third must block until the window clears.
    limiter = RateLimiter([(2, 1)])
    limiter.wait_for_capacity()
    limiter.wait_for_capacity()

    start = time.monotonic()
    limiter.wait_for_capacity()
    elapsed = time.monotonic() - start

    assert elapsed >= 0.5
    assert limiter.total_requests == 3


@pytest.mark.unit
def test_report_error_raises_and_success_relaxes_multiplier():
    limiter = RateLimiter([(1000, 1)])
    assert limiter.error_multiplier == 1.0

    limiter.report_error(is_rate_limit=True)
    assert limiter.error_multiplier == pytest.approx(1.5)
    limiter.report_error(is_rate_limit=True)
    assert limiter.error_multiplier == pytest.approx(2.25)

    limiter.report_success()
    assert limiter.error_multiplier < 2.25
    assert limiter.error_multiplier >= 1.0


@pytest.mark.unit
def test_error_multiplier_capped():
    limiter = RateLimiter([(1000, 1)])
    for _ in range(50):
        limiter.report_error(is_rate_limit=True)
    assert limiter.error_multiplier <= MAX_ERROR_MULTIPLIER


@pytest.mark.unit
def test_non_rate_limit_errors_need_threshold():
    limiter = RateLimiter([(1000, 1)])
    # Below the consecutive-error threshold the multiplier stays at 1.0.
    limiter.report_error(is_rate_limit=False)
    limiter.report_error(is_rate_limit=False)
    assert limiter.error_multiplier == 1.0
    # Crossing the threshold finally tightens it.
    limiter.report_error(is_rate_limit=False)
    assert limiter.error_multiplier > 1.0


@pytest.mark.unit
def test_get_stats_shape():
    limiter = RateLimiter([(1000, 1)])
    limiter.wait_for_capacity()
    stats = limiter.get_stats()
    for key in (
        "total_requests",
        "total_wait_time",
        "average_wait",
        "current_rate",
        "current_queue_lengths",
        "error_multiplier",
    ):
        assert key in stats


@pytest.mark.unit
def test_get_rate_limits_reads_config(monkeypatch):
    class _Loader:
        def get_concurrency_config(self):
            return {"concurrency": {"rate_limits": [[10, 1], [100, 60]]}}

    monkeypatch.setattr(
        "modules.config.loader.get_config_loader", lambda *a, **k: _Loader()
    )
    assert get_rate_limits() == [(10, 1), (100, 60)]


@pytest.mark.unit
def test_get_rate_limits_defaults_when_absent(monkeypatch):
    class _Loader:
        def get_concurrency_config(self):
            return {"concurrency": {}}

    monkeypatch.setattr(
        "modules.config.loader.get_config_loader", lambda *a, **k: _Loader()
    )
    assert get_rate_limits() == DEFAULT_RATE_LIMITS


@pytest.mark.unit
def test_get_rate_limits_defaults_on_error(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("no config")

    monkeypatch.setattr("modules.config.loader.get_config_loader", _boom)
    assert get_rate_limits() == DEFAULT_RATE_LIMITS


@pytest.mark.unit
def test_shared_limiter_is_per_provider(monkeypatch):
    monkeypatch.setattr(rate_limit, "get_rate_limits", lambda: [(1000, 1)])
    reset_shared_rate_limiters()

    a1 = get_shared_rate_limiter("openai")
    a2 = get_shared_rate_limiter("openai")
    b = get_shared_rate_limiter("anthropic")

    assert a1 is a2
    assert a1 is not b

    reset_shared_rate_limiters()
    a3 = get_shared_rate_limiter("openai")
    assert a3 is not a1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_await_capacity_runs_off_loop():
    limiter = RateLimiter([(1000, 1)])
    waited = await await_capacity(limiter)
    assert waited >= 0.0
    assert limiter.total_requests == 1
