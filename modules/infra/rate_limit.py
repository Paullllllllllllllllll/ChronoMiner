"""Client-side rate limiter for LLM API calls with adaptive backoff.

A multi-window sliding-log limiter: it tracks request timestamps across several
concurrent windows (e.g. per-second, per-minute, per-hour) and blocks until all
windows have capacity before admitting the next call. An adaptive error
multiplier lengthens waits after rate-limit / server errors and relaxes again on
success, smoothing bursts that would otherwise trip provider 429s.

The limiter is synchronous and thread-safe (it is shared across the asyncio
fan-out, which dispatches synchronous LLM calls onto worker threads). Use
:func:`await_capacity` from coroutine code so the blocking wait runs off the
event loop.

Usage::

    limiter = get_shared_rate_limiter("openai")
    await await_capacity(limiter)      # blocks off-loop until capacity
    try:
        ...                            # make the API call
        limiter.report_success()
    except Exception:
        limiter.report_error(is_rate_limit=True)
        raise
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any

from modules.infra.logger import setup_logger

logger = setup_logger(__name__)

# --------------------------------------------------------------------------- #
# Tuning constants
# --------------------------------------------------------------------------- #
# Sleep bounds applied between capacity checks (jitter avoids thundering herd).
MIN_SLEEP_TIME = 0.05
MAX_SLEEP_TIME = 0.50
# Adaptive error-multiplier dynamics.
ERROR_MULTIPLIER_DECREASE_RATE = 0.9
ERROR_MULTIPLIER_INCREASE_RATE_LIMIT = 1.5
ERROR_MULTIPLIER_INCREASE_OTHER = 1.2
CONSECUTIVE_ERRORS_THRESHOLD = 2
MAX_ERROR_MULTIPLIER = 5.0

# Permissive defaults applied when no rate_limits block is configured. Chosen so
# the limiter is effectively transparent for typical workloads.
DEFAULT_RATE_LIMITS: list[tuple[int, int]] = [(120, 1), (15000, 60), (15000, 3600)]


class RateLimiter:
    """Manage API call rates across multiple windows with adaptive backoff.

    Tracks request timestamps per window and adjusts wait times based on
    observed error patterns.
    """

    limits: list[tuple[int, int]]
    request_timestamps: list[deque[float]]
    lock: threading.Lock
    total_requests: int
    total_wait_time: float
    last_stats_update: float
    request_count_since_last_update: int
    consecutive_errors: int
    error_multiplier: float
    max_error_multiplier: float

    def __init__(self, limits: list[tuple[int, int]]) -> None:
        """Initialize the limiter.

        :param limits: List of ``(max_requests, window_seconds)`` tuples.
        """
        self.limits = list(limits)
        self.request_timestamps = [deque(maxlen=limit[0]) for limit in self.limits]
        self.lock = threading.Lock()

        # Statistics.
        self.total_requests = 0
        self.total_wait_time = 0.0
        self.last_stats_update = time.time()
        self.request_count_since_last_update = 0

        # Adaptive backoff.
        self.consecutive_errors = 0
        self.error_multiplier = 1.0
        self.max_error_multiplier = MAX_ERROR_MULTIPLIER

    def wait_for_capacity(self) -> float:
        """Block until every window has capacity, then record the request.

        :return: Total time waited, in seconds.
        """
        wait_start = time.time()

        while True:
            wait_time = 0.0

            with self.lock:
                now = time.time()

                for i, (max_requests, seconds) in enumerate(self.limits):
                    cutoff = now - seconds
                    while (
                        self.request_timestamps[i]
                        and self.request_timestamps[i][0] < cutoff
                    ):
                        self.request_timestamps[i].popleft()

                    if len(self.request_timestamps[i]) >= max_requests:
                        oldest_request_time = self.request_timestamps[i][0]
                        required_wait = oldest_request_time + seconds - now
                        wait_time = max(wait_time, required_wait)

                # Lengthen the wait when recent errors have raised the multiplier.
                wait_time *= self.error_multiplier

                if wait_time <= 0:
                    for timestamps in self.request_timestamps:
                        timestamps.append(now)
                    self.total_requests += 1
                    self.request_count_since_last_update += 1
                    total_wait = time.time() - wait_start
                    self.total_wait_time += total_wait
                    return total_wait

            # Sleep with jitter to avoid a thundering herd of retries.
            sleep_time = min(wait_time + MIN_SLEEP_TIME, MAX_SLEEP_TIME)
            time.sleep(sleep_time)

    def report_success(self) -> None:
        """Record a successful call, relaxing the adaptive multiplier."""
        with self.lock:
            self.consecutive_errors = 0
            if self.error_multiplier > 1.0:
                self.error_multiplier = max(
                    1.0, self.error_multiplier * ERROR_MULTIPLIER_DECREASE_RATE
                )

    def report_error(self, is_rate_limit: bool = False) -> None:
        """Record a failed call, tightening the adaptive multiplier.

        :param is_rate_limit: True for a rate-limit (429) or server (5xx) error,
            which raises the multiplier immediately; other errors raise it only
            after crossing the consecutive-error threshold.
        """
        with self.lock:
            self.consecutive_errors += 1

            if is_rate_limit:
                self.error_multiplier = min(
                    self.max_error_multiplier,
                    self.error_multiplier * ERROR_MULTIPLIER_INCREASE_RATE_LIMIT,
                )
            elif self.consecutive_errors > CONSECUTIVE_ERRORS_THRESHOLD:
                self.error_multiplier = min(
                    self.max_error_multiplier,
                    self.error_multiplier * ERROR_MULTIPLIER_INCREASE_OTHER,
                )

    def get_stats(self) -> dict[str, Any]:
        """Return current limiter statistics.

        Not a pure query: ``current_rate`` is measured since the previous call,
        so invoking this resets the per-poll counters.
        """
        with self.lock:
            now = time.time()
            time_since_last_update = now - self.last_stats_update
            requests_per_second = self.request_count_since_last_update / max(
                1, time_since_last_update
            )

            stats = {
                "total_requests": self.total_requests,
                "total_wait_time": round(self.total_wait_time, 2),
                "average_wait": round(
                    self.total_wait_time / max(1, self.total_requests), 4
                ),
                "current_rate": round(requests_per_second, 2),
                "current_queue_lengths": [len(ts) for ts in self.request_timestamps],
                "error_multiplier": round(self.error_multiplier, 2),
            }

            self.last_stats_update = now
            self.request_count_since_last_update = 0

            return stats


def get_rate_limits() -> list[tuple[int, int]]:
    """Resolve the configured rate-limit windows, or permissive defaults.

    Reads ``concurrency.rate_limits`` (a list of ``[max_requests, window]``
    pairs) from concurrency_config.yaml. Returns :data:`DEFAULT_RATE_LIMITS`
    when the block is absent, malformed, or the config cannot be read, so the
    feature stays transparent without configuration.
    """
    default_limits = list(DEFAULT_RATE_LIMITS)
    try:
        from modules.config.loader import get_config_loader

        concurrency_cfg = get_config_loader().get_concurrency_config() or {}
        raw_limits = (concurrency_cfg.get("concurrency", {}) or {}).get("rate_limits")

        if not isinstance(raw_limits, list):
            return default_limits

        limits: list[tuple[int, int]] = []
        for item in raw_limits:
            if isinstance(item, list | tuple) and len(item) == 2:
                try:
                    limits.append((int(item[0]), int(item[1])))
                except (ValueError, TypeError):
                    continue

        return limits if limits else default_limits
    except Exception as exc:
        logger.debug("Error loading rate limits: %s", exc)
        return default_limits


# --------------------------------------------------------------------------- #
# Shared per-provider limiters
# --------------------------------------------------------------------------- #
# One RateLimiter per provider so every synchronous call for that provider draws
# on one shared set of windows (separate instances would multiply the rate).
_SHARED_LIMITERS: dict[str, RateLimiter] = {}
_SHARED_LIMITERS_LOCK = threading.Lock()


def get_shared_rate_limiter(provider: str | None) -> RateLimiter:
    """Return the process-wide :class:`RateLimiter` shared by *provider*."""
    key = provider or "default"
    with _SHARED_LIMITERS_LOCK:
        limiter = _SHARED_LIMITERS.get(key)
        if limiter is None:
            limiter = RateLimiter(get_rate_limits())
            _SHARED_LIMITERS[key] = limiter
        return limiter


def reset_shared_rate_limiters() -> None:
    """Drop all shared limiters (test isolation; re-created on next request)."""
    with _SHARED_LIMITERS_LOCK:
        _SHARED_LIMITERS.clear()


async def await_capacity(limiter: RateLimiter) -> float:
    """Acquire capacity from *limiter* without blocking the event loop.

    ``wait_for_capacity`` is synchronous and may sleep; running it via
    :func:`asyncio.to_thread` keeps the loop free for other coroutines.

    :return: Total time waited, in seconds.
    """
    return await asyncio.to_thread(limiter.wait_for_capacity)


__all__ = [
    "RateLimiter",
    "get_rate_limits",
    "get_shared_rate_limiter",
    "reset_shared_rate_limiters",
    "await_capacity",
]
