"""Client-side rate limiting for model calls, with adaptive backoff.

A :class:`RateLimiter` keeps a sliding log of request times for several
windows (for example per second, per minute and per hour) and admits a call
only when every window has room. An error multiplier lengthens waits after
rate-limit or server errors and relaxes again on success. A
:class:`RateLimiterRegistry` holds one limiter per provider for one run, so
all calls to a provider share one set of windows.

Usage::

    registry = RateLimiterRegistry(limits)
    limiter = registry.get("openai")
    await limiter.acquire()
    try:
        ...                              # the API call
    except Exception:
        limiter.report_error(is_rate_limit=True)
        raise
    limiter.report_success()
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from typing import Any

__all__ = [
    "DEFAULT_RATE_LIMITS",
    "RateLimitWaitAborted",
    "RateLimiter",
    "RateLimiterRegistry",
    "parse_rate_limits",
]

logger = logging.getLogger(__name__)

MIN_SLEEP_TIME = 0.05
MAX_SLEEP_TIME = 0.50
ERROR_MULTIPLIER_DECREASE_RATE = 0.9
ERROR_MULTIPLIER_INCREASE_RATE_LIMIT = 1.5
ERROR_MULTIPLIER_INCREASE_OTHER = 1.2
CONSECUTIVE_ERRORS_THRESHOLD = 2
MAX_ERROR_MULTIPLIER = 5.0
# Per-request delay (seconds) per unit of multiplier elevation, applied when no
# window is saturated, so repeated errors still slow admission.
ERROR_BASE_PENALTY_SECONDS = 0.5

DEFAULT_RATE_LIMITS: tuple[tuple[int, int], ...] = (
    (120, 1),
    (15000, 60),
    (15000, 3600),
)


class RateLimitWaitAborted(Exception):
    """Raised by :meth:`RateLimiter.acquire` when ``should_abort`` turns True.

    No request is recorded, so the caller must not make the call.
    """


class RateLimiter:
    """Admit calls under several ``(max_requests, window_seconds)`` limits."""

    def __init__(
        self,
        limits: Iterable[tuple[int, int]] = DEFAULT_RATE_LIMITS,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limits: list[tuple[int, int]] = [(int(n), int(s)) for n, s in limits]
        self.request_timestamps: list[deque[float]] = [
            deque(maxlen=max_requests) for max_requests, _ in self.limits
        ]
        self.consecutive_errors = 0
        self.error_multiplier = 1.0
        self.max_error_multiplier = MAX_ERROR_MULTIPLIER
        self.total_requests = 0
        self.total_wait_time = 0.0
        self._clock = clock
        self._lock = threading.Lock()

    def _required_wait(self, now: float, wait_start: float) -> float:
        """Return the wait the windows and the multiplier impose at ``now``."""
        wait_time = 0.0
        for timestamps, (max_requests, seconds) in zip(
            self.request_timestamps, self.limits, strict=True
        ):
            cutoff = now - seconds
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()
            if len(timestamps) >= max_requests:
                wait_time = max(wait_time, timestamps[0] + seconds - now)

        if self.error_multiplier > 1.0:
            # The penalty is a deadline measured from wait_start, not a floor
            # applied on every pass: a floor would never admit, and only an
            # admission lets report_success lower the multiplier.
            penalty = (self.error_multiplier - 1.0) * ERROR_BASE_PENALTY_SECONDS
            wait_time = max(
                wait_time * self.error_multiplier, wait_start + penalty - now
            )
        return wait_time

    def try_acquire(self, wait_start: float | None = None) -> float:
        """Record a request and return 0.0, or return the seconds still to wait.

        ``wait_start`` is the clock reading when the caller began waiting; it
        anchors the error penalty. It defaults to now.
        """
        with self._lock:
            now = self._clock()
            start = now if wait_start is None else wait_start
            wait_time = self._required_wait(now, start)
            if wait_time > 0:
                return wait_time
            for timestamps in self.request_timestamps:
                timestamps.append(now)
            self.total_requests += 1
            self.total_wait_time += now - start
            return 0.0

    async def acquire(self, should_abort: Callable[[], bool] | None = None) -> float:
        """Wait until every window has room, record the request, return the wait.

        Sleeps in slices of at most ``MAX_SLEEP_TIME``. Raises
        :class:`RateLimitWaitAborted` without recording a request when
        ``should_abort`` returns True between slices.
        """
        wait_start = self._clock()
        while True:
            wait_time = self.try_acquire(wait_start)
            if wait_time <= 0:
                return self._clock() - wait_start
            if should_abort is not None and should_abort():
                raise RateLimitWaitAborted("rate-limit wait aborted")
            await asyncio.sleep(min(wait_time + MIN_SLEEP_TIME, MAX_SLEEP_TIME))

    def report_success(self) -> None:
        """Record a successful call and relax the error multiplier."""
        with self._lock:
            self.consecutive_errors = 0
            if self.error_multiplier > 1.0:
                self.error_multiplier = max(
                    1.0, self.error_multiplier * ERROR_MULTIPLIER_DECREASE_RATE
                )

    def report_error(self, is_rate_limit: bool = False) -> None:
        """Record a failed call and raise the error multiplier.

        A rate-limit or server error raises it at once; other errors raise it
        only after ``CONSECUTIVE_ERRORS_THRESHOLD`` consecutive failures.
        """
        with self._lock:
            self.consecutive_errors += 1
            if is_rate_limit:
                factor = ERROR_MULTIPLIER_INCREASE_RATE_LIMIT
            elif self.consecutive_errors > CONSECUTIVE_ERRORS_THRESHOLD:
                factor = ERROR_MULTIPLIER_INCREASE_OTHER
            else:
                return
            self.error_multiplier = min(
                self.max_error_multiplier, self.error_multiplier * factor
            )

    def stats(self) -> dict[str, Any]:
        """Return request count, total and average wait, and the multiplier."""
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "total_wait_time": round(self.total_wait_time, 2),
                "average_wait": round(
                    self.total_wait_time / max(1, self.total_requests), 4
                ),
                "queue_lengths": [len(ts) for ts in self.request_timestamps],
                "error_multiplier": round(self.error_multiplier, 2),
            }


def parse_rate_limits(
    raw: Any, default: Sequence[tuple[int, int]] = DEFAULT_RATE_LIMITS
) -> list[tuple[int, int]]:
    """Parse a list of ``[max_requests, window_seconds]`` pairs from settings.

    Malformed or non-positive entries are skipped with a warning; ``default``
    is returned when the value is not a list or no entry is usable.
    """
    if not isinstance(raw, list | tuple):
        return list(default)
    limits: list[tuple[int, int]] = []
    for item in raw:
        if not (isinstance(item, list | tuple) and len(item) == 2):
            logger.warning("Skipping malformed rate limit entry: %r", item)
            continue
        try:
            max_requests, window_seconds = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            logger.warning("Skipping non-integer rate limit entry: %r", item)
            continue
        if max_requests < 1 or window_seconds <= 0:
            logger.warning("Skipping non-positive rate limit entry: %r", item)
            continue
        limits.append((max_requests, window_seconds))
    return limits or list(default)


class RateLimiterRegistry:
    """One :class:`RateLimiter` per provider, owned by one run.

    ``per_provider`` overrides the limits for named providers.
    """

    def __init__(
        self,
        limits: Iterable[tuple[int, int]] = DEFAULT_RATE_LIMITS,
        *,
        per_provider: dict[str, Sequence[tuple[int, int]]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limits = tuple(limits)
        self._per_provider = dict(per_provider or {})
        self._clock = clock
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

    def get(self, provider: str | None) -> RateLimiter:
        """Return the limiter of ``provider``, creating it on first use."""
        key = provider or "default"
        with self._lock:
            limiter = self._limiters.get(key)
            if limiter is None:
                limits = self._per_provider.get(key, self._limits)
                limiter = RateLimiter(limits, clock=self._clock)
                self._limiters[key] = limiter
            return limiter

    def __contains__(self, provider: object) -> bool:
        return provider in self._limiters
