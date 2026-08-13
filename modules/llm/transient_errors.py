"""
Transient-failure classification and the chunk watchdog ceiling.

Provider SDKs wrap the underlying transport failure rather than raising
it, so a retry policy that inspects only the top-level exception type
misclassifies most network trouble. The helpers here walk the exception
chain and answer two separate questions -- "did this time out?" and "was
this a connection failure?" -- which carry different retry budgets.

Also parses the ``chunk_timeout`` key of
``concurrency.extraction.timeouts`` into the wall-clock ceiling that
guards a single extraction chunk across all of its retry attempts.
"""

from __future__ import annotations

from typing import Any

from modules.infra.logger import setup_logger

logger = setup_logger(__name__)

_TIMEOUT_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "APITimeoutError",
        "APIConnectionTimeoutError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
        "DeadlineExceeded",
        "ServerTimeoutError",
    }
)

# Belt-and-braces bound alongside the id()-based cycle set: a pathological
# wrapper chain should not turn classification into a long walk.
_MAX_CHAIN_DEPTH = 20

_DEFAULT_TOTAL_TIMEOUT = 600.0
_AUTO_TIMEOUT_MARGIN = 300.0


def is_timeout_error(exc: BaseException) -> bool:
    """Return ``True`` when ``exc`` or one of its causes is a timeout.

    Deliberately narrower than :func:`is_connection_error`: a refused
    connection is cheap and unbilled, whereas a read timeout means a
    billed server-side generation has already burned -- and its usage
    payload never arrives -- so timeouts get their own, smaller retry
    budget. The class-name fallback covers SDK timeout types raised
    without an httpx cause (a bare ``openai.APITimeoutError``, for
    instance), and matching the builtin ``TimeoutError`` also covers
    asyncio-raised timeouts, since ``asyncio.TimeoutError`` *is*
    ``TimeoutError`` on Python 3.11+.
    """
    import httpx

    seen: set[int] = set()
    current: BaseException | None = exc
    depth = 0
    while current is not None and depth < _MAX_CHAIN_DEPTH:
        if id(current) in seen:
            return False
        seen.add(id(current))
        if isinstance(current, httpx.TimeoutException | TimeoutError):
            return True
        if type(current).__name__ in _TIMEOUT_CLASS_NAMES:
            return True
        current = current.__cause__ or current.__context__
        depth += 1
    return False


def is_connection_error(exc: BaseException) -> bool:
    """Return ``True`` when ``exc`` or one of its causes is a transport failure.

    Provider SDKs wrap the underlying httpx transport error (an
    ``openai.APIConnectionError`` raised from an ``httpx.ConnectError``,
    for example), so a top-level type check misses them; the chain is
    walked instead.
    """
    import httpx

    seen: set[int] = set()
    current: BaseException | None = exc
    depth = 0
    while current is not None and depth < _MAX_CHAIN_DEPTH:
        if id(current) in seen:
            return False
        seen.add(id(current))
        if isinstance(current, httpx.ConnectError | httpx.TimeoutException):
            return True
        current = current.__cause__ or current.__context__
        depth += 1
    return False


class ChunkTimeoutError(Exception):
    """Raised when one extraction chunk overruns its wall-clock ceiling.

    Subclasses :class:`Exception` -- never :class:`BaseException` -- so
    that the per-unit handlers convert an overrun into a single failed
    chunk instead of tearing down the whole run.
    """

    def __init__(self, label: str, seconds: float) -> None:
        self.label = label
        self.seconds = seconds
        super().__init__(
            f"{label!r} exceeded its {seconds:.0f}s wall-clock ceiling "
            "across all retry attempts"
        )


def _auto_chunk_timeout(timeouts_cfg: dict[str, Any] | None, attempts: int) -> float:
    """Compute the automatic ceiling from the per-request total budget."""
    total = _DEFAULT_TOTAL_TIMEOUT
    raw_total: Any = (timeouts_cfg or {}).get("total")
    if not isinstance(raw_total, bool):
        try:
            candidate = float(raw_total)
        except (TypeError, ValueError):
            candidate = 0.0
        if candidate > 0:
            total = candidate
    return total * attempts + _AUTO_TIMEOUT_MARGIN


def resolve_chunk_timeout(
    timeouts_cfg: dict[str, Any] | None,
    *,
    timeout_attempts: int,
) -> float | None:
    """Resolve the wall-clock ceiling for a single extraction chunk.

    Reads ``chunk_timeout`` from ``concurrency.extraction.timeouts``,
    defaulting to ``"auto"`` when the key or the whole mapping is
    absent. Accepted values:

    - ``"auto"`` (or ``True``, or an unparseable string): the ceiling is
      ``total * timeout_attempts + 300`` seconds, where ``total`` falls
      back to 600 s when missing or unusable.
    - a positive number, or a string parsing to one: that value in
      seconds.
    - ``None``, ``False``, a non-positive number, or one of ``"off"``,
      ``"none"``, ``"disabled"``: no ceiling. Note that YAML 1.1 parses a
      bare ``off`` as the boolean ``False``, which is why ``False`` is
      accepted as "disabled" too.
    - anything else: no ceiling.

    The ``auto`` ceiling deliberately sits *above* what the timeout
    retry budget can consume on its own. It is a backstop, not a second
    per-request timeout: it also catches mixed-classification
    pathologies -- alternating timeouts, connection errors, and 429
    backoffs -- which would otherwise still be allowed to run the full
    ``timeout_attempts * total``.
    """
    auto = _auto_chunk_timeout(timeouts_cfg, timeout_attempts)
    if not timeouts_cfg:
        return auto

    raw = timeouts_cfg.get("chunk_timeout", "auto")
    if raw is None:
        return None
    # Booleans are ints in Python, so branch on them before numbers.
    if isinstance(raw, bool):
        return auto if raw else None
    if isinstance(raw, int | float):
        return float(raw) if raw > 0 else None
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"off", "none", "disabled"}:
            return None
        if text == "auto":
            return auto
        try:
            value = float(text)
        except ValueError:
            logger.debug(
                "Unparseable chunk_timeout %r; falling back to the auto ceiling",
                raw,
            )
            return auto
        return value if value > 0 else None
    return None
