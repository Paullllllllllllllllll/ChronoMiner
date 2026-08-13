"""
Transient-failure classification and the chunk watchdog ceiling.

Provider SDKs wrap the underlying transport failure rather than raising
it, so a retry policy that inspects only the top-level exception type
misclassifies most network trouble. The helpers here walk the exception
chain and answer two separate questions -- "did this time out?" and "was
this a connection failure?" -- which carry different retry budgets.

Providers that stringify their failures instead of exposing a structured
status code are handled by :func:`is_rate_limit_message` and
:func:`is_server_error_message`, the single source of 429/5xx string
classification for both the extraction and the readjustment paths.

Also parses the ``chunk_timeout`` key of
``concurrency.extraction.timeouts`` into the wall-clock ceiling that
guards a single extraction chunk across all of its retry attempts.
"""

from __future__ import annotations

import re
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


# A 5xx status code counts as a transient server error, but ONLY when it
# appears in a status/HTTP/error-code context or next to a canonical 5xx
# reason phrase. A blanket ``\b5\d{2}\b`` false-positives on any stray
# number (e.g. "line 502 of file.py"), burning the whole retry budget on a
# non-retryable error. Covers Cloudflare edge codes (520-526) too.
_SERVER_ERROR_CODE_RE = re.compile(
    r"(?:status(?:[ _]?code)?|http|error[ _]?code|code)\s*[:=]?\s*5\d{2}\b"
    r"|\b5\d{2}\b\s*(?:internal server error|server error|bad gateway"
    r"|service unavailable|gateway timeout|origin)",
    re.IGNORECASE,
)

# Same context-gating for 429: a blanket ``"429" in msg`` matches any stray
# substring (e.g. "you requested 132429 tokens" or "position 429"), retrying
# a permanently failing request for ~10 minutes and throttling the shared
# rate limiter for every concurrent chunk of that provider.
_RATE_LIMIT_CODE_RE = re.compile(
    r"(?:status(?:[ _]?code)?|http|error[ _]?code|code)\s*[:=]?\s*429\b"
    r"|\b429\b\s*(?:too many requests|rate limit)"
    r"|too many requests",
    re.IGNORECASE,
)


def is_rate_limit_message(message: str) -> bool:
    """Return ``True`` when a stringified error looks like a 429.

    Message-level fallback only: a structured ``status_code`` on the
    exception is authoritative and should be consulted first.
    """
    return "rate_limit" in message.lower() or bool(_RATE_LIMIT_CODE_RE.search(message))


def is_server_error_message(message: str) -> bool:
    """Return ``True`` when a stringified error looks like a 5xx.

    Beyond the context-gated status pattern this also accepts the shapes
    providers use instead of a code: SDK class names, gateway/upstream
    wording, self-declared retryable bodies, and reset or refused
    connections. Message-level fallback only, as above.
    """
    msg = message.lower()
    return (
        bool(_SERVER_ERROR_CODE_RE.search(message))
        or "internalservererror" in msg
        or "upstream" in msg
        # Cloudflare-style bodies self-declare retryability.
        or "'retryable': true" in msg
        or '"retryable": true' in msg
        or ("connection" in msg and ("reset" in msg or "refused" in msg))
        # openai SDK APIConnectionError stringifies to the bare message
        # "Connection error." (transport-level failure, e.g. a stale
        # keep-alive connection the server already closed). Always
        # transient: a retry opens a fresh connection. Frequent under
        # service_tier=flex, which closes connections after each response.
        or "connection error" in msg
    )


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
