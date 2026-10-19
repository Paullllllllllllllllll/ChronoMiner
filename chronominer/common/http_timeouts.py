"""Per-phase httpx timeouts for OpenAI-compatible clients.

A scalar timeout handed to an httpx-backed client applies to all four phases
(connect, read, write, pool), so a long read budget also becomes the connect
budget. :func:`build_httpx_timeout` puts the configured value on the read phase
only and gives connect, write and pool their own, shorter budgets.

The Anthropic and Google LangChain wrappers require a plain float timeout and
should not receive an :class:`httpx.Timeout`.
"""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_WRITE_TIMEOUT = 30.0
DEFAULT_POOL_TIMEOUT = 30.0


def _positive_float(value: Any, fallback: float) -> float:
    """Coerce a config value to a positive float, else return ``fallback``.

    Booleans are rejected explicitly (``isinstance(True, int)`` is ``True``),
    as are non-numeric values and anything at or below zero.
    """
    if isinstance(value, bool):
        return fallback
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if parsed <= 0:
        return fallback
    return parsed


def build_httpx_timeout(
    read_timeout: float | int | None,
    *,
    connect: Any = DEFAULT_CONNECT_TIMEOUT,
    write: Any = DEFAULT_WRITE_TIMEOUT,
    pool: Any = DEFAULT_POOL_TIMEOUT,
) -> httpx.Timeout | None:
    """Build an :class:`httpx.Timeout` whose read phase is ``read_timeout``.

    Args:
        read_timeout: The request timeout in seconds, or ``None`` to keep the
            SDK's own defaults.
        connect: Connect timeout; a non-positive value takes the default.
        write: Write timeout; a non-positive value takes the default.
        pool: Pool timeout; a non-positive value takes the default.

    Returns:
        ``None`` when ``read_timeout`` is ``None``; otherwise a timeout whose
        positional default (and hence the read phase) is ``read_timeout``.
    """
    if read_timeout is None:
        return None

    connect = _positive_float(connect, DEFAULT_CONNECT_TIMEOUT)
    write = _positive_float(write, DEFAULT_WRITE_TIMEOUT)
    pool = _positive_float(pool, DEFAULT_POOL_TIMEOUT)

    # The positional argument is httpx's default, which the read phase inherits.
    return httpx.Timeout(float(read_timeout), connect=connect, write=write, pool=pool)


__all__ = [
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_POOL_TIMEOUT",
    "DEFAULT_WRITE_TIMEOUT",
    "build_httpx_timeout",
]
