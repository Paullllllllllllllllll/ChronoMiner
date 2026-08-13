"""
Per-phase HTTP timeouts for the ChatOpenAI-family providers.

A scalar float handed to an httpx-backed client is applied to all four
httpx timeout phases at once (connect, read, write, pool). The generous
read budget an extraction chunk needs -- 900 s is routine for a long
reasoning response -- therefore silently becomes a 900 s *connect*
budget as well, so a dead peer or a black-holed TCP handshake goes
undetected for the full quarter hour. This builder keeps the configured
value on the read phase only and pins tight connect/write/pool budgets.

Only the ChatOpenAI-based providers (``openai``, ``openrouter``, and
custom OpenAI-compatible endpoints) receive an :class:`httpx.Timeout`.
The Anthropic and Google LangChain wrappers require a plain float --
``ChatAnthropic`` compares its timeout with ``> 0`` and
``ChatGoogleGenerativeAI`` computes ``int(timeout * 1000)`` -- and are
deliberately left on the scalar value.

Note that :class:`httpx.Timeout` is unhashable, which bypasses
langchain-openai's ``lru_cache``-based client sharing, so every chat
model builds its own HTTP client. That is accepted here: providers are
constructed once per extractor, not per chunk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_WRITE_TIMEOUT = 30.0
DEFAULT_POOL_TIMEOUT = 30.0


def coerce_positive_float(value: Any, fallback: float) -> float:
    """Coerce ``value`` to a positive float, falling back when unusable."""
    # Booleans are ints in Python, so reject them before numeric coercion.
    if isinstance(value, bool):
        return fallback
    if not isinstance(value, int | float):
        return fallback
    if value <= 0:
        return fallback
    return float(value)


def build_httpx_timeout(
    read_timeout: float | None,
    *,
    connect: Any = None,
    write: Any = None,
    pool: Any = None,
) -> httpx.Timeout | None:
    """Build a per-phase timeout that spends the budget on reads only.

    ``read_timeout`` is the configured per-request budget from
    ``concurrency.extraction.timeouts``. The ``connect``, ``write``, and
    ``pool`` arguments accept raw override values of any type; each is
    coerced against its module default, so garbage values degrade to the
    default rather than raising. Returns ``None`` when ``read_timeout``
    is ``None`` (no timeout configured).
    """
    if read_timeout is None:
        return None

    # Imported lazily: modules.llm re-exports eagerly, so a module-level
    # import would drag httpx onto the CLI startup path.
    import httpx

    # The positional argument is httpx's ``default``; ``read`` is
    # intentionally not overridden so it inherits that value.
    return httpx.Timeout(
        float(read_timeout),
        connect=coerce_positive_float(connect, DEFAULT_CONNECT_TIMEOUT),
        write=coerce_positive_float(write, DEFAULT_WRITE_TIMEOUT),
        pool=coerce_positive_float(pool, DEFAULT_POOL_TIMEOUT),
    )
