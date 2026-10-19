"""Token usage of model calls: a value type, extraction and per-role totals.

Cache tokens count at full weight in the input and total counts and are
never counted twice. LangChain folds prompt-cache tokens into
``input_tokens`` and reports the breakdown in ``input_token_details``;
OpenAI cached prompt tokens are a subset of the prompt count. Raw Anthropic
usage reports ``cache_read_input_tokens`` and ``cache_creation_input_tokens``
beside an ``input_tokens`` that excludes them, so those are added on top.
``cached_tokens`` counts only the tokens read from the cache; tokens
written to it count as input but not as cached.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .responses import unwrap_raw_response

__all__ = [
    "Usage",
    "UsageTotals",
    "additive_cache_tokens",
    "coerce_token_count",
    "usage_from_exception",
    "usage_from_response",
    "usage_metadata_folds_cache",
]

logger = logging.getLogger(__name__)

_CACHE_READ_KEY = "cache_read_input_tokens"
_CACHE_WRITE_KEY = "cache_creation_input_tokens"


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts of one or more calls; zero where the provider reports none.

    ``input_tokens`` and ``total_tokens`` include cache reads and writes;
    ``cached_tokens`` (input read from the cache) and ``reasoning_tokens``
    are the reported subsets.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.total_tokens + other.total_tokens,
            self.cached_tokens + other.cached_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
        )

    @property
    def is_empty(self) -> bool:
        """Whether no tokens were counted."""
        return self == Usage()

    def as_dict(self) -> dict[str, int]:
        """Return the counts keyed by field name."""
        return asdict(self)


def coerce_token_count(value: Any) -> int | None:
    """Return a reported token count as ``int``, or None when unusable.

    Floats such as ``123.0`` are rounded; ``bool`` and non-numbers are None.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    return None


def _count(mapping: Mapping[str, Any], key: str) -> int:
    """Return ``mapping[key]`` as a non-negative count, 0 when unusable."""
    return max(0, coerce_token_count(mapping.get(key)) or 0)


def _nested_count(mapping: Mapping[str, Any], outer: str, inner: str) -> int:
    """Return ``mapping[outer][inner]`` as a count, 0 when absent."""
    nested = mapping.get(outer)
    return _count(nested, inner) if isinstance(nested, Mapping) else 0


def _detail_count(mapping: Mapping[str, Any], outer: str, name: str) -> int:
    """Return the ``name`` token detail of ``mapping[outer]``, 0 when absent.

    langchain-openai prefixes detail keys with the service tier for tiers
    other than the default (``flex_cache_read``, ``priority_reasoning``) and
    adds the tier's total under the bare tier name, which is not a detail.
    """
    nested = mapping.get(outer)
    if not isinstance(nested, Mapping):
        return 0
    suffix = f"_{name}"
    return sum(
        _count(nested, key)
        for key in nested
        if isinstance(key, str) and (key == name or key.endswith(suffix))
    )


def usage_metadata_folds_cache(usage_meta: Mapping[str, Any]) -> bool:
    """Whether a LangChain ``usage_metadata`` already includes cache tokens.

    A nested ``input_token_details`` dict means LangChain normalized the
    cache into ``input_tokens`` and ``total_tokens``.
    """
    return isinstance(usage_meta.get("input_token_details"), dict)


def additive_cache_tokens(usage_meta: Mapping[str, Any], target: Any) -> int:
    """Return cache tokens reported beside a cache-exclusive input count.

    Reads ``cache_read_input_tokens`` and ``cache_creation_input_tokens`` from
    the usage metadata, then from the raw provider usage on the message
    (``response_metadata["usage"]`` or a ``usage`` attribute), and returns
    their sum. Call it only when :func:`usage_metadata_folds_cache` is False.
    """
    read, written = _additive_cache(usage_meta, target)
    return read + written


def _additive_cache(usage_meta: Mapping[str, Any], target: Any) -> tuple[int, int]:
    """Return the cache reads and writes beside a cache-exclusive input count."""
    candidates: list[Mapping[str, Any]] = [usage_meta]
    response_metadata = getattr(target, "response_metadata", None)
    if isinstance(response_metadata, dict):
        raw_usage = response_metadata.get("usage")
        if isinstance(raw_usage, dict):
            candidates.append(raw_usage)
    raw_usage_attr = getattr(target, "usage", None)
    if isinstance(raw_usage_attr, dict):
        candidates.append(raw_usage_attr)

    read, written = (
        next((_count(c, key) for c in candidates if _count(c, key)), 0)
        for key in (_CACHE_READ_KEY, _CACHE_WRITE_KEY)
    )
    return read, written


def usage_from_response(response: Any) -> Usage | None:
    """Return the usage of a LangChain response, or None when none is usable.

    Unwraps an ``include_raw`` dict. Prefers ``total_tokens`` and falls back to
    input plus output; adds raw Anthropic cache tokens unless the metadata
    already folds them in. ``cached_tokens`` counts cache reads only
    (``cache_read`` details, tier-prefixed ones included).
    """
    target = unwrap_raw_response(response)
    usage_meta = getattr(target, "usage_metadata", None)
    if not isinstance(usage_meta, dict) or not usage_meta:
        return None

    input_tokens = _count(usage_meta, "input_tokens")
    output_tokens = _count(usage_meta, "output_tokens")
    total = _count(usage_meta, "total_tokens") or input_tokens + output_tokens

    if usage_metadata_folds_cache(usage_meta):
        cache_add = 0
        cached = _detail_count(usage_meta, "input_token_details", "cache_read")
    else:
        cached, written = _additive_cache(usage_meta, target)
        cache_add = cached + written

    if total + cache_add <= 0:
        return None
    return Usage(
        input_tokens=input_tokens + cache_add,
        output_tokens=output_tokens,
        total_tokens=total + cache_add,
        cached_tokens=cached,
        reasoning_tokens=_detail_count(usage_meta, "output_token_details", "reasoning"),
    )


def _raw_usage_from_exception(exc: BaseException) -> Mapping[str, Any] | None:
    """Return the usage dict of ``exc.body`` or ``exc.response.json()``."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and isinstance(body.get("usage"), dict):
        usage: Mapping[str, Any] = body["usage"]
        return usage
    response = getattr(exc, "response", None)
    json_method = getattr(response, "json", None)
    if not callable(json_method):
        return None
    try:
        payload = json_method()
    except (ValueError, RuntimeError):
        logger.debug("Exception response carries no JSON body", exc_info=True)
        return None
    if isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
        found: Mapping[str, Any] = payload["usage"]
        return found
    return None


def _pair(raw: Mapping[str, Any], first: str, second: str) -> tuple[int, int] | None:
    """Return two counts when both are present and their sum is positive."""
    a = coerce_token_count(raw.get(first))
    b = coerce_token_count(raw.get(second))
    if a is None or b is None or a + b <= 0:
        return None
    return a, b


def _usage_from_raw(raw: Mapping[str, Any]) -> Usage | None:
    """Return the usage of a provider usage dict from an error body.

    Raw Anthropic cache reads and writes add to the input and total counts;
    ``cached_tokens`` counts the reads (or OpenAI's cached prompt tokens).
    """
    pair = _pair(raw, "prompt_tokens", "completion_tokens") or _pair(
        raw, "input_tokens", "output_tokens"
    )
    input_tokens, output_tokens = pair or (0, 0)
    total = coerce_token_count(raw.get("total_tokens")) or 0
    if total <= 0:
        total = input_tokens + output_tokens
    cache_read = _count(raw, _CACHE_READ_KEY)
    cache_add = cache_read + _count(raw, _CACHE_WRITE_KEY)
    if total + cache_add <= 0:
        return None
    cached = cache_read or _nested_count(raw, "prompt_tokens_details", "cached_tokens")
    reasoning = _nested_count(
        raw, "completion_tokens_details", "reasoning_tokens"
    ) or _nested_count(raw, "output_tokens_details", "reasoning_tokens")
    return Usage(
        input_tokens=input_tokens + cache_add,
        output_tokens=output_tokens,
        total_tokens=total + cache_add,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
    )


def usage_from_exception(exc: BaseException) -> Usage | None:
    """Recover the usage a failed call reported, or None. Never raises.

    Reads, in order: a ``usage`` attribute holding a :class:`Usage` (set by
    callers that discard a response), an integer ``discarded_total``
    attribute, ``exc.body["usage"]`` and ``exc.response.json()["usage"]``.
    Totals come from ``total_tokens``, else prompt plus completion tokens,
    else input plus output tokens.
    """
    try:
        attached = getattr(exc, "usage", None)
        if isinstance(attached, Usage):
            return None if attached.is_empty else attached
        discarded = coerce_token_count(getattr(exc, "discarded_total", None))
        if discarded is not None and discarded > 0:
            return Usage(total_tokens=discarded)
        raw = _raw_usage_from_exception(exc)
        return _usage_from_raw(raw) if raw is not None else None
    except Exception:  # noqa: BLE001 - usage recovery is best effort
        logger.debug("Token recovery from a failed call failed", exc_info=True)
        return None


class UsageTotals:
    """Usage summed per role, for example one role per request type."""

    def __init__(self) -> None:
        self._by_role: dict[str, Usage] = {}

    def add(self, role: str, usage: Usage | None) -> None:
        """Add ``usage`` to ``role``; None is ignored."""
        if usage is None:
            return
        self._by_role[role] = self._by_role.get(role, Usage()) + usage

    def get(self, role: str) -> Usage:
        """Return the usage of ``role`` (zero when nothing was added)."""
        return self._by_role.get(role, Usage())

    @property
    def roles(self) -> tuple[str, ...]:
        """Roles with recorded usage, in order of first use."""
        return tuple(self._by_role)

    def total(self) -> Usage:
        """Return the sum over all roles."""
        return _sum(self._by_role.values())

    def as_dict(self) -> dict[str, dict[str, int]]:
        """Return the counts per role."""
        return {role: usage.as_dict() for role, usage in self._by_role.items()}


def _sum(values: Iterable[Usage]) -> Usage:
    result = Usage()
    for value in values:
        result = result + value
    return result
