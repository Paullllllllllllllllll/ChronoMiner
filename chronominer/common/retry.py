"""Error classification, backoff policy and the async retry ladder.

Provider SDKs wrap transport failures (an ``APIConnectionError`` raised from
an ``httpx.ConnectError``) and some carry the HTTP status only on a wrapped
cause, so the classifier walks the ``__cause__``/``__context__`` chain.
Timeouts are decided before connection failures and both before the HTTP
status, so a gateway timeout with a 5xx status counts as a timeout.

:func:`call_with_retry` is the single retry authority for a model call: SDK
clients are built with ``max_retries=0``. It retries transient errors on the
attempt budget and the caller's validation errors on a separate, smaller
budget, honors ``Retry-After``, recovers usage from failed attempts, and lets
an ``on_error`` hook decide what happens after each transient failure.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import inspect
import logging
import random
import re
import ssl
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, TypeVar

import httpx

from .responses import stop_of
from .usage import Usage, usage_from_exception

__all__ = [
    "Decision",
    "ErrorDecision",
    "ErrorKind",
    "RetryEvent",
    "RetryGate",
    "RetryHooks",
    "RetryPolicy",
    "call_with_retry",
    "classify_error",
    "default_decision",
    "http_status",
    "is_connection_error",
    "is_rate_limit_message",
    "is_server_error_message",
    "is_timeout_error",
    "parse_retry_after",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorKind(StrEnum):
    """Retry class of a failed call; the value keys the backoff multipliers."""

    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    SERVER_ERROR = "server_error"
    OTHER = "other"
    CLIENT_ERROR = "client_error"
    VALIDATION = "validation"

    @property
    def retryable(self) -> bool:
        """Whether the ladder retries this class by default."""
        return self is not ErrorKind.CLIENT_ERROR


class Decision(StrEnum):
    """What the ladder does after a transient failure."""

    RETRY = "retry"
    SWITCH_KEY = "switch_key"
    WAIT = "wait"
    FAIL = "fail"


@dataclass(frozen=True)
class ErrorDecision:
    """A decision with an optional delay in seconds.

    For ``WAIT`` the delay is how long to wait for the reset; ``None`` falls
    back to ``Retry-After`` or the backoff.
    """

    action: Decision
    delay: float | None = None


_MAX_CHAIN_DEPTH = 20

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

# Status codes count only in a status, HTTP or code context or next to a 5xx
# reason phrase, so stray numbers ("line 502", "132429 tokens") never match.
_SERVER_ERROR_CODE_RE = re.compile(
    r"(?:status(?:[ _]?code)?|http|error[ _]?code|code)\s*[:=]?\s*5\d{2}\b"
    r"|\b5\d{2}\b\s*(?:internal server error|server error|bad gateway"
    r"|service unavailable|gateway timeout|origin)",
    re.IGNORECASE,
)
_RATE_LIMIT_CODE_RE = re.compile(
    r"(?:status(?:[ _]?code)?|http|error[ _]?code|code)\s*[:=]?\s*429\b"
    r"|\b429\b\s*(?:too many requests|rate limit)"
    r"|too many requests"
    r"|rate[ _-]?limit",
    re.IGNORECASE,
)
_CONNECTION_RE = re.compile(
    r"connection (?:error|reset|refused|aborted)|connecterror", re.IGNORECASE
)
_CONNECTION_TYPES: tuple[type[BaseException], ...] = (
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx.ProxyError,
    httpx.TimeoutException,
    ssl.SSLError,
    ConnectionError,
)


def _chain(exc: BaseException) -> list[BaseException]:
    """Return ``exc`` and its causes, bounded and cycle-safe."""
    seen: set[int] = set()
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while (
        current is not None
        and id(current) not in seen
        and len(chain) < _MAX_CHAIN_DEPTH
    ):
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def is_timeout_error(exc: BaseException) -> bool:
    """Whether ``exc`` or a cause is a request timeout.

    Narrower than :func:`is_connection_error`: a timed-out request may have
    been billed without returning usage. Matches ``httpx.TimeoutException``,
    the builtin ``TimeoutError`` and SDK timeout classes by name.
    """
    return any(
        isinstance(current, httpx.TimeoutException | TimeoutError)
        or type(current).__name__ in _TIMEOUT_CLASS_NAMES
        for current in _chain(exc)
    )


def is_connection_error(exc: BaseException) -> bool:
    """Whether ``exc`` or a cause is a transport failure (timeouts included).

    Network, remote-protocol, proxy and TLS failures count;
    ``httpx.UnsupportedProtocol`` and ``httpx.LocalProtocolError`` do not,
    since they are configuration or programming errors.
    """
    return any(isinstance(current, _CONNECTION_TYPES) for current in _chain(exc))


def http_status(exc: BaseException) -> int | None:
    """Return the first HTTP status found on ``exc`` or a cause, else None.

    Reads ``status_code``, then ``code``, then ``status`` at each level and
    accepts only integers from 100 to 599; a string ``status`` such as
    ``"RESOURCE_EXHAUSTED"`` is ignored.
    """
    for current in _chain(exc):
        for attr in ("status_code", "code", "status"):
            value = getattr(current, attr, None)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if 100 <= value <= 599:
                return value
    return None


def is_rate_limit_message(message: str) -> bool:
    """Whether an error message describes a rate limit or exhausted quota."""
    lowered = message.lower()
    return (
        bool(_RATE_LIMIT_CODE_RE.search(message))
        or "quota" in lowered
        or "resource_exhausted" in lowered
        or "resource has been exhausted" in lowered
    )


def is_server_error_message(message: str) -> bool:
    """Whether an error message describes a transient server-side failure."""
    lowered = message.lower()
    return (
        bool(_SERVER_ERROR_CODE_RE.search(message))
        or re.search(r"server[ _]?error", lowered) is not None
        or "internal error" in lowered
        or "server had an error" in lowered
        or "overloaded" in lowered
        or "capacity" in lowered
        or re.search(r"\bupstream\b", lowered) is not None
        or "'retryable': true" in lowered
        or '"retryable": true' in lowered
    )


def _classify_status(status: int) -> ErrorKind | None:
    """Return the class an HTTP status decides, or None for 1xx to 3xx."""
    if status == 429:
        return ErrorKind.RATE_LIMIT
    if status >= 500:
        return ErrorKind.SERVER_ERROR
    if status in (408, 409):
        return ErrorKind.OTHER
    if status >= 400:
        return ErrorKind.CLIENT_ERROR
    return None


def classify_error(exc: BaseException) -> ErrorKind:
    """Return the retry class of a failed call.

    Order: an answer cut at the output-token limit or stopped by an in-body
    content filter (:func:`~.responses.stop_of`), a non-retryable client
    error; timeout, connection failure, HTTP status (an in-body error's
    code, see :class:`~.responses.ProviderResponseError`), then message
    patterns. 429 is a rate limit, 5xx a server error, 408 and 409 other
    retryable errors, and every other 4xx a non-retryable client error.
    """
    if stop_of(exc) is not None:
        return ErrorKind.CLIENT_ERROR
    if is_timeout_error(exc):
        return ErrorKind.TIMEOUT
    if is_connection_error(exc):
        return ErrorKind.CONNECTION
    status = http_status(exc)
    if status is not None:
        by_status = _classify_status(status)
        if by_status is not None:
            return by_status

    message = f"{type(exc).__name__}: {exc}"
    lowered = message.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return ErrorKind.TIMEOUT
    if _CONNECTION_RE.search(message):
        return ErrorKind.CONNECTION
    if is_rate_limit_message(message):
        return ErrorKind.RATE_LIMIT
    if is_server_error_message(message):
        return ErrorKind.SERVER_ERROR
    return ErrorKind.CLIENT_ERROR


def default_decision(exc: BaseException) -> Decision:
    """Return RETRY for a retryable error and FAIL otherwise."""
    return Decision.RETRY if classify_error(exc).retryable else Decision.FAIL


def _header(headers: Any, name: str) -> str | None:
    """Return a header value by name, trying the lower-case and title forms."""
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    for key in (name, name.title()):
        value = getter(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def parse_retry_after(
    exc: BaseException | None, *, now: dt.datetime | None = None
) -> float | None:
    """Return the delay in seconds a server asked for, or None.

    Reads ``retry-after-ms`` and ``retry-after`` from the headers of
    ``exc.response`` or ``exc``; ``retry-after`` may be seconds or an HTTP
    date. Negative values become 0; unusable values give None.
    """
    if exc is None:
        return None
    try:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if headers is None:
            headers = getattr(exc, "headers", None)
        if headers is None:
            return None
        millis = _header(headers, "retry-after-ms")
        if millis is not None:
            try:
                return max(0.0, float(millis) / 1000.0)
            except ValueError:
                pass
        value = _header(headers, "retry-after")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
        target = parsedate_to_datetime(value)
        if target.tzinfo is None:
            target = target.replace(tzinfo=dt.UTC)
        reference = now or dt.datetime.now(dt.UTC)
        return max(0.0, (target - reference).total_seconds())
    except (AttributeError, TypeError, ValueError, OverflowError):
        logger.debug("Unusable Retry-After header", exc_info=True)
        return None


DEFAULT_BACKOFF_MULTIPLIERS: Mapping[str, float] = MappingProxyType(
    {
        ErrorKind.RATE_LIMIT.value: 2.0,
        ErrorKind.TIMEOUT.value: 1.5,
        ErrorKind.CONNECTION.value: 1.5,
        ErrorKind.SERVER_ERROR.value: 2.0,
        ErrorKind.OTHER.value: 2.0,
        ErrorKind.VALIDATION.value: 1.5,
    }
)


def _as_int(value: Any, default: int) -> int:
    """Return ``int(value)``, or ``default`` when it does not convert."""
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    """Return ``float(value)``, or ``default`` when it does not convert."""
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RetryPolicy:
    """Attempt budgets and backoff: ``base * multiplier**n + jitter``, capped.

    ``max_attempts`` counts every call, the first included;
    ``validation_attempts`` bounds the calls that end in a validation error.
    ``backoff_cap`` also caps a server's ``Retry-After``.
    """

    max_attempts: int = 8
    validation_attempts: int = 3
    backoff_base: float = 0.5
    backoff_cap: float = 120.0
    multipliers: Mapping[str, float] = field(
        default_factory=lambda: DEFAULT_BACKOFF_MULTIPLIERS
    )
    jitter_min: float = 0.5
    jitter_max: float = 1.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> RetryPolicy:
        """Build a policy from a settings mapping; bad values keep defaults.

        Keys: ``max_attempts``, ``validation_attempts``, ``backoff_base``,
        ``backoff_cap``, ``backoff_multipliers`` and ``jitter`` (``min``,
        ``max``).
        """
        base = cls()
        cfg: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
        multipliers = dict(base.multipliers)
        raw_multipliers = cfg.get("backoff_multipliers")
        if isinstance(raw_multipliers, Mapping):
            for key, value in raw_multipliers.items():
                multipliers[str(key)] = _as_float(value, multipliers.get(key, 2.0))
        jitter = cfg.get("jitter")
        jitter_cfg: Mapping[str, Any] = jitter if isinstance(jitter, Mapping) else {}
        return cls(
            max_attempts=max(1, _as_int(cfg.get("max_attempts"), base.max_attempts)),
            validation_attempts=max(
                1,
                _as_int(cfg.get("validation_attempts"), base.validation_attempts),
            ),
            backoff_base=_as_float(cfg.get("backoff_base"), base.backoff_base),
            backoff_cap=_as_float(cfg.get("backoff_cap"), base.backoff_cap),
            multipliers=MappingProxyType(multipliers),
            jitter_min=_as_float(jitter_cfg.get("min"), base.jitter_min),
            jitter_max=_as_float(jitter_cfg.get("max"), base.jitter_max),
        )

    def backoff(self, attempt: int, kind: str, *, jitter: float | None = None) -> float:
        """Return the wait before retry ``attempt`` (0-based) of class ``kind``.

        Unknown classes use multiplier 2.0; the exponent is clamped so a large
        attempt number cannot overflow before the cap applies.
        """
        multiplier = self.multipliers.get(str(kind), 2.0)
        if jitter is None:
            jitter = random.uniform(self.jitter_min, self.jitter_max)
        try:
            raw = self.backoff_base * multiplier ** min(max(attempt, 0), 64) + jitter
        except OverflowError:
            raw = self.backoff_cap
        return max(0.0, min(self.backoff_cap, raw))

    def delay(
        self,
        attempt: int,
        kind: str,
        retry_after: float | None = None,
        *,
        jitter: float | None = None,
    ) -> float:
        """Return the backoff raised to ``retry_after``, both under the cap."""
        wait = self.backoff(attempt, kind, jitter=jitter)
        if retry_after is not None:
            wait = min(self.backoff_cap, max(wait, retry_after))
        return wait


@dataclass(frozen=True)
class RetryEvent:
    """One scheduled retry, reported before the ladder sleeps."""

    attempt: int
    kind: ErrorKind
    decision: Decision
    delay: float
    error: BaseException
    label: str = ""


class RetryGate(Protocol):
    """Admission control called around every attempt, such as a rate limiter."""

    async def acquire(self) -> Any:
        """Wait for capacity before an attempt."""

    def report_success(self) -> None:
        """Record an attempt that reached the provider and returned."""

    def report_error(self, is_rate_limit: bool = False) -> None:
        """Record a failed attempt."""


OnError = Callable[
    [BaseException],
    "Decision | ErrorDecision | Awaitable[Decision | ErrorDecision]",
]


@dataclass(frozen=True)
class RetryHooks:
    """Optional collaborators of :func:`call_with_retry`.

    ``on_error`` decides after each transient failure; without it the
    classifier decides. ``on_usage`` receives the usage recovered from each
    failed attempt; leave it unset when ``on_error`` records usage itself.
    ``on_retry`` sees each scheduled retry before the sleep. ``gate`` is
    acquired before and told the outcome after every attempt. ``sleep``
    defaults to ``asyncio.sleep``.
    """

    on_error: OnError | None = None
    on_usage: Callable[[Usage], None] | None = None
    on_retry: Callable[[RetryEvent], None] | None = None
    gate: RetryGate | None = None
    sleep: Callable[[float], Awaitable[None]] | None = None


@dataclass
class _Ladder:
    """Budgets and counters of one :func:`call_with_retry` run."""

    policy: RetryPolicy
    hooks: RetryHooks
    label: str
    attempts: int = 0
    transient_failures: int = 0
    validation_failures: int = 0

    def _recover_usage(self, exc: BaseException) -> None:
        if self.hooks.on_usage is None:
            return
        usage = usage_from_exception(exc)
        if usage is not None:
            self.hooks.on_usage(usage)

    def _event(
        self, kind: ErrorKind, decision: Decision, delay: float, exc: BaseException
    ) -> RetryEvent:
        return RetryEvent(self.attempts, kind, decision, delay, exc, self.label)

    def after_validation_error(self, exc: BaseException) -> RetryEvent | None:
        """Return the retry after invalid output, or None when out of budget."""
        if self.hooks.gate is not None:
            self.hooks.gate.report_success()
        self._recover_usage(exc)
        self.validation_failures += 1
        if (
            self.validation_failures >= self.policy.validation_attempts
            or self.attempts >= self.policy.max_attempts
        ):
            return None
        kind = ErrorKind.VALIDATION
        delay = self.policy.delay(self.validation_failures - 1, kind)
        return self._event(kind, Decision.RETRY, delay, exc)

    async def after_error(self, exc: BaseException) -> RetryEvent | None:
        """Return the retry after a failed call, or None when it must fail."""
        kind = classify_error(exc)
        retry_after = parse_retry_after(exc)
        if self.hooks.gate is not None:
            self.hooks.gate.report_error(
                is_rate_limit=kind in (ErrorKind.RATE_LIMIT, ErrorKind.SERVER_ERROR)
                or retry_after is not None
            )
        self._recover_usage(exc)
        decision = await _decide(self.hooks.on_error, exc, kind)
        if decision.action is Decision.FAIL:
            return None
        if decision.action is Decision.WAIT:
            self.attempts -= 1
            delay = decision.delay
            if delay is None:
                delay = self.policy.delay(self.transient_failures, kind, retry_after)
            return self._event(kind, decision.action, delay, exc)
        self.transient_failures += 1
        if self.attempts >= self.policy.max_attempts:
            return None
        delay = 0.0
        if decision.action is Decision.RETRY:
            delay = self.policy.delay(self.transient_failures - 1, kind, retry_after)
        return self._event(kind, decision.action, delay, exc)

    async def pause(self, event: RetryEvent) -> None:
        """Report the retry, then sleep its delay."""
        logger.warning(
            "Retrying %s after %s on attempt %d/%d (%s): %s; waiting %.1fs",
            event.label or "call",
            event.kind.value,
            event.attempt,
            self.policy.max_attempts,
            event.decision.value,
            f"{type(event.error).__name__}: {str(event.error)[:200]}",
            event.delay,
        )
        if self.hooks.on_retry is not None:
            self.hooks.on_retry(event)
        if event.delay > 0:
            await (self.hooks.sleep or asyncio.sleep)(event.delay)


async def _decide(
    on_error: OnError | None, exc: BaseException, kind: ErrorKind
) -> ErrorDecision:
    """Ask the hook, or the classifier, what to do after ``exc``."""
    if on_error is None:
        return ErrorDecision(Decision.RETRY if kind.retryable else Decision.FAIL)
    result = on_error(exc)
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, ErrorDecision):
        return result
    return ErrorDecision(Decision(result))


async def call_with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    validation_errors: tuple[type[BaseException], ...] = (),
    hooks: RetryHooks | None = None,
    label: str = "",
) -> T:
    """Await ``call()`` until it succeeds or a budget or decision ends it.

    Transient errors consume ``policy.max_attempts``. Instances of
    ``validation_errors`` (unparseable or invalid output) also consume
    ``policy.validation_attempts`` and bypass ``on_error``. After any other
    failure the decision applies: RETRY backs off, SWITCH_KEY retries at once
    (the hook has switched the key), WAIT sleeps for the reset without using
    an attempt, FAIL re-raises. Cancellation is never retried; the last error
    is re-raised.
    """
    ladder = _Ladder(policy or RetryPolicy(), hooks or RetryHooks(), label)
    gate = ladder.hooks.gate
    while True:
        if gate is not None:
            await gate.acquire()
        ladder.attempts += 1
        try:
            result = await call()
        except validation_errors as exc:
            event = ladder.after_validation_error(exc)
            if event is None:
                raise
        except Exception as exc:
            event = await ladder.after_error(exc)
            if event is None:
                raise
        else:
            if gate is not None:
                gate.report_success()
            return result
        await ladder.pause(event)
