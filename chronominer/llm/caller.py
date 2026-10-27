"""Model calls of one role: structured calls, the retry ladder and attempts.

A :class:`ModelCaller` builds its chat model once, through
:func:`chronominer.ext.attempt`, and sends every request through
:func:`~chronominer.common.structured.structured_call`. Each HTTP
attempt, retries included, runs inside an entered attempt: the attempt's
identity supplies the key variable, ``commit`` records the usage of a
response, ``on_error`` records usage recovered from a failure, on entry or of
the request, and decides what the ladder of
:func:`~chronominer.common.retry.call_with_retry` does next. Committed
and recovered usage is summed per role in :attr:`CallEnv.usage` and in every
open :func:`collect_usage` block; retry waits go to :meth:`RunEvents.waiting`.

Two bounds of the settings apply around the ladder: timeouts are retried at
most ``retry.timeout_attempts`` times in all, and one call, its retries
included, ends with :class:`PageTimeoutError` after ``retry.page_timeout``.
The time an attempt waits on entry and the WAIT pauses of the ladder do not
count against this deadline; requests and RETRY backoff do.
A response with fewer input tokens than ``retry.min_input_tokens`` (image
requests only) and a malformed SDK response count as invalid output, retried
within ``retry.validation_attempts``.

With a recorder installed (:func:`~chronominer.common.capture.use_recorder`),
each HTTP attempt runs in its own capture scope, and its
:class:`~chronominer.common.capture.AttemptRecord` is stored through
the recorder before the call returns, raises or retries.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import itertools
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from chronominer import ext
from chronominer.common.capture import HttpAttempt
from chronominer.common.rate_limit import RateLimiterRegistry
from chronominer.common.retry import (
    Decision,
    ErrorKind,
    RetryEvent,
    RetryHooks,
    call_with_retry,
    classify_error,
)
from chronominer.common.structured import (
    PreparedRequest,
    StructuredRequest,
    StructuredResult,
    default_invoke,
    structured_call,
)
from chronominer.common.usage import Usage, UsageTotals, usage_from_response
from chronominer.events import NullEvents, RunEvents, Waiting
from chronominer.llm.client import build_chat_model
from chronominer.llm.retrying import AttemptDecisions
from chronominer.settings import Settings
from chronominer.spec import ModelSpec

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

__all__ = [
    "CallEnv",
    "HttpAttempt",
    "InputTokensBelowThresholdError",
    "ModelCaller",
    "PageTimeoutError",
    "collect_usage",
]

logger = logging.getLogger(__name__)


class InputTokensBelowThresholdError(Exception):
    """An image request was billed for fewer input tokens than the minimum.

    The provider probably dropped the image, so the answer is not about it.
    """

    def __init__(self, actual: int, threshold: int) -> None:
        self.actual = actual
        self.threshold = threshold
        super().__init__(
            f"Input tokens ({actual}) below minimum threshold ({threshold}); "
            "image payload may have been dropped"
        )


class PageTimeoutError(Exception):
    """One call, its retries included, ran past its wall-clock limit."""

    def __init__(self, label: str, seconds: float) -> None:
        self.label = label
        self.seconds = seconds
        super().__init__(
            f"Page {label!r} exceeded its {seconds:.0f}s wall-clock ceiling "
            "across all retry attempts"
        )


_INVALID_OUTPUT: tuple[type[BaseException], ...] = (
    InputTokensBelowThresholdError,
    TypeError,
    AttributeError,
)
"""Errors retried within ``retry.validation_attempts``, as invalid output."""

_CLIENT_ATTRIBUTES = (
    "root_async_client",
    "root_client",
    "async_client",
    "client",
    "_async_client",
    "_client",
)
"""Attributes under which LangChain chat models keep their SDK clients."""


async def _close_quietly(client: Any) -> None:
    """Call ``client.close()``, awaiting an async closer; never raises."""
    close = getattr(client, "close", None)
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # noqa: BLE001 - closing is best effort
        logger.debug("Error closing client %r: %s", type(client).__name__, exc)


_SCOPE: ContextVar[tuple[UsageTotals, ...]] = ContextVar("usage_scope", default=())


@contextlib.contextmanager
def collect_usage() -> Iterator[UsageTotals]:
    """Collect, per role, the usage of the calls made inside the block.

    The block's task and the tasks it starts share the collector. An
    enclosing block also counts the calls of blocks nested in it, including
    those in child tasks; the run's totals in :attr:`CallEnv.usage` count the
    same calls as well.
    """
    totals = UsageTotals()
    token = _SCOPE.set(_SCOPE.get() + (totals,))
    try:
        yield totals
    finally:
        _SCOPE.reset(token)


@dataclass(frozen=True)
class CallEnv:
    """What the model calls of one run share.

    *limiters* holds one rate limiter per provider, *usage* the token totals
    per role, and *events* receives the retry waits.
    """

    settings: Settings = field(default_factory=Settings)
    limiters: RateLimiterRegistry = field(default_factory=RateLimiterRegistry)
    usage: UsageTotals = field(default_factory=UsageTotals)
    events: RunEvents = field(default_factory=NullEvents)

    @classmethod
    def create(
        cls, settings: Settings | None = None, events: RunEvents | None = None
    ) -> CallEnv:
        """Return a fresh environment with the rate limits of *settings*."""
        settings = settings or Settings()
        return cls(
            settings=settings,
            limiters=RateLimiterRegistry(settings.rate_limits),
            usage=UsageTotals(),
            events=events or NullEvents(),
        )

    def record(self, role: str, usage: Usage) -> None:
        """Add *usage* to the run's totals and every open :func:`collect_usage`."""
        if usage.is_empty:
            return
        self.usage.add(role, usage)
        for scope in _SCOPE.get():
            scope.add(role, usage)

    def caller(
        self,
        role: str,
        model: ModelSpec,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> ModelCaller:
        """Return a new caller of *role* for *model*.

        *overrides* replace constructor kwargs of the chat model.

        Raises what :class:`ModelCaller` raises, for example ``OSError`` when
        the key variable is unset.
        """
        return ModelCaller(role, model, self, overrides=overrides)


class _Deadline:
    """The page watchdog of one call, whose clock can stop during waits."""

    def __init__(self, watchdog: asyncio.Timeout | None = None) -> None:
        self._watchdog = watchdog

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        """Stop the clock in the block; then move the deadline by its length."""
        watchdog = self._watchdog
        when = None if watchdog is None else watchdog.when()
        if watchdog is None or when is None or watchdog.expired():
            yield
            return
        loop = asyncio.get_running_loop()
        started = loop.time()
        watchdog.reschedule(None)
        try:
            yield
        finally:
            watchdog.reschedule(when + loop.time() - started)

    def expired(self) -> bool:
        """Whether the watchdog has fired."""
        return self._watchdog is not None and self._watchdog.expired()


class ModelCaller:
    """Send the requests of one role, ``extract`` or ``adjust``.

    The chat model is built once, here, through an attempt that is not
    entered; an attempt whose identity names another key variable gets a
    client of its own. *overrides* replace constructor kwargs of the chat
    model.
    """

    def __init__(
        self,
        role: str,
        model: ModelSpec,
        env: CallEnv,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.role = role
        self.spec = model
        self.env = env
        self.provider: str = model.provider
        retry = env.settings.retry
        self._policy = retry.policy()
        self._timeout_attempts = retry.timeout_attempts
        self.deadline = retry.page_deadline(env.settings.timeouts.request)
        self._overrides = dict(overrides or {})
        self._clients: dict[str, BaseChatModel] = {}
        self.model: BaseChatModel = self._attempt().build()

    def _attempt(self) -> ext.Attempt[BaseChatModel]:
        return ext.attempt(
            self.role,
            self.provider,
            None,
            settings=self.env.settings,
            build_client=self.client_for,
            endpoint=self.spec.endpoint,
            model=self.spec.model,
        )

    def client_for(self, identity: ext.Identity) -> BaseChatModel:
        """Return the client for an attempt identity, building it if new.

        The client takes the extension's client options of the identity.
        """
        client = self._clients.get(identity.api_key_env)
        if client is None:
            client = build_chat_model(
                self.spec,
                self.env.settings,
                api_key_env=identity.api_key_env,
                overrides=self._overrides,
                options=ext.client_options(identity),
            )
            self._clients[identity.api_key_env] = client
        return client

    async def call(
        self,
        messages: Sequence[Any],
        request: StructuredRequest,
        *,
        label: str,
        min_input_tokens: int = 0,
    ) -> StructuredResult:
        """Request one answer in ``request.response_format``.

        Transient errors are retried by the ladder; the last one propagates.
        *min_input_tokens* above 0 retries answers billed for fewer input
        tokens.

        Raises:
            PageTimeoutError: The call ran past the page deadline.
        """
        numbers = itertools.count(1)
        watchdog = None if self.deadline is None else asyncio.timeout(self.deadline)
        deadline = _Deadline(watchdog)

        async def invoke(prepared: PreparedRequest, sent: Sequence[Any]) -> Any:
            return await self._with_retries(
                prepared, sent, label, min_input_tokens, numbers, deadline
            )

        if watchdog is None or self.deadline is None:
            return await structured_call(self.model, messages, request, invoke=invoke)
        try:
            async with watchdog:
                return await structured_call(
                    self.model, messages, request, invoke=invoke
                )
        except TimeoutError as exc:
            if not watchdog.expired():
                raise
            logger.error(
                "Page watchdog fired for %s after %.0fs (all retry attempts "
                "combined); abandoning this page.",
                label,
                self.deadline,
            )
            raise PageTimeoutError(label, self.deadline) from exc

    def _record(self, usage: Usage) -> None:
        self.env.record(self.role, usage)

    @contextlib.contextmanager
    def _abandonment(
        self, call: ext.Attempt[BaseChatModel], clock: _Deadline
    ) -> Iterator[None]:
        """Report a request the page watchdog cut off to its attempt as a timeout.

        The provider may bill the abandoned request.
        """
        try:
            yield
        except asyncio.CancelledError:
            if clock.expired():
                call.on_error(TimeoutError("page deadline reached"))
                self._record(call.usage)
            raise

    async def _with_retries(
        self,
        prepared: PreparedRequest,
        messages: Sequence[Any],
        label: str,
        min_input_tokens: int = 0,
        numbers: Iterator[int] | None = None,
        deadline: _Deadline | None = None,
    ) -> Any:
        decisions = AttemptDecisions()
        timeouts = 0
        counter = numbers if numbers is not None else itertools.count(1)
        clock = deadline if deadline is not None else _Deadline()
        pending: list[Decision] = []

        async def one_attempt() -> Any:
            attempt = self._attempt()
            async with contextlib.AsyncExitStack() as stack:
                with clock.paused():
                    call = await decisions.enter(
                        stack, attempt, on_failure=lambda: self._record(attempt.usage)
                    )
                http = HttpAttempt(
                    provider=self.provider,
                    model=self.spec.model,
                    role=self.role,
                    attempt=next(counter),
                    label=label,
                )
                try:
                    with http.capturing(), self._abandonment(call, clock):
                        response = await default_invoke(
                            prepared.for_model(call.client), messages
                        )
                except Exception as exc:
                    decisions.failed(call, exc)
                    self._record(call.usage)
                    await http.store(usage=call.usage, error=exc)
                    raise
                usage = usage_from_response(response)
                call.commit(usage)
                self._record(call.usage)
                billed = usage.input_tokens if usage is not None else 0
                short: InputTokensBelowThresholdError | None = None
                if 0 < billed < min_input_tokens:
                    short = InputTokensBelowThresholdError(billed, min_input_tokens)
                await http.store(response=response, usage=usage, error=short)
                if short is not None:
                    raise short
                return response

        def decide(exc: BaseException) -> Decision:
            nonlocal timeouts
            decision = decisions.decide(exc)
            retrying = decision in (Decision.RETRY, Decision.SWITCH_KEY)
            if not retrying or classify_error(exc) is not ErrorKind.TIMEOUT:
                return decision
            timeouts += 1
            if timeouts < self._timeout_attempts:
                return decision
            logger.error(
                "Request timeout attempts exhausted (%d/%d) for %s; not retrying: %s",
                timeouts,
                self._timeout_attempts,
                label,
                str(exc)[:200],
            )
            return Decision.FAIL

        def report(event: RetryEvent) -> None:
            pending[:] = [event.decision]
            self._report_retry(event)

        async def sleep(delay: float) -> None:
            if pending == [Decision.WAIT]:
                with clock.paused():
                    await asyncio.sleep(delay)
            else:
                await asyncio.sleep(delay)

        hooks = RetryHooks(
            on_error=decide,
            on_retry=report,
            gate=self.env.limiters.get(self.provider),
            sleep=sleep,
        )
        return await call_with_retry(
            one_attempt,
            policy=self._policy,
            validation_errors=_INVALID_OUTPUT,
            hooks=hooks,
            label=label,
        )

    async def aclose(self) -> None:
        """Close the SDK clients of every chat model built. Never raises."""
        models, self._clients = list(self._clients.values()), {}
        for model in models:
            for attr in _CLIENT_ATTRIBUTES:
                await _close_quietly(getattr(model, attr, None))

    def _report_retry(self, event: RetryEvent) -> None:
        self.env.events.waiting(
            Waiting(
                reason=(
                    f"{event.label or self.role}: {event.kind.value} on attempt "
                    f"{event.attempt}, {event.decision.value}"
                ),
                seconds=event.delay,
            )
        )
