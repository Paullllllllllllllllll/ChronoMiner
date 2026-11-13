"""The ``on_error`` decisions of the attempts of one retried call.

A site that retries a call with
:func:`~chronominer.common.retry.call_with_retry` asks each failed
attempt what follows, through ``on_error``, and hands the answer to the
ladder through :meth:`AttemptDecisions.decide`. Decisions are kept per
exception object, so attempts that fail concurrently keep their own.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from chronominer import ext
from chronominer.common.retry import (
    Decision,
    StopSignal,
    default_decision,
    heed_stop,
)

__all__ = ["AttemptDecisions"]


class AttemptDecisions:
    """The decisions of failed attempts, until the retry ladder asks for them.

    *stop* is the run's stop signal, if the call shares one.
    """

    def __init__(self, stop: StopSignal | None = None) -> None:
        self._pending: dict[int, tuple[BaseException, Decision]] = {}
        self._stop = stop

    async def enter[C](
        self,
        stack: contextlib.AsyncExitStack,
        attempt: ext.Attempt[C],
        *,
        on_failure: Callable[[], None] | None = None,
    ) -> ext.Attempt[C]:
        """Enter *attempt* on *stack*; on failure record its decision and raise.

        *on_failure* runs after the decision is recorded.
        """
        try:
            return await stack.enter_async_context(attempt)
        except Exception as exc:
            self.failed(attempt, exc)
            if on_failure is not None:
                on_failure()
            raise

    def failed[C](self, attempt: ext.Attempt[C], exc: Exception) -> None:
        """Record the decision of *attempt* about *exc*, which it raised.

        A ``STOP`` decision sets the run's stop signal at once, before the
        caller records the attempt.
        """
        decision = attempt.on_error(exc)
        heed_stop(self._stop, decision, exc)
        self._pending[id(exc)] = (exc, decision)

    def decide(self, exc: BaseException) -> Decision:
        """Return the recorded decision about *exc*, or the default decision."""
        entry = self._pending.pop(id(exc), None)
        if entry is None:
            return default_decision(exc)
        return entry[1]
