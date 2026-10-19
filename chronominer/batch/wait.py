"""Wait for batches to end: poll their status until each is terminal.

:func:`wait_for_submissions` waits for the provider batches of submissions
of the store; :func:`wait_for_batches` polls given batch handles. Each
status pass runs through :func:`~chronominer.common.polling.
poll_until_settled`. Neither writes anything: finalization is the caller's
next step.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from chronominer.batch.clients import BatchClients, status_of
from chronominer.batch.finalize import submission_handle
from chronominer.common.batch import TERMINAL_STATUSES, BatchHandle
from chronominer.common.polling import WaitLimits, WaitTick, poll_until_settled
from chronominer.common.retry import classify_error
from chronominer.common.store import Store
from chronominer.events import RunEvents, Waiting

__all__ = [
    "POLL_SECONDS",
    "WAIT_TIMEOUT_SECONDS",
    "WaitOutcome",
    "wait_for_batches",
    "wait_for_submissions",
]

logger = logging.getLogger(__name__)

POLL_SECONDS = 10.0
WAIT_TIMEOUT_SECONDS = 7200.0


def _clock() -> float:
    """The wait's clock; tests replace it together with :func:`_sleep`."""
    return time.monotonic()


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


@dataclass(frozen=True)
class WaitOutcome:
    """How a wait ended.

    *running* lists the batches still running at the end; *error* names the
    batch and the non-retryable error that stopped the wait.
    """

    running: list[str] = field(default_factory=list)
    error: str | None = None


async def _status_pass(
    handles: Sequence[BatchHandle], clients: BatchClients, events: RunEvents
) -> WaitOutcome:
    """Read the status of every batch once.

    A retryable error counts the batch as running and is reported as a
    warning; a non-retryable one ends the pass with the error.
    """
    running: list[str] = []
    for handle in handles:
        try:
            info = await clients.call(handle.provider, status_of(handle))
        except Exception as exc:  # noqa: BLE001 - classified just below
            kind = classify_error(exc)
            if not kind.retryable:
                logger.error("Status of batch %s failed: %s", handle.batch_id, exc)
                return WaitOutcome(error=f"{handle.batch_id}: {exc}")
            events.warning(
                f"Could not read the status of batch {handle.batch_id} "
                f"({kind}): {exc}; polling again."
            )
            running.append(handle.batch_id)
            continue
        if info.status not in TERMINAL_STATUSES:
            running.append(handle.batch_id)
    return WaitOutcome(running=running)


def _pending(outcome: WaitOutcome) -> int:
    return 0 if outcome.error is not None else len(outcome.running)


async def wait_for_batches(
    handles: Sequence[BatchHandle],
    clients: BatchClients,
    events: RunEvents,
    *,
    poll_seconds: float = POLL_SECONDS,
    timeout_seconds: float = WAIT_TIMEOUT_SECONDS,
) -> WaitOutcome:
    """Poll *handles* until every batch has ended or the timeout passes.

    A retryable error while reading a status counts the batch as running
    for that poll and is reported as a warning; a non-retryable one stops
    the wait with the error in :attr:`WaitOutcome.error`.
    """

    async def status_pass(_number: int) -> WaitOutcome:
        return await _status_pass(handles, clients, events)

    def waiting(tick: WaitTick) -> None:
        events.waiting(
            Waiting(
                reason=(
                    f"{tick.pending} batch(es) still running "
                    f"({int(tick.elapsed)}s elapsed)"
                ),
                seconds=tick.next_in,
            )
        )

    polled = await poll_until_settled(
        status_pass,
        _pending,
        WaitLimits(poll_seconds, timeout_seconds),
        on_wait=waiting,
        retryable=_never,
        clock=_clock,
        sleep=_sleep,
    )
    if polled.last is None:  # pragma: no cover - every pass returns or raises
        return WaitOutcome(running=[handle.batch_id for handle in handles])
    return polled.last


def _never(_exc: BaseException) -> bool:
    """Let an error that escapes a status pass propagate."""
    return False


async def wait_for_submissions(
    store: Store,
    submission_ids: Sequence[str],
    clients: BatchClients,
    events: RunEvents,
    *,
    poll_seconds: float = POLL_SECONDS,
    timeout_seconds: float = WAIT_TIMEOUT_SECONDS,
) -> WaitOutcome:
    """Poll the provider batches of the store's *submission_ids* until they end.

    Submissions without a provider batch id are left out.
    """
    handles = [
        submission_handle(submission)
        for submission in (store.submission(sid) for sid in submission_ids)
        if submission.batch_id
    ]
    return await wait_for_batches(
        handles,
        clients,
        events,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    )
