"""Cancel and list batches of OpenAI, Anthropic and Google.

:func:`cancel_batches` cancels batches by id, the provider inferred from the
id's shape. :func:`list_batches` lists the unfinished batches of every
provider whose key variable is set, and :func:`cancel_unfinished` cancels
them. :func:`cancel_submissions` cancels the submitted batches recorded in
the databases below a folder and marks their submissions cancelled, so
their chunks are pending again. Every call is retried on transient errors;
an error that outlasts the retries is reported for its batch or provider.
Each takes the parsed command line as *args*, if any, for the extension's
key variables.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from chronominer.batch.check import find_databases
from chronominer.batch.clients import BatchClients
from chronominer.common.batch import (
    BATCH_PROVIDERS,
    BatchBackend,
    BatchHandle,
    BatchStatusInfo,
    BatchSummary,
)
from chronominer.common.batch_reconcile import lease_submission, provider_of
from chronominer.common.store import (
    LeaseLostError,
    Store,
    StoreError,
    Submission,
    SubmissionStateError,
)
from chronominer.output.store import open_store, read_store
from chronominer.settings import Settings

__all__ = [
    "CancelOutcome",
    "CancelReport",
    "Listing",
    "cancel_batches",
    "cancel_submissions",
    "cancel_unfinished",
    "list_batches",
    "provider_of",
]

logger = logging.getLogger(__name__)

ROLE = "extract"


@dataclass(frozen=True)
class CancelOutcome:
    """The cancellation of one batch; *error* says why it failed.

    *submission_id* names the store submission of the batch, if known.
    """

    batch_id: str
    provider: str
    cancelled: bool
    error: str | None = None
    submission_id: str | None = None


@dataclass
class Listing:
    """Unfinished batches per provider, and the providers left out.

    *skipped* holds the providers whose key variable is unset; *errors* the
    providers whose listing failed, with the reason.
    """

    batches: list[BatchSummary] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class CancelReport:
    """The outcomes of a cancel command and the listing it worked from.

    *refused* holds the databases the store refused, with the reason;
    *skipped* the submissions of an ``--input`` cancel that have no provider
    batch to cancel or whose batch has already ended.
    """

    outcomes: list[CancelOutcome] = field(default_factory=list)
    listing: Listing | None = None
    refused: dict[Path, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    @property
    def cancelled(self) -> int:
        """The number of batches cancelled."""
        return sum(1 for outcome in self.outcomes if outcome.cancelled)

    @property
    def failed(self) -> int:
        """Failed cancellations plus failed listings."""
        listing_errors = len(self.listing.errors) if self.listing else 0
        return len(self.outcomes) - self.cancelled + listing_errors


def _cancel_of(handle: BatchHandle) -> Callable[[BatchBackend], None]:
    """Return the operation that asks a backend to cancel *handle*.

    The provider's error propagates, so the retry ladder sees it and the
    outcome reports it.
    """
    return lambda backend: backend.request_cancel(handle)


async def _cancel_one(
    clients: BatchClients,
    batch_id: str,
    provider: str | None = None,
    *,
    model: str | None = None,
) -> CancelOutcome:
    provider = provider or provider_of(batch_id)
    handle = BatchHandle(provider=provider, batch_id=batch_id)
    try:
        await clients.call(
            provider,
            _cancel_of(handle),
            model=model,
            label=f"cancel of batch {batch_id}",
        )
    except Exception as exc:  # noqa: BLE001 - a failed cancel is reported
        return CancelOutcome(batch_id, provider, False, str(exc))
    return CancelOutcome(batch_id, provider, True)


async def cancel_batches(
    batch_ids: Sequence[str],
    settings: Settings,
    *,
    args: argparse.Namespace | None = None,
) -> CancelReport:
    """Cancel each batch of *batch_ids* with its provider's key."""
    clients = BatchClients(settings, ROLE, args=args)
    report = CancelReport()
    for batch_id in batch_ids:
        report.outcomes.append(await _cancel_one(clients, batch_id))
    return report


async def list_batches(
    settings: Settings,
    providers: Iterable[str] = BATCH_PROVIDERS,
    *,
    clients: BatchClients | None = None,
    args: argparse.Namespace | None = None,
) -> Listing:
    """List the unfinished batches of each provider whose key variable is set."""
    clients = clients or BatchClients(settings, ROLE, args=args)
    listing = Listing()
    for provider in providers:
        if not clients.has_key(provider):
            listing.skipped.append(provider)
            continue
        try:
            found = await clients.call(
                provider,
                lambda backend: backend.list_active(),
                label=f"listing of {provider} batches",
            )
        except Exception as exc:  # noqa: BLE001 - a failed listing is reported
            listing.errors[provider] = str(exc)
            continue
        listing.batches.extend(found)
    return listing


async def cancel_unfinished(
    settings: Settings,
    providers: Iterable[str] = BATCH_PROVIDERS,
    *,
    args: argparse.Namespace | None = None,
) -> CancelReport:
    """Cancel every unfinished batch :func:`list_batches` finds."""
    clients = BatchClients(settings, ROLE, args=args)
    listing = await list_batches(settings, providers, clients=clients)
    report = CancelReport(listing=listing)
    for summary in listing.batches:
        report.outcomes.append(
            await _cancel_one(clients, summary.batch_id, summary.provider)
        )
    return report


async def _ended(
    clients: BatchClients, submission: Submission
) -> BatchStatusInfo | None:
    """Return the provider status of a submission's batch if it has ended.

    A batch that has ended keeps its submission as it is: ``batch check``
    finalizes a completed one. A status call that fails returns None, and
    the cancel is attempted.
    """
    handle = BatchHandle(
        provider=submission.provider, batch_id=submission.batch_id or ""
    )
    try:
        info = await clients.call(
            submission.provider,
            lambda backend: backend.status(handle),
            model=submission.model,
            label=f"status of batch {handle.batch_id}",
        )
    except Exception as exc:  # noqa: BLE001 - the cancel is attempted anyway
        logger.warning("Status of batch %s unknown: %s", handle.batch_id, exc)
        return None
    return info if info.is_terminal else None


async def _cancel_submission(
    store: Store, submission: Submission, clients: BatchClients
) -> CancelOutcome | str:
    """Cancel one submitted batch and mark its submission cancelled.

    Returns why nothing was cancelled when the batch has already ended.
    """
    batch_id = submission.batch_id or ""
    submission_id = submission.submission_id

    def failed(error: str) -> CancelOutcome:
        return CancelOutcome(batch_id, submission.provider, False, error, submission_id)

    ended = await _ended(clients, submission)
    if ended is not None:
        return (
            f"batch {batch_id} of submission {submission_id} has already ended "
            f"({ended.provider_status or ended.status.value}); run 'chronominer "
            "batch check' to finalize it"
        )
    if not await lease_submission(store, submission_id):
        return failed("a document of the submission is leased by another run")
    outcome = await _cancel_one(
        clients, batch_id, submission.provider, model=submission.model
    )
    if not outcome.cancelled:
        return failed(outcome.error or "not cancelled")
    try:
        await store.mark_cancelled(submission_id)
    except (LeaseLostError, SubmissionStateError) as exc:
        return failed(f"cancelled at the provider, not recorded: {exc}")
    logger.info("Cancelled batch %s of submission %s", batch_id, submission_id)
    return CancelOutcome(batch_id, submission.provider, True, None, submission_id)


async def cancel_submissions(
    directory: Path,
    settings: Settings,
    *,
    args: argparse.Namespace | None = None,
) -> CancelReport:
    """Cancel the submitted batches of every database below *directory*.

    Each cancelled submission is marked cancelled, so its chunks are pending
    again. Submissions without a provider batch (submitting or unresolved)
    and submissions whose batch has already ended at the provider are
    listed in :attr:`CancelReport.skipped`; ``batch check --abandon`` gives
    an unresolved one up, and ``batch check`` finalizes an ended one. A
    database the store refuses is listed in :attr:`CancelReport.refused`.
    """
    clients = BatchClients(settings, ROLE, args=args)
    report = CancelReport()
    for path in find_databases(directory):
        try:
            reader = read_store(path.parent)
        except StoreError as exc:
            report.refused[path] = str(exc)
            continue
        if reader is None:
            continue
        report.skipped.extend(
            f"{path}: submission {submission.submission_id} is {submission.status}"
            for submission in reader.submissions(("submitting", "unresolved"))
        )
        submitted = reader.submissions(("submitted",))
        if not submitted:
            continue
        async with open_store(path.parent) as store:
            await store.begin_run("batch cancel", spec={"directory": str(directory)})
            try:
                async with store.heartbeat():
                    for submission in submitted:
                        outcome = await _cancel_submission(store, submission, clients)
                        if isinstance(outcome, str):
                            report.skipped.append(f"{path}: {outcome}")
                        else:
                            report.outcomes.append(outcome)
            finally:
                await store.end_run(1 if report.failed else 0)
    return report
