"""Batch check: finalize the batch submissions below a folder.

:func:`check_submissions` finds every ``chronominer.sqlite`` below a
folder and works its open submissions. A submission left ``submitting``
without a provider batch id is marked unresolved first, and every
unresolved submission is reconciled with the provider's batch list (see
:mod:`chronominer.common.batch_reconcile`); ``--adopt`` and ``--abandon``
settle one by hand. Each submitted submission is checked and, once its
batch has finished, finalized into the store
(:func:`~chronominer.batch.finalize.finalize_submission`) under the
leases of its documents. The finalized documents are then rendered
(:func:`~chronominer.batch.finalize.render_documents`) while the leases
are held, in the formats of the run that submitted them unless
``--formats`` names others, and the batch's input and result files are
deleted from the provider. ``--schema`` limits the check to the
submissions of documents extracted with that schema.

:func:`check_until_settled` repeats the check with ``--wait`` until no
submission a check can advance is pending, or its deadline has passed. A
:class:`WaitLedger` records what every pass saw, and :func:`final_totals`
counts those submissions by their current state in the store.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from chronominer.batch.clients import BatchClients
from chronominer.batch.finalize import (
    FAILED,
    FINALIZED_OUTCOME,
    PENDING,
    SKIPPED,
    SubmissionFinalization,
    delete_provider_files,
    document_schema,
    finalize_submission,
    render_documents,
)
from chronominer.common.batch import BatchListing
from chronominer.common.batch_reconcile import (
    abandon,
    adopt,
    count_matches,
    lease_submission,
    list_candidates,
    provider_of,
    reconcile,
)
from chronominer.common.polling import (
    PollResult,
    WaitLimits,
    WaitTick,
    poll_until_settled,
)
from chronominer.common.retry import classify_error
from chronominer.common.store import (
    LeaseLostError,
    Store,
    StoreError,
    Submission,
    SubmissionStateError,
)
from chronominer.common.usage import Usage
from chronominer.output.store import (
    DATABASE_NAME,
    KIND_CHUNK,
    open_store,
    read_store,
)
from chronominer.settings import Settings
from chronominer.spec import SpecError, resolve_schema

__all__ = [
    "ABANDONED",
    "ERROR",
    "INTERRUPTED",
    "UNRESOLVED",
    "CheckResult",
    "CheckStats",
    "CheckTargetError",
    "SubmissionCheck",
    "WaitLedger",
    "check_outputs",
    "check_submissions",
    "check_until_settled",
    "check_usage",
    "failed_units",
    "final_totals",
    "find_databases",
    "pending_work",
    "run_formats",
]

logger = logging.getLogger(__name__)

ROLE = "extract"

UNRESOLVED: Final = "unresolved"
ABANDONED: Final = "abandoned"
_WORKED: Final = ("submitting", "submitted", UNRESOLVED)

INTERRUPTED: Final = "interrupted"
"""How a wait ended on Ctrl+C, besides ``settled`` and ``timeout``."""
ERROR: Final = "error"
"""How a wait ended on an error out of a pass that is not retried."""

_clock: Callable[[], float] = time.monotonic
"""The wait's clock; tests replace it together with :data:`_sleep`."""
_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


@dataclass
class CheckStats:
    """How many submissions ended in each outcome.

    ``skipped`` counts submissions that were no longer submitted;
    ``unresolved`` the submissions that wait for an adoption or an
    abandonment.
    """

    finalized: int = 0
    pending: int = 0
    failed: int = 0
    skipped: int = 0
    unresolved: int = 0
    abandoned: int = 0

    @property
    def had_failure(self) -> bool:
        """Whether something failed or waits for the user; exit code 1."""
        return self.failed > 0 or self.unresolved > 0

    def count(self, outcome: str) -> None:
        """Count one outcome."""
        if outcome == FINALIZED_OUTCOME:
            self.finalized += 1
        elif outcome == PENDING:
            self.pending += 1
        elif outcome == FAILED:
            self.failed += 1
        elif outcome == SKIPPED:
            self.skipped += 1
        elif outcome == UNRESOLVED:
            self.unresolved += 1
        elif outcome == ABANDONED:
            self.abandoned += 1


@dataclass(frozen=True)
class SubmissionCheck:
    """What one check did with one submission.

    *outcome* is ``finalized``, ``pending``, ``failed``, ``unresolved`` or
    ``abandoned``; *detail* says why. *outputs* are the rendered files of
    the finalized documents, *complete* those documents left with no
    pending or failed chunk, *usage* the tokens of the finalized results,
    *errors* the units finalized as error placeholders and *candidates* the
    batches an unresolved submission may belong to.
    """

    database: Path
    submission_id: str
    provider: str
    outcome: str
    batch_id: str | None = None
    detail: str | None = None
    errors: int = 0
    outputs: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()
    candidates: tuple[BatchListing, ...] = ()
    complete: int = 0
    usage: Usage = field(default_factory=Usage)


class CheckTargetError(ValueError):
    """``--adopt`` or ``--abandon`` names nothing the check can act on."""


@dataclass
class CheckResult:
    """The counts and the per-submission results of one check.

    *submissions* are the per-submission results and *refused* the
    databases the store refused (versionless, newer or unreadable), with
    the reason.
    """

    stats: CheckStats = field(default_factory=CheckStats)
    submissions: list[SubmissionCheck] = field(default_factory=list)
    refused: dict[Path, str] = field(default_factory=dict)

    def add(self, check: SubmissionCheck) -> None:
        """Record and count one submission's result."""
        self.submissions.append(check)
        self.stats.count(check.outcome)

    @property
    def outputs(self) -> list[Path]:
        """The files the check rendered, in check order, each once."""
        return check_outputs(self.submissions)

    @property
    def complete(self) -> int:
        """The documents the check left complete."""
        return sum(check.complete for check in self.submissions)

    @property
    def had_failure(self) -> bool:
        """Whether the check should exit with code 1.

        True when a submission failed, waits for the user or finalized a
        failed chunk (:func:`failed_units`).
        """
        return self.stats.had_failure or failed_units(self.submissions) > 0

    @property
    def usage(self) -> dict[str, Usage]:
        """The tokens of the finalized results, under the role ``extract``."""
        return check_usage(self.submissions)


def failed_units(checks: Sequence[SubmissionCheck]) -> int:
    """Return the units *checks* finalized as error placeholders."""
    return sum(check.errors for check in checks)


def check_outputs(checks: Sequence[SubmissionCheck]) -> list[Path]:
    """Return the files *checks* rendered, in order, each once."""
    return list(dict.fromkeys(path for check in checks for path in check.outputs))


def check_usage(checks: Sequence[SubmissionCheck]) -> dict[str, Usage]:
    """Return the tokens of the results *checks* finalized, per role."""
    finalized = [check for check in checks if check.outcome == FINALIZED_OUTCOME]
    if not finalized:
        return {}
    total = Usage()
    for check in finalized:
        total = total + check.usage
    return {ROLE: total}


def find_databases(directory: Path) -> list[Path]:
    """Return the tool's databases at or below *directory*, sorted."""
    return sorted(
        path for path in Path(directory).rglob(DATABASE_NAME) if path.is_file()
    )


@dataclass
class _Database:
    """A database found by the check, read before any write."""

    path: Path
    reader: Store
    submissions: list[Submission]

    @property
    def root(self) -> Path:
        return self.path.parent


@dataclass(frozen=True)
class _Targets:
    """The submissions ``--adopt`` and ``--abandon`` name: (database, id)."""

    adopt: tuple[Path, str, str] | None = None
    abandon: tuple[Path, str] | None = None

    def adopts(self, path: Path, submission_id: str) -> str | None:
        """Return the batch to adopt for this submission, if any."""
        if self.adopt is not None and self.adopt[:2] == (path, submission_id):
            return self.adopt[2]
        return None


def _read_databases(directory: Path, result: CheckResult) -> list[_Database]:
    found = []
    for path in find_databases(directory):
        try:
            reader = read_store(path.parent)
        except StoreError as exc:
            result.refused[path] = str(exc)
            logger.error("Batch check skips %s: %s", path, result.refused[path])
            continue
        if reader is not None:
            found.append(_Database(path, reader, reader.submissions(_WORKED)))
    return found


def _unsettled(submission: Submission) -> bool:
    """Whether *submission* is unresolved or was left without a batch id."""
    return submission.status == UNRESOLVED or (
        submission.status == "submitting" and submission.batch_id is None
    )


def _abandon_target(
    found: Sequence[_Database], submission_id: str, directory: Path
) -> tuple[Path, str]:
    for database in found:
        try:
            submission = database.reader.submission(submission_id)
        except KeyError:
            continue
        if not (_unsettled(submission) or submission.status == "submitted"):
            raise CheckTargetError(
                f"Submission {submission_id} is {submission.status}; only an "
                "unresolved or submitted submission can be abandoned"
            )
        return database.path, submission_id
    raise CheckTargetError(
        f"No submission {submission_id} in the databases below {directory}"
    )


async def _fitting(
    choices: Sequence[tuple[_Database, Submission]],
    batch_id: str,
    clients: BatchClients,
) -> list[tuple[_Database, Submission]]:
    """Keep the submissions whose time window holds *batch_id*.

    Those with the batch's request count win over the others.
    """
    fitting: list[tuple[_Database, Submission, bool]] = []
    for database, submission in choices:
        try:
            candidates = await list_candidates(submission, clients)
        except Exception as exc:
            raise CheckTargetError(
                f"Cannot list the provider's batches: {exc}"
            ) from exc
        listing = next((c for c in candidates if c.batch_id == batch_id), None)
        if listing is not None:
            fitting.append((database, submission, count_matches(listing, submission)))
    counted = [entry for entry in fitting if entry[2]]
    return [(database, submission) for database, submission, _ in counted or fitting]


async def _adopt_target(
    found: Sequence[_Database],
    batch_id: str,
    clients: BatchClients,
    directory: Path,
) -> tuple[Path, str, str]:
    provider = provider_of(batch_id)
    for database in found:
        for submission in database.submissions:
            if submission.batch_id == batch_id:
                raise CheckTargetError(
                    f"Batch {batch_id} already belongs to submission "
                    f"{submission.submission_id}"
                )
    choices = [
        (database, submission)
        for database in found
        for submission in database.submissions
        if submission.provider == provider and _unsettled(submission)
    ]
    if not choices:
        raise CheckTargetError(
            f"No unresolved {provider} submission below {directory} to adopt "
            f"batch {batch_id} into"
        )
    if len(choices) > 1:
        choices = await _fitting(choices, batch_id, clients)
    if len(choices) != 1:
        ids = ", ".join(submission.submission_id for _, submission in choices)
        raise CheckTargetError(
            f"Batch {batch_id} fits {len(choices)} unresolved submissions"
            + (f" ({ids})" if ids else "")
            + "; abandon the others first"
        )
    database, submission = choices[0]
    return database.path, submission.submission_id, batch_id


def _schema_name(argument: str) -> str:
    """Return the schema name ``--schema`` names; the argument when unknown."""
    try:
        return resolve_schema(argument).name
    except SpecError:
        return argument


def _of_schema(store: Store, submission: Submission, schema: str | None) -> bool:
    """Whether a document of *submission* was extracted with schema *schema*."""
    if schema is None:
        return True
    doc_ids = {
        key.doc_id
        for key in store.submission_requests(submission.submission_id).values()
    }
    return any(document_schema(store, doc_id).name == schema for doc_id in doc_ids)


def run_formats(store: Store, run_id: str | None) -> tuple[str, ...]:
    """Return the ``--formats`` the run *run_id* recorded in its spec.

    A run without them (a repair) has none.
    """
    row = store.run(run_id) if run_id else None
    spec: Any = row.get("spec") if row else None
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except ValueError:
            return ()
    entry = spec.get("output.formats") if isinstance(spec, Mapping) else None
    value = entry.get("value") if isinstance(entry, Mapping) else None
    if not isinstance(value, list):
        return ()
    return tuple(str(name) for name in value)


def _complete(store: Store, doc_ids: Sequence[int]) -> int:
    """Return how many of the documents have no pending or failed chunk."""
    count = 0
    for doc_id in doc_ids:
        state = store.resume_state(doc_id)
        if state is None:
            continue
        open_units = (*state.pending, *state.failed)
        if not any(unit.kind == KIND_CHUNK for unit in open_units):
            count += 1
    return count


async def _finalize(  # noqa: PLR0913 - the submission and how to render it
    store: Store,
    path: Path,
    submission: Submission,
    clients: BatchClients,
    formats: Sequence[str] | None,
    note: str | None,
) -> SubmissionCheck:
    """Finalize a submitted submission, render it and delete its provider files.

    *formats* None renders the formats of the run that submitted it. The
    files are deleted only once the results are stored and rendered.
    """
    finalization: SubmissionFinalization = await finalize_submission(
        store, submission, clients
    )
    detail = "; ".join(part for part in (note, finalization.detail) if part) or None
    if finalization.outcome != FINALIZED_OUTCOME:
        return SubmissionCheck(
            path,
            submission.submission_id,
            submission.provider,
            finalization.outcome,
            submission.batch_id,
            detail,
        )
    chosen = run_formats(store, submission.run_id) if formats is None else formats
    documents = finalization.documents
    outputs, warnings = await render_documents(store, documents, chosen)
    await delete_provider_files(clients, submission)
    return SubmissionCheck(
        path,
        submission.submission_id,
        submission.provider,
        FINALIZED_OUTCOME,
        submission.batch_id,
        detail,
        errors=finalization.errors,
        outputs=tuple(outputs),
        warnings=tuple(warnings),
        complete=_complete(store, documents),
        usage=finalization.usage,
    )


async def _settle(
    store: Store,
    submission: Submission,
    clients: BatchClients,
    batch_id: str | None,
) -> tuple[Submission | None, str | None, tuple[BatchListing, ...]]:
    """Adopt *batch_id*, else reconcile; return the submission once submitted.

    Without a submitted submission the detail and the candidates explain
    what the user can do.
    """
    if batch_id is not None:
        adopted = await adopt(store, submission, batch_id, clients)
        return adopted, f"adopted batch {batch_id}", ()
    found = await reconcile(
        store, submission, clients, program="chronominer", units="chunks"
    )
    if found.adopted is None:
        return None, found.detail, found.candidates
    return found.submission, found.detail, ()


async def _check_one(
    store: Store,
    path: Path,
    submission: Submission,
    clients: BatchClients,
    formats: Sequence[str] | None,
    targets: _Targets,
) -> SubmissionCheck:
    submission_id = submission.submission_id

    def result(outcome: str, detail: str | None = None) -> SubmissionCheck:
        return SubmissionCheck(
            path,
            submission_id,
            submission.provider,
            outcome,
            submission.batch_id,
            detail,
        )

    abandoning = targets.abandon == (path, submission_id)
    batch_id = targets.adopts(path, submission_id)
    if submission.status == "submitting" and not abandoning and batch_id is None:
        return result(PENDING, "another run is submitting it")
    if not await lease_submission(store, submission_id):
        return result(PENDING, "a document of the submission is leased by another run")
    try:
        if abandoning:
            await abandon(store, submission, clients)
            return result(ABANDONED, "its chunks are pending again")
        note: str | None = None
        if submission.status != "submitted":
            try:
                settled, note, candidates = await _settle(
                    store, submission, clients, batch_id
                )
            except LeaseLostError:
                raise
            except Exception as exc:  # noqa: BLE001 - a failure is reported
                action = f"batch {batch_id} not adopted" if batch_id else "not settled"
                return result(FAILED, f"{action}: {exc}")
            if settled is None:
                return replace(result(UNRESOLVED, note), candidates=candidates)
            submission = settled
        return await _finalize(store, path, submission, clients, formats, note)
    except LeaseLostError as exc:
        return result(FAILED, f"the lease was lost: {exc}")
    except SubmissionStateError as exc:
        return result(FAILED, str(exc))


async def _check_database(
    database: _Database,
    clients: BatchClients,
    formats: Sequence[str] | None,
    wanted: set[str],
    targets: _Targets,
    result: CheckResult,
    schema: str | None = None,
) -> None:
    """Mark, reconcile and finalize the submissions of one database."""
    async with open_store(database.root) as store:
        await store.begin_run(
            "batch check",
            spec={
                "directory": str(database.root),
                "formats": None if formats is None else list(formats),
                "batch_ids": sorted(wanted),
                "schema": schema,
            },
        )
        checks: list[SubmissionCheck] = []
        try:
            async with store.heartbeat():
                for submission_id in await store.mark_stale_submissions():
                    logger.warning(
                        "Submission %s in %s was left without a provider batch id; "
                        "it is unresolved",
                        submission_id,
                        database.path,
                    )
                for submission in store.submissions(_WORKED):
                    abandoning = targets.abandon == (
                        database.path,
                        submission.submission_id,
                    )
                    if (
                        submission.status == "submitted"
                        and not abandoning
                        and (
                            (wanted and submission.batch_id not in wanted)
                            or not _of_schema(store, submission, schema)
                        )
                    ):
                        continue
                    check = await _check_one(
                        store, database.path, submission, clients, formats, targets
                    )
                    checks.append(check)
                    result.add(check)
        finally:
            failed = failed_units(checks) > 0 or any(
                check.outcome in (FAILED, UNRESOLVED) for check in checks
            )
            await store.end_run(1 if failed else 0)


async def check_submissions(  # noqa: PLR0913 - one keyword per check option
    directory: Path,
    settings: Settings,
    *,
    formats: Sequence[str] | None = None,
    batch_ids: Sequence[str] = (),
    adopt: str | None = None,
    abandon: str | None = None,
    schema: str | None = None,
    args: argparse.Namespace | None = None,
    clients: BatchClients | None = None,
    result: CheckResult | None = None,
) -> CheckResult:
    """Check the open batch submissions of every database below *directory*.

    A finalized document's ``_output.json`` is rendered, and its CSV when
    *formats* holds ``csv`` or the document was rendered as CSV before;
    *formats* None takes the formats of the run that submitted it.
    *batch_ids*, when given, limit the finalization to those batches, and
    *schema* (a ``--schema`` argument) to the submissions of documents
    extracted with that schema; unresolved submissions are reconciled in
    every check. *adopt* records a
    provider batch for the unresolved submission it fits, *abandon* gives an
    unresolved submission up. A database the store refuses is listed in
    :attr:`CheckResult.refused` and left alone. Only databases with a
    submission to work are opened for writing. *args* holds the parsed
    command line, if any, for the extension's key variables. *clients* are
    the batch clients to use (new ones by default); *result*, when given,
    is filled as the check goes and returned, so its caller keeps what was
    checked before an exception.

    Raises:
        CheckTargetError: *adopt* or *abandon* fits no submission, or more
            than one.
    """
    directory = Path(directory)
    result = result if result is not None else CheckResult()
    if clients is None:
        clients = BatchClients(settings, ROLE, args=args)
    found = _read_databases(directory, result)
    targets = _Targets(
        adopt=(
            await _adopt_target(found, adopt, clients, directory) if adopt else None
        ),
        abandon=_abandon_target(found, abandon, directory) if abandon else None,
    )
    wanted = set(batch_ids)
    name = _schema_name(schema) if schema is not None else None
    for database in found:
        if not database.submissions:
            continue
        chosen = None if formats is None else tuple(formats)
        await _check_database(database, clients, chosen, wanted, targets, result, name)
    return result


def pending_work(result: CheckResult) -> int:
    """Return the submissions a later check can still advance.

    These are the pending ones: still running, a retryable provider error,
    another run submitting, a document leased elsewhere. Failed, unresolved
    and filtered submissions do not keep a wait alive.
    """
    return result.stats.pending


_Key = tuple[Path, str]


@dataclass
class WaitLedger:
    """What the passes of a ``--wait`` check saw.

    *seen* maps each submission (database, submission id) to its latest
    check, *failed* holds the submissions some pass reported as failed,
    *checks* every check of every pass in order and *refused* the databases
    any pass refused. *polls* counts the passes that ended, *started* is
    the wait's clock at its start and *stopped* says why the wait ended:
    ``settled``, ``timeout``, :data:`INTERRUPTED` or :data:`ERROR` (None
    while it runs).
    """

    seen: dict[_Key, SubmissionCheck] = field(default_factory=dict)
    failed: set[_Key] = field(default_factory=set)
    checks: list[SubmissionCheck] = field(default_factory=list)
    refused: dict[Path, str] = field(default_factory=dict)
    polls: int = 0
    started: float | None = None
    stopped: str | None = None

    def record(self, result: CheckResult, *, ended: bool) -> None:
        """Record the checks of one pass; *ended* counts the pass as a poll."""
        for check in result.submissions:
            key = (check.database, check.submission_id)
            self.seen[key] = check
            if check.outcome == FAILED:
                self.failed.add(key)
            self.checks.append(check)
        self.refused.update(result.refused)
        if ended:
            self.polls += 1

    def elapsed(self) -> float:
        """Return the seconds since the wait started; 0 before it started."""
        return 0.0 if self.started is None else _clock() - self.started


_STORED_OUTCOMES: Final = {
    "submitting": PENDING,
    "submitted": PENDING,
    UNRESOLVED: UNRESOLVED,
    "failed": FAILED,
    "finalized": FINALIZED_OUTCOME,
    ABANDONED: ABANDONED,
    "cancelled": SKIPPED,
}


def final_totals(ledger: WaitLedger) -> CheckStats:
    """Count every submission a pass saw by its current state in the store.

    A submission a later pass no longer listed is counted too. One that a
    pass reported as failed stays failed unless the store shows it
    finalized. When its database cannot be read, the latest check counts.
    """
    stats = CheckStats()
    readers: dict[Path, Store | None] = {}
    for key, check in ledger.seen.items():
        outcome = _stored_outcome(key, readers) or check.outcome
        if key in ledger.failed and outcome != FINALIZED_OUTCOME:
            outcome = FAILED
        stats.count(outcome)
    return stats


def _stored_outcome(key: _Key, readers: dict[Path, Store | None]) -> str | None:
    path, submission_id = key
    if path not in readers:
        try:
            readers[path] = read_store(path.parent)
        except StoreError as exc:
            logger.warning("Cannot read %s for the totals: %s", path, exc)
            readers[path] = None
    reader = readers[path]
    if reader is None:
        return None
    try:
        status = reader.submission(submission_id).status
    except KeyError:
        return None
    return _STORED_OUTCOMES.get(status)


def _retryable(exc: BaseException) -> bool:
    """Whether a pass that raised *exc* is retried at the next poll."""
    if isinstance(exc, CheckTargetError | StoreError):
        return False
    return classify_error(exc).retryable


def _ignore(*_: object) -> None:
    return None


async def check_until_settled(  # noqa: PLR0913 - the check's options and hooks
    directory: Path,
    settings: Settings,
    limits: WaitLimits,
    ledger: WaitLedger,
    *,
    formats: Sequence[str] | None = None,
    batch_ids: Sequence[str] = (),
    adopt: str | None = None,
    abandon: str | None = None,
    schema: str | None = None,
    args: argparse.Namespace | None = None,
    on_pass: Callable[[int, CheckResult, bool], None] = _ignore,
    on_wait: Callable[[WaitTick], None] = _ignore,
    on_retry: Callable[[int, Exception, float | None], None] = _ignore,
) -> PollResult[CheckResult]:
    """Repeat :func:`check_submissions` until no submission is pending.

    Every pass shares one set of batch clients, closed when the wait ends;
    *adopt* and *abandon* apply to the first pass only. *ledger*, owned by
    the caller, records each pass as it ends, also a pass that raises, so
    the caller keeps the totals after an exception. *on_pass* receives the
    number and result of each pass and whether to show its pending
    submissions: in the first pass and in the pass the deadline ends.
    *on_wait* receives the state before each sleep; *on_retry* the number of
    a pass that failed with a retryable error, the error and the seconds
    until the next pass (None when the deadline has passed).

    Raises:
        CheckTargetError: *adopt* or *abandon* fits no submission, or more
            than one.
    """
    clients = BatchClients(settings, ROLE, args=args)
    ledger.started = _clock()

    async def run_pass(number: int) -> CheckResult:
        result = CheckResult()
        first = number == 1
        try:
            await check_submissions(
                directory,
                settings,
                formats=formats,
                batch_ids=batch_ids,
                adopt=adopt if first else None,
                abandon=abandon if first else None,
                schema=schema,
                args=args,
                clients=clients,
                result=result,
            )
        except Exception:
            ledger.record(result, ended=True)
            on_pass(number, result, first)
            raise
        except BaseException:
            ledger.record(result, ended=False)
            raise
        ledger.record(result, ended=True)
        ending = pending_work(result) > 0 and ledger.elapsed() >= limits.timeout
        on_pass(number, result, first or ending)
        return result

    def retried(number: int, exc: Exception) -> None:
        remaining = limits.timeout - ledger.elapsed()
        next_in = min(limits.interval, remaining) if remaining > 0 else None
        on_retry(number, exc, next_in)

    try:
        polled = await poll_until_settled(
            run_pass,
            pending_work,
            limits,
            on_wait=on_wait,
            on_retry=retried,
            retryable=_retryable,
            clock=_clock,
            sleep=_sleep,
        )
    finally:
        clients.close()
    ledger.stopped = polled.stop.value
    return polled
