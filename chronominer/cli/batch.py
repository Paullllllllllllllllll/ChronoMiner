"""The batch subcommands: check and cancel.

``batch check`` exits 1 when a submission failed or waits unresolved for
``--adopt`` or ``--abandon``, and 2 when the folder is missing, a database
below it is refused (versionless, newer or unreadable) or ``--adopt`` or
``--abandon`` names nothing to act on. ``batch cancel`` exits 1 when a
cancellation or a provider listing failed, and 2 for a missing folder or a
refused database. The batch check JSON summary counts the submissions by
outcome (``unresolved``: waiting for ``--adopt`` or ``--abandon``), the
documents left ``complete``, the rendered ``outputs`` and the ``usage`` of
the finalized results, and adds ``adopted`` or ``abandoned`` when those
options are given.

``batch check --wait`` checks again every ``--poll-minutes`` until no
submission is pending. It counts every submission a check saw by its state
in the store at the end and adds ``polls``, ``waited_seconds`` and
``stopped`` (``settled``, ``timeout``, ``interrupted`` or ``error``) to the
summary. It also exits 1 when ``--wait-hours`` pass with work pending, or
when a check fails with an error that is not retried.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Coroutine, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from chronominer import ext
from chronominer.batch.cancel import (
    CancelReport,
    cancel_batches,
    cancel_submissions,
    cancel_unfinished,
)
from chronominer.batch.check import (
    ABANDONED,
    ERROR,
    INTERRUPTED,
    UNRESOLVED,
    CheckResult,
    CheckStats,
    CheckTargetError,
    SubmissionCheck,
    WaitLedger,
    check_outputs,
    check_until_settled,
    check_usage,
    failed_units,
    final_totals,
)
from chronominer.batch.check import check_submissions as check_batches
from chronominer.batch.finalize import (
    FAILED,
    FINALIZED_OUTCOME,
    PENDING,
    SKIPPED,
)
from chronominer.cli.command import (
    Summary,
    cancel_fields,
    check_fields,
    guarded,
    load_settings,
    setup_logging,
)
from chronominer.cli.console import (
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from chronominer.cli.heartbeat import BatchHeartbeat
from chronominer.common.contract import ExitCode, exit_code_for
from chronominer.common.polling import Stop, WaitLimits, WaitTick
from chronominer.common.retry import classify_error
from chronominer.common.store import StoreError
from chronominer.settings import (
    Settings,
    SettingsError,
    SpecError,
    resolve_batch_cancel,
    resolve_batch_check,
)
from chronominer.spec import (
    BATCH_CANCEL_OPTIONS,
    BATCH_CHECK_OPTIONS,
    BatchCheckSpec,
    CommandSpec,
)

__all__ = [
    "check_batches",
    "execute_cancel",
    "execute_check",
]

logger = logging.getLogger(__name__)


def _label(check: SubmissionCheck) -> str:
    batch = f" batch {check.batch_id}" if check.batch_id else ""
    return (
        f"{check.database.parent.name}/{check.database.name}: submission "
        f"{check.submission_id} ({check.provider}{batch})"
    )


def _report_unresolved(check: SubmissionCheck) -> None:
    print_warning(f"{_label(check)} is unresolved: {check.detail}")
    for listing in check.candidates:
        created = listing.created_at.isoformat() if listing.created_at else "unknown"
        count = "unknown" if listing.request_count is None else listing.request_count
        print_info(
            f"  candidate {listing.batch_id}: created {created}, {count} "
            f"request(s), {listing.provider_status or listing.status.value}"
        )


def _report_submission(check: SubmissionCheck) -> None:
    label = _label(check)
    if check.outcome == FINALIZED_OUTCOME:
        outputs = ", ".join(path.name for path in check.outputs) or "no file"
        print_success(f"Finalized {label} -> {outputs}")
        if check.detail:
            print_info(f"{label}: {check.detail}")
        for warning in check.warnings:
            print_warning(warning)
        if check.errors:
            print_warning(
                f"{check.errors} chunk(s) of {label} failed in the batch; repair "
                "them with 'chronominer repair'."
            )
    elif check.outcome == PENDING:
        print_info(f"{label}: not finished yet ({check.detail})")
    elif check.outcome == FAILED:
        print_error(f"{label}: {check.detail}")
    elif check.outcome == UNRESOLVED:
        _report_unresolved(check)
    elif check.outcome == ABANDONED:
        print_info(f"Abandoned {label}: {check.detail}")
    elif check.outcome == SKIPPED:
        print_info(f"{label}: {check.detail}")


def _report_totals(stats: CheckStats) -> None:
    print_info(
        f"Batch check: {stats.finalized} finalized, {stats.pending} pending, "
        f"{stats.failed} failed, {stats.unresolved} unresolved."
    )


def _report_check(
    result: CheckResult, *, pending: bool = True, totals: bool = True
) -> None:
    """Write one line per submission, the refused databases and the totals.

    Without *pending* the submissions that are not finished yet are left
    out; without *totals* the closing counts.
    """
    for path, reason in result.refused.items():
        print_error(f"{path}: {reason}")
    for check in result.submissions:
        if pending or check.outcome != PENDING:
            _report_submission(check)
    if totals:
        _report_totals(result.stats)


def _clock_time(seconds: float) -> str:
    """Return the local time *seconds* from now as HH:MM."""
    return (datetime.now() + timedelta(seconds=seconds)).strftime("%H:%M")


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f}"


def _report_pass(number: int, result: CheckResult, pending: bool) -> None:
    """Write a pass of a wait: the first in full, later ones without totals."""
    _report_check(result, pending=pending, totals=number == 1)


def _report_wait(tick: WaitTick) -> None:
    print_info(
        f"Waiting for batches: {tick.pending} pending; next check at "
        f"{_clock_time(tick.next_in)} (check {tick.passes + 1}; "
        f"{_hours(tick.elapsed)} of {tick.timeout / 3600:g} h)."
    )


def _report_retry(number: int, exc: Exception, next_in: float | None) -> None:
    then = (
        f"checking again at {_clock_time(next_in)}"
        if next_in is not None
        else "the deadline has passed"
    )
    print_warning(f"Check {number} failed ({classify_error(exc)}): {exc}; {then}.")


def _check_options(spec: BatchCheckSpec) -> dict[str, Any]:
    """Return the keyword options of the check beyond formats and batch IDs."""
    options: dict[str, Any] = {}
    if spec.adopt is not None:
        options["adopt"] = spec.adopt
    if spec.abandon is not None:
        options["abandon"] = spec.abandon
    if spec.schema is not None:
        options["schema"] = spec.schema
    return options


def _check_base(flags: Mapping[str, Any]) -> dict[str, Any]:
    """Return the check summary's fields with zero counts.

    ``adopted`` and ``abandoned`` appear when their option is given, and
    ``polls``, ``waited_seconds`` and ``stopped`` with ``--wait``.
    """
    base = check_fields()
    for option, key in (("adopt", "adopted"), ("abandon", "abandoned")):
        if flags.get(option):
            base[key] = False
    if flags.get("wait"):
        base.update(polls=0, waited_seconds=0, stopped=None)
    return base


def _result_fields(
    spec: BatchCheckSpec, stats: CheckStats, checks: Sequence[SubmissionCheck]
) -> dict[str, Any]:
    """Return the summary fields of a check that saw *checks*, *stats* counted."""
    fields = check_fields(
        finalized=stats.finalized,
        pending=stats.pending,
        failed=stats.failed,
        unresolved=stats.unresolved,
        complete=sum(check.complete for check in checks),
        outputs=[str(path) for path in check_outputs(checks)],
        usage=check_usage(checks),
    )
    if spec.adopt is not None:
        fields["adopted"] = any(
            check.batch_id == spec.adopt and check.outcome != FAILED for check in checks
        )
    if spec.abandon is not None:
        fields["abandoned"] = any(
            check.submission_id == spec.abandon and check.outcome == ABANDONED
            for check in checks
        )
    return fields


def _startup(
    spec: CommandSpec,
    settings: Settings,
    args: argparse.Namespace,
    summary: Summary,
    directory: Path | None,
) -> int | None:
    """Check the input folder and run the extension startup.

    Returns the usage exit code if either refuses.
    """
    if directory is not None and not directory.is_dir():
        print_error(f"Invalid arguments: Input folder does not exist: {directory}")
        return summary.emit(ExitCode.USAGE)
    try:
        ext.startup(spec, settings=settings, args=args)
    except SettingsError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    return None


def _execute_check(args: argparse.Namespace, summary: Summary) -> int:
    flags = BATCH_CHECK_OPTIONS.values(args)
    setup_logging()
    try:
        settings = load_settings(flags)
        spec = resolve_batch_check(settings, flags).spec
    except (SettingsError, SpecError) as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    setup_logging(settings)
    directory = spec.directory.expanduser().resolve()
    refused = _startup(spec, settings, args, summary, directory)
    if refused is not None:
        return refused
    print_info(f"Scanning '{directory}' for batch submissions...")
    if spec.wait:
        return _execute_wait(spec, settings, args, summary, directory)
    try:
        result = asyncio.run(
            check_batches(
                directory,
                settings,
                formats=spec.formats,
                batch_ids=spec.batch_ids,
                args=args,
                **_check_options(spec),
            )
        )
    except CheckTargetError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    _report_check(result)
    stats = result.stats
    if result.refused:
        code = ExitCode.USAGE
    else:
        code = ExitCode.FAILURE if result.had_failure else ExitCode.OK
    return summary.emit(code, **_result_fields(spec, stats, result.submissions))


def _wait_code(ledger: WaitLedger, stats: CheckStats) -> ExitCode:
    """Return the exit code of a wait that ended settled or by its deadline."""
    if ledger.refused:
        return ExitCode.USAGE
    if (
        ledger.stopped == Stop.TIMEOUT
        or stats.had_failure
        or failed_units(ledger.checks) > 0
    ):
        return ExitCode.FAILURE
    return ExitCode.OK


def _end_wait(
    summary: Summary, spec: BatchCheckSpec, ledger: WaitLedger, code: int | None
) -> int:
    """Write the wait's totals and end line and emit its summary.

    *code* None takes the exit code from the totals (:func:`_wait_code`).
    """
    stats = final_totals(ledger)
    waited = ledger.elapsed()
    _report_totals(stats)
    checks = "check" if ledger.polls == 1 else "checks"
    print_info(
        f"Stopped after {ledger.polls} {checks} and {_hours(waited)} h: "
        f"{ledger.stopped}."
    )
    return summary.emit(
        _wait_code(ledger, stats) if code is None else code,
        **_result_fields(spec, stats, ledger.checks),
        polls=ledger.polls,
        waited_seconds=int(waited),
        stopped=ledger.stopped,
    )


def _execute_wait(
    spec: BatchCheckSpec,
    settings: Settings,
    args: argparse.Namespace,
    summary: Summary,
    directory: Path,
) -> int:
    """Check every ``--poll-minutes`` until nothing is pending or the deadline.

    The totals count every submission a pass saw by its state in the store.
    Ctrl+C emits them and re-raises; an error a pass does not retry ends the
    wait with 1, a refused database with 2.
    """
    limits = WaitLimits(spec.poll_minutes * 60, spec.wait_hours * 3600)
    ledger = WaitLedger()
    beat = BatchHeartbeat.from_env()

    def on_pass(number: int, result: CheckResult, pending: bool) -> None:
        _report_pass(number, result, pending)
        if beat is not None:
            beat.on_pass(result)

    def on_wait(tick: WaitTick) -> None:
        _report_wait(tick)
        if beat is not None:
            beat.on_wait(tick)

    try:
        asyncio.run(
            check_until_settled(
                directory,
                settings,
                limits,
                ledger,
                formats=spec.formats,
                batch_ids=spec.batch_ids,
                args=args,
                on_pass=on_pass,
                on_wait=on_wait,
                on_retry=_report_retry,
                **_check_options(spec),
            )
        )
    except CheckTargetError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    except KeyboardInterrupt:
        ledger.stopped = INTERRUPTED
        _end_wait(summary, spec, ledger, ExitCode.INTERRUPTED)
        raise
    except StoreError as exc:
        print_error(f"Invalid arguments: {exc}")
        ledger.stopped = ERROR
        return _end_wait(summary, spec, ledger, exc.exit_code)
    except Exception as exc:
        logger.exception("Batch check failed: %s", exc)
        print_error(f"Check {ledger.polls} failed: {exc}")
        ledger.stopped = ERROR
        return _end_wait(summary, spec, ledger, ExitCode.FAILURE)
    return _end_wait(summary, spec, ledger, None)


def execute_check(args: argparse.Namespace) -> int:
    """Run the parsed batch check options and return the exit code."""
    flags = BATCH_CHECK_OPTIONS.values(args)
    summary = Summary(sys.stdout if flags.get("json") else None, _check_base(flags))
    return guarded(summary, lambda: _execute_check(args, summary))


def _report_cancel(report: CancelReport) -> None:
    listing = report.listing
    if listing is not None:
        for provider in listing.skipped:
            print_info(f"{provider}: key variable not set; not listed.")
        for provider, error in listing.errors.items():
            print_error(f"{provider}: listing failed: {error}")
        print_info(f"Found {len(listing.batches)} unfinished batch(es) to cancel.")
    for path, reason in report.refused.items():
        print_error(f"{path}: {reason}")
    for skipped in report.skipped:
        print_info(f"Not cancelled: {skipped}")
    for outcome in report.outcomes:
        if outcome.cancelled:
            print_success(f"Cancelled: {outcome.batch_id} ({outcome.provider})")
        else:
            print_error(f"Failed to cancel {outcome.batch_id}: {outcome.error}")
    print_header("CANCELLATION COMPLETE", "")
    print_success(f"Successfully cancelled: {report.cancelled}")
    if report.failed:
        print_error(f"Failed: {report.failed}")


def _cancel(
    directory: Path | None,
    batch_ids: tuple[str, ...],
    settings: Settings,
    args: argparse.Namespace,
) -> Coroutine[Any, Any, CancelReport]:
    if directory is not None:
        print_info(f"Cancelling the open batch submissions below '{directory}'...")
        return cancel_submissions(directory, settings, args=args)
    if batch_ids:
        print_info(f"Targeting {len(batch_ids)} specific batch(es)...")
        return cancel_batches(batch_ids, settings, args=args)
    print_info("Listing the unfinished batches of every provider...")
    return cancel_unfinished(settings, args=args)


def _execute_cancel(args: argparse.Namespace, summary: Summary) -> int:
    flags = BATCH_CANCEL_OPTIONS.values(args)
    setup_logging()
    try:
        settings = load_settings(flags)
        spec = resolve_batch_cancel(settings, flags).spec
    except (SettingsError, SpecError) as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    setup_logging(settings)
    directory = None
    if spec.directory is not None:
        directory = spec.directory.expanduser().resolve()
    refused = _startup(spec, settings, args, summary, directory)
    if refused is not None:
        return refused
    print_header("BATCH CANCELLATION", "")
    report = asyncio.run(_cancel(directory, spec.batch_ids, settings, args))
    _report_cancel(report)
    code = ExitCode.USAGE if report.refused else exit_code_for(failed=report.failed)
    return summary.emit(code, cancelled=report.cancelled, failed=report.failed)


def execute_cancel(args: argparse.Namespace) -> int:
    """Run the parsed batch cancel options and return the exit code."""
    flags = BATCH_CANCEL_OPTIONS.values(args)
    summary = Summary(sys.stdout if flags.get("json") else None, cancel_fields())
    return guarded(summary, lambda: _execute_cancel(args, summary))
