"""Shared plumbing of the subcommands: summary line, errors, logging, warnings.

Exit codes: 0 when every item completed, 1 when an item failed or another
run leases one, 2 for usage and configuration errors, 130 on Ctrl+C or
Ctrl+Break. With ``--json`` exactly one JSON summary line goes to stdout on
every exit; everything else goes to stderr.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TextIO

from chronominer import ext
from chronominer.cli.console import print_error, print_warning
from chronominer.cli.parser import UsageError
from chronominer.common.contract import (
    USAGE_KEY,
    ExitCode,
    SummaryEmitter,
    usage_field,
)
from chronominer.common.log_setup import configure_logging
from chronominer.common.options import OptionTable
from chronominer.common.store import StoreError
from chronominer.common.usage import Usage, UsageTotals
from chronominer.llm.caller import collect_usage
from chronominer.settings import Settings, load
from chronominer.spec import CommandSpec, condition_warnings

if TYPE_CHECKING:
    from chronominer.run import PlannedItem, RunPlan

__all__ = [
    "LOGGER_NAME",
    "Summary",
    "cancel_fields",
    "check_fields",
    "condition_warnings",
    "drop_fields",
    "guarded",
    "load_settings",
    "plan_fields",
    "render_fields",
    "repair_fields",
    "report_usage_error",
    "setup_logging",
    "summary_fields",
    "summary_for",
    "warn_conditions",
]

LOGGER_NAME = "chronominer"

logger = logging.getLogger(__name__)


def summary_fields(  # noqa: PLR0913 - one keyword per summary field
    *,
    total: int = 0,
    complete: int = 0,
    failed: int = 0,
    skipped: int = 0,
    blocked: int = 0,
    submitted: int = 0,
    outputs: Sequence[str] = (),
    usage: Mapping[str, Usage] | None = None,
) -> dict[str, Any]:
    """Return the fields of the JSON run summary.

    *blocked* counts the items another run leases or a batch holds; they
    are among the *skipped* items too, and a leased one exits 1.
    *submitted* counts the items whose chunks wait in batches (``--batch``);
    they count neither complete nor failed. *usage* holds the run's tokens
    per role; the summary adds their total.
    """
    return {
        "items_total": total,
        "items_complete": complete,
        "items_failed": failed,
        "items_skipped": skipped,
        "items_blocked": blocked,
        "items_submitted": submitted,
        "outputs": list(outputs),
        USAGE_KEY: usage_field(usage),
    }


def plan_fields(run_plan: RunPlan) -> dict[str, Any]:
    """Return the fields a dry run adds to the run summary.

    ``to_process`` holds the items to extract with their input type and
    resume state (none or partial); ``skipped`` the complete items;
    ``unstored``, ``blocked`` and ``refused`` the items left out, each with
    the reason. ``spec`` holds the effective value and source of every run
    option.
    """
    return {
        "to_process": [
            {"name": item.name, "type": item.type, "state": item.state}
            for item in run_plan.to_process
        ],
        "skipped": [item.name for item in run_plan.skipped],
        "unstored": _reasons(run_plan.unstored),
        "blocked": _reasons(run_plan.blocked),
        "refused": _reasons(run_plan.refused),
        "spec": run_plan.resolved_spec(),
    }


def _reasons(items: Sequence[PlannedItem]) -> list[dict[str, str]]:
    return [{"name": item.name, "reason": item.reason} for item in items]


def repair_fields(
    *,
    repaired: int = 0,
    failed: int = 0,
    usage: Mapping[str, Usage] | None = None,
) -> dict[str, Any]:
    """Return the fields of the repair summary; *usage* as in a run summary."""
    return {
        "command": "repair",
        "repaired": repaired,
        "failed": failed,
        USAGE_KEY: usage_field(usage),
    }


def check_fields(  # noqa: PLR0913 - one keyword per summary field
    *,
    finalized: int = 0,
    pending: int = 0,
    failed: int = 0,
    unresolved: int = 0,
    submitted: int = 0,
    complete: int = 0,
    outputs: Sequence[str] = (),
    usage: Mapping[str, Usage] | None = None,
) -> dict[str, Any]:
    """Return the fields of the batch check summary.

    The first four count submissions by outcome; *submitted* counts the
    submissions the check opened (none: ChronoMiner's checks only collect)
    and *complete* the documents it left with no pending or failed chunk.
    *outputs* are the files it rendered and *usage* the tokens of the
    finalized results per role.
    """
    return {
        "command": "check_batches",
        "finalized": finalized,
        "pending": pending,
        "failed": failed,
        "unresolved": unresolved,
        "submitted": submitted,
        "complete": complete,
        "outputs": list(outputs),
        USAGE_KEY: usage_field(usage),
    }


def cancel_fields(*, cancelled: int = 0, failed: int = 0) -> dict[str, Any]:
    """Return the fields of the batch cancel summary."""
    return {"command": "cancel_batches", "cancelled": cancelled, "failed": failed}


def render_fields(
    *,
    databases: int = 0,
    documents: int = 0,
    written: int = 0,
    unchanged: int = 0,
    kept: int = 0,
    stale: int = 0,
    skipped: int = 0,
    outputs: Sequence[str] = (),
) -> dict[str, Any]:
    """Return the fields of the render summary.

    ``kept`` counts files changed outside the tool and left alone, ``stale``
    files that could not be replaced and ``skipped`` documents another run
    holds; ``outputs`` lists the files written.
    """
    return {
        "command": "render",
        "databases": databases,
        "documents": documents,
        "written": written,
        "unchanged": unchanged,
        "kept": kept,
        "stale": stale,
        "skipped": skipped,
        "outputs": list(outputs),
    }


def drop_fields(
    *,
    dropped: Sequence[str] = (),
    refused: Sequence[Mapping[str, Any]] = (),
    deleted_files: Sequence[str] = (),
    kept_files: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return the fields of the drop summary.

    ``dropped`` names the documents deleted (in a dry run, those a drop
    would delete), ``refused`` each document left in place with its
    ``doc_id``, ``name`` and ``reason``, ``deleted_files`` the rendered files
    removed and ``kept_files`` each file left in place with its ``path`` and
    ``reason``.
    """
    return {
        "command": "drop",
        "dropped": list(dropped),
        "refused": [dict(item) for item in refused],
        "deleted_files": list(deleted_files),
        "kept_files": [dict(item) for item in kept_files],
    }


class Summary:
    """The JSON summary line of one command, written at most once.

    *base* holds the command's fields with zero counts; :meth:`emit` writes
    them updated by its arguments after the ``dry_run`` flag. Without a
    stream nothing is written.
    """

    def __init__(
        self, stream: TextIO | None, base: Mapping[str, Any], *, dry_run: bool = False
    ) -> None:
        self._emitter = SummaryEmitter(stream, dry_run=dry_run)
        self._base = dict(base)

    @property
    def reports_usage(self) -> bool:
        """Whether the summary carries the run's token usage."""
        return USAGE_KEY in self._base

    def emit(self, exit_code: int, **fields: Any) -> int:
        """Write the line once and return *exit_code*."""
        self._emitter.emit({**self._base, **fields})
        return int(exit_code)


def summary_for(argv: Sequence[str]) -> Summary:
    """Return the summary of the command *argv* names (extract by default).

    ``ranges generate`` and ``ranges adjust`` share the extract summary;
    ``ui`` has no JSON summary, so its summary writes nothing.
    """
    words = list(argv)
    if words[:1] == ["ui"]:
        return Summary(None, {})
    stream = sys.stdout if "--json" in argv else None
    dry_run = "--dry-run" in argv
    if words[:1] == ["repair"]:
        return Summary(stream, repair_fields(), dry_run=dry_run)
    if words[:2] == ["batch", "check"]:
        return Summary(stream, check_fields())
    if words[:2] == ["batch", "cancel"]:
        return Summary(stream, cancel_fields())
    if words[:1] == ["render"]:
        return Summary(stream, render_fields(), dry_run=dry_run)
    if words[:1] == ["drop"]:
        return Summary(stream, drop_fields(), dry_run=dry_run)
    return Summary(stream, summary_fields(), dry_run=dry_run)


def report_usage_error(error: UsageError, argv: Sequence[str]) -> int:
    """Write a parse error with the usage text, plus the JSON line if asked."""
    sys.stderr.write(error.usage)
    print_error(f"Invalid arguments: {error}")
    return summary_for(argv).emit(ExitCode.USAGE)


def setup_logging(settings: Settings | None = None) -> None:
    """Send the tool's log records to the settings' log file.

    The console handler only shows critical records; the tool reports to
    the console through its own messages. Without *settings* no file is
    written.
    """
    configure_logging(
        logger_name=LOGGER_NAME,
        level=logging.CRITICAL,
        log_file=settings.log_file() if settings is not None else None,
    )


def load_settings(flags: Mapping[str, Any]) -> Settings:
    """Load the settings file named by ``--settings``, else the default one."""
    return load(flags.get("settings"), blocks=ext.settings_blocks())


def guarded(summary: Summary, body: Callable[[], int]) -> int:
    """Run *body*; Ctrl+C returns 130 and a crash returns 1, each with the line.

    A database the store refuses (versionless, newer or unreadable) returns
    2. On these exits a run summary holds the tokens spent until then.
    """
    with collect_usage() as spent:
        try:
            return body()
        except StoreError as exc:
            print_error(f"Invalid arguments: {exc}")
            return summary.emit(exc.exit_code, **_spent(summary, spent))
        except KeyboardInterrupt:
            print_error("Interrupted by the user.")
            return summary.emit(ExitCode.INTERRUPTED, **_spent(summary, spent))
        except Exception as exc:
            logger.exception("Command failed: %s", exc)
            print_error(f"Unexpected error: {exc}")
            return summary.emit(ExitCode.FAILURE, **_spent(summary, spent))


def _spent(summary: Summary, totals: UsageTotals) -> dict[str, Any]:
    if not summary.reports_usage:
        return {}
    return {USAGE_KEY: usage_field({role: totals.get(role) for role in totals.roles})}


def warn_conditions(
    table: OptionTable,
    flags: Mapping[str, Any],
    settings: Settings,
    spec: CommandSpec,
) -> None:
    """Write the condition warnings of *spec* to stderr."""
    for message in condition_warnings(table, flags, settings.defaults, spec):
        print_warning(message)
