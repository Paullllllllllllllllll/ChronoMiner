"""The extract subcommand: resolve the spec, plan the run and run it."""

from __future__ import annotations

import argparse
import asyncio
import sys

from chronominer import ext
from chronominer.cli.command import (
    Summary,
    guarded,
    load_settings,
    plan_fields,
    setup_logging,
    summary_fields,
    warn_conditions,
)
from chronominer.cli.console import print_error
from chronominer.cli.heartbeat import with_heartbeat
from chronominer.cli.reporter import StderrReporter
from chronominer.common.contract import ExitCode
from chronominer.run import PlanError, RunPlan, plan, run
from chronominer.settings import Settings, SettingsError, SpecError, resolve
from chronominer.spec import OPTIONS

__all__ = ["execute"]


def _process(run_plan: RunPlan, reporter: StderrReporter, summary: Summary) -> int:
    """Run *run_plan* and write its summary.

    Skipped items are the complete, unstored and blocked ones and those of
    which ``--pages`` selects no chunk; an item another run's lease holds
    exits 1.
    """
    skipped = run_plan.not_processed
    reporter.beginning(run_plan)
    events, progress = with_heartbeat(reporter)
    try:
        result = asyncio.run(run(run_plan, events))
    except PlanError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(
            exc.exit_code,
            **summary_fields(
                total=exc.items_total,
                skipped=skipped,
                blocked=len(run_plan.blocked),
            ),
        )
    if progress is not None:
        progress.run_finished()
    reporter.completion(result)
    return summary.emit(
        result.exit_code,
        **summary_fields(
            total=len(result.items),
            complete=result.complete,
            failed=result.failed,
            skipped=len(result.skipped)
            + len(run_plan.unstored)
            + len(result.blocked)
            + len(result.unselected),
            blocked=len(result.blocked),
            submitted=result.submitted,
            outputs=result.outputs,
            usage=result.usage,
        ),
    )


def _run(
    args: argparse.Namespace, settings: Settings, run_plan: RunPlan, summary: Summary
) -> int:
    """Start the extension, then report or run the plan."""
    reporter = StderrReporter()
    counts = summary_fields(
        total=len(run_plan.to_process),
        skipped=run_plan.not_processed,
        blocked=len(run_plan.blocked),
    )
    if run_plan.dry_run or run_plan.to_process:
        try:
            ext.startup(run_plan.spec, settings=settings, args=args)
        except SettingsError as exc:
            print_error(f"Invalid arguments: {exc}")
            return summary.emit(ExitCode.USAGE, **counts)
    reporter.plan(run_plan)
    if run_plan.dry_run:
        reporter.dry_run(run_plan)
        return summary.emit(ExitCode.OK, **counts, **plan_fields(run_plan))
    return _process(run_plan, reporter, summary)


def _execute(args: argparse.Namespace, summary: Summary) -> int:
    flags = OPTIONS.values(args)
    setup_logging()
    try:
        settings = load_settings(flags)
        resolution = resolve(settings, flags)
    except (SettingsError, SpecError) as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    spec = resolution.spec
    if not spec.run.dry_run:
        setup_logging(settings)
    warn_conditions(OPTIONS, flags, settings, spec)
    try:
        run_plan = plan(spec, settings, sources=resolution.sources, args=args)
    except PlanError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(exc.exit_code, **summary_fields(total=exc.items_total))
    return _run(args, settings, run_plan, summary)


def execute(args: argparse.Namespace) -> int:
    """Run the parsed extract options and return the exit code."""
    flags = OPTIONS.values(args)
    summary = Summary(
        sys.stdout if flags.get("json") else None,
        summary_fields(),
        dry_run=bool(flags.get("dry_run")),
    )
    return guarded(summary, lambda: _execute(args, summary))
