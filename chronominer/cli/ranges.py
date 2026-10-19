"""The ranges subcommands: generate and adjust the line ranges of text inputs.

Both resolve their spec, plan the run and run it like ``extract``, and write
the extract summary: one item per text file, its outputs the written
``_line_ranges.txt`` files and the databases. ``ranges adjust`` exits 2
before any write when the schema, the key variable, the endpoint or a
context file is unusable, or no text file has a line-ranges file; its dry
run lists an unset key variable as a warning instead.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Mapping
from typing import Any

from chronominer import ext
from chronominer.cli.command import (
    Summary,
    guarded,
    load_settings,
    setup_logging,
    summary_fields,
    warn_conditions,
)
from chronominer.cli.console import print_error, print_info, print_warning
from chronominer.cli.reporter import StderrReporter
from chronominer.common.contract import ExitCode, exit_code_for
from chronominer.common.options import OptionTable
from chronominer.run import PlanError, RangesPlan, plan_ranges, run_ranges
from chronominer.settings import (
    Settings,
    SettingsError,
    SpecError,
    resolve_ranges_adjust,
    resolve_ranges_generate,
)
from chronominer.spec import RANGES_ADJUST_OPTIONS, RANGES_GENERATE_OPTIONS, Resolution

__all__ = ["execute_adjust", "execute_generate", "ranges_plan_fields"]

type Resolver = Callable[[Settings, Mapping[str, Any]], Resolution[Any]]


def ranges_plan_fields(ranges_plan: RangesPlan) -> dict[str, Any]:
    """Return the fields a dry run adds to the summary.

    ``to_process`` names the text files the run would write ranges for;
    ``spec`` holds the effective value and source of every option.
    """
    return {
        "to_process": [
            {"name": path.name, "type": "text"} for path in ranges_plan.files
        ],
        "spec": ranges_plan.effective_spec(),
    }


def _dry_run(ranges_plan: RangesPlan) -> None:
    for message in ranges_plan.warnings:
        print_warning(message)
    print_info(
        f"[DRY RUN] {ranges_plan.command}: {len(ranges_plan.files)} text "
        "file(s) would be processed."
    )
    for path in ranges_plan.files:
        print_info(f"[DRY RUN]   would process: {path.name}")


def _process(
    ranges_plan: RangesPlan, reporter: StderrReporter, summary: Summary
) -> int:
    """Run *ranges_plan* and write its summary."""
    print_info(
        f"Beginning {ranges_plan.command} of {len(ranges_plan.files)} file(s)..."
    )
    try:
        result = asyncio.run(run_ranges(ranges_plan, reporter))
    except PlanError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(exc.exit_code, **summary_fields(total=exc.items_total))
    reporter.completion(result)
    return summary.emit(
        exit_code_for(failed=result.failed),
        **summary_fields(
            total=len(result.items),
            complete=result.complete,
            failed=result.failed,
            outputs=result.outputs,
            usage=result.usage,
        ),
    )


def _execute(
    args: argparse.Namespace,
    summary: Summary,
    table: OptionTable,
    resolve: Resolver,
) -> int:
    flags = table.values(args)
    setup_logging()
    try:
        settings = load_settings(flags)
        resolution = resolve(settings, flags)
    except (SettingsError, SpecError) as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    spec = resolution.spec
    if not flags.get("dry_run"):
        setup_logging(settings)
    warn_conditions(table, flags, settings, spec)
    try:
        ranges_plan = plan_ranges(spec, settings, sources=resolution.sources, args=args)
    except PlanError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(exc.exit_code, **summary_fields(total=exc.items_total))
    counts = summary_fields(total=len(ranges_plan.files))
    try:
        ext.startup(spec, settings=settings, args=args)
    except SettingsError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE, **counts)
    if ranges_plan.dry_run:
        _dry_run(ranges_plan)
        return summary.emit(ExitCode.OK, **counts, **ranges_plan_fields(ranges_plan))
    return _process(ranges_plan, StderrReporter(), summary)


def _execute_command(
    args: argparse.Namespace, table: OptionTable, resolve: Resolver
) -> int:
    flags = table.values(args)
    summary = Summary(
        sys.stdout if flags.get("json") else None,
        summary_fields(),
        dry_run=bool(flags.get("dry_run")),
    )
    return guarded(summary, lambda: _execute(args, summary, table, resolve))


def execute_generate(args: argparse.Namespace) -> int:
    """Run the parsed ranges generate options and return the exit code."""
    return _execute_command(args, RANGES_GENERATE_OPTIONS, resolve_ranges_generate)


def execute_adjust(args: argparse.Namespace) -> int:
    """Run the parsed ranges adjust options and return the exit code."""
    return _execute_command(args, RANGES_ADJUST_OPTIONS, resolve_ranges_adjust)
