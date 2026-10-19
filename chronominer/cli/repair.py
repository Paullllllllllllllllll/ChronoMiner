"""The repair subcommand: extract the failed chunks of earlier runs again.

``repair`` exits 0 on a clean repair; 1 when a targeted chunk stays
unrepaired or the repair work fails; 2 when the repair cannot be planned
(missing input, an input no store records, a refused database, an
unresolvable schema or context, no key or batch API, a model without image
input) or the extension refuses it. A dry run plans without the key
variable and lists it as a warning. The extension starts on every dry run
and on every run with chunks to repair.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from chronominer import ext
from chronominer.batch.repair import (
    RepairPlanError,
    RepairResult,
    plan_repair,
    run_repair,
)
from chronominer.cli.command import (
    Summary,
    guarded,
    load_settings,
    repair_fields,
    setup_logging,
    warn_conditions,
)
from chronominer.cli.console import (
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from chronominer.cli.reporter import StderrReporter
from chronominer.common.contract import ExitCode, exit_code_for
from chronominer.llm.caller import collect_usage
from chronominer.settings import SettingsError, SpecError, resolve_repair
from chronominer.spec import REPAIR_OPTIONS, RepairSpec

__all__ = ["execute_repair"]


def _report_repair(spec: RepairSpec, result: RepairResult) -> None:
    """Write the repair's outcome."""
    name = spec.input.name or str(spec.input)
    if result.message is not None:
        print_info(result.message)
        return
    if result.dry_run:
        print_info(f"[DRY RUN] {result.selected} chunk(s) would be repaired.")
        return
    print_header("REPAIR COMPLETE", "")
    if result.failed:
        print_warning(
            f"Repair of '{name}' ({result.mode}) left {result.failed} of "
            f"{result.selected} chunk(s) unrepaired."
        )
    else:
        print_success(
            f"Repaired {result.repaired} chunk(s) of '{name}' ({result.mode})."
        )
    if result.database is not None:
        print_info(f"Stored in: {result.database}")
    for output in result.outputs:
        print_info(f"Rendered: {output}")


def _execute_repair(args: argparse.Namespace, summary: Summary) -> int:
    flags = REPAIR_OPTIONS.values(args)
    setup_logging()
    try:
        settings = load_settings(flags)
        spec = resolve_repair(settings, flags).spec
    except (SettingsError, SpecError) as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    if not spec.run.dry_run:
        setup_logging(settings)
    warn_conditions(REPAIR_OPTIONS, flags, settings, spec)
    print_header("REPAIR EXTRACTIONS", "")
    try:
        repair_plan = plan_repair(spec, settings, args=args)
    except RepairPlanError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(exc.exit_code)
    for warning in repair_plan.warnings:
        print_warning(warning)
    if repair_plan.dry_run or repair_plan.to_repair:
        try:
            ext.startup(spec, settings=settings, args=args)
        except SettingsError as exc:
            print_error(f"Invalid arguments: {exc}")
            return summary.emit(ExitCode.USAGE)
    with collect_usage() as spent:
        if repair_plan.dry_run or not repair_plan.to_repair:
            result = repair_plan.outcome()
        else:
            result = asyncio.run(run_repair(repair_plan, StderrReporter()))
    _report_repair(spec, result)
    return summary.emit(
        exit_code_for(failed=result.failed),
        **repair_fields(
            repaired=result.repaired,
            failed=result.failed,
            usage={role: spent.get(role) for role in spent.roles},
        ),
    )


def execute_repair(args: argparse.Namespace) -> int:
    """Run the parsed repair options and return the exit code."""
    flags = REPAIR_OPTIONS.values(args)
    summary = Summary(
        sys.stdout if flags.get("json") else None,
        repair_fields(),
        dry_run=bool(flags.get("dry_run")),
    )
    return guarded(summary, lambda: _execute_repair(args, summary))
