"""The classify subcommands: decision-model answers about the store, shadow only.

``classify chunks``, ``classify entries`` and ``classify documents`` resolve
their spec, plan (classifier, documents, states) and run like ``extract``,
and write the classify summary. Exit codes: 0 when every document was
classified or nothing was left to ask (a dry run always), 1 when a document
failed, was refused because its input changed, or another run leases it, 2
for usage and configuration errors (an unknown classifier or one of another
unit, ``--chunking adjust``, a missing line-ranges file, an unset key
variable, no documents), 130 on Ctrl+C or Ctrl+Break.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Mapping
from typing import Any

from chronominer import ext
from chronominer.classify.plan import (
    BLOCKED,
    READY,
    REFUSED,
    SKIPPED,
    ClassifyPlan,
    plan_classify,
)
from chronominer.classify.run import (
    CLASSIFIED,
    FAILED,
    ClassifyResult,
    run_classify,
)
from chronominer.cli.command import (
    Summary,
    classify_fields,
    guarded,
    load_settings,
    setup_logging,
    warn_conditions,
)
from chronominer.cli.console import (
    print_error,
    print_info,
    print_success,
    print_warning,
)
from chronominer.cli.reporter import StderrReporter
from chronominer.common.contract import ExitCode
from chronominer.common.options import OptionTable
from chronominer.run import PlanError
from chronominer.settings import Settings, SettingsError, SpecError
from chronominer.spec import (
    CLASSIFY_CHUNKS_OPTIONS,
    CLASSIFY_DOCUMENTS_OPTIONS,
    CLASSIFY_ENTRIES_OPTIONS,
    ClassifySpec,
    Resolution,
    resolve_classify_entries,
    resolve_classify_items,
)

__all__ = ["execute_chunks", "execute_documents", "execute_entries"]

_TABLES: Mapping[str, OptionTable] = {
    "chunk": CLASSIFY_CHUNKS_OPTIONS,
    "entry": CLASSIFY_ENTRIES_OPTIONS,
    "document": CLASSIFY_DOCUMENTS_OPTIONS,
}


def _resolve(
    unit: str, settings: Settings, flags: Mapping[str, Any]
) -> Resolution[Any]:
    if unit == "entry":
        return resolve_classify_entries(flags, settings.defaults)
    return resolve_classify_items(unit, flags, settings.defaults)


def _plan_fields(plan: ClassifyPlan, result: ClassifyResult) -> dict[str, Any]:
    """Return the fields a dry run adds: each document's outcome and the spec."""
    return {
        "targets": [
            {
                "name": target.name,
                "document": target.document,
                "state": target.state,
                "reason": target.reason,
                "units_asked": target.asked,
                "units_reused": target.reused,
            }
            for target in result.targets
        ],
        "spec": plan.effective_spec(),
    }


def _report(plan: ClassifyPlan, result: ClassifyResult) -> None:
    prefix = "[DRY RUN] " if plan.dry_run else ""
    asked = "to ask" if plan.dry_run else "asked"
    reused = "to reuse" if plan.dry_run else "reused"
    for target in result.targets:
        if target.state == CLASSIFIED:
            print_info(
                f"{prefix}{target.name}: {target.asked} {plan.unit}(s) {asked}, "
                f"{target.reused} {reused}"
            )
        elif target.state == FAILED:
            print_error(f"{target.name}: {target.reason}")
    counts = (
        f"{prefix}{plan.command} with {plan.classifier.name}: "
        f"{result.count(CLASSIFIED)} of {len(result.targets)} document(s) "
        f"classified; {result.asked} {plan.unit}(s) {asked}, {result.reused} "
        f"{reused}."
    )
    if result.exit_code == ExitCode.OK:
        print_success(counts)
    else:
        print_warning(counts)
    for database in result.databases:
        print_info(f"Database: {database}")
    tokens = result.total_usage
    if tokens.total_tokens:
        print_info(
            f"Tokens: {tokens.input_tokens:,} input, {tokens.output_tokens:,} "
            f"output, {tokens.total_tokens:,} total."
        )


def _run(
    args: argparse.Namespace,
    settings: Settings,
    spec: ClassifySpec,
    plan: ClassifyPlan,
    summary: Summary,
) -> int:
    base = classify_fields(
        unit=plan.unit, classifier=plan.classifier.name, total=len(plan.targets)
    )
    try:
        ext.startup(spec, settings=settings, args=args)
    except SettingsError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE, **base)
    for message in plan.warnings:
        print_warning(message)
    ready = len(plan.having(READY))
    prefix = "[DRY RUN] " if plan.dry_run else ""
    print_info(
        f"{prefix}{plan.command} with classifier {plan.classifier.name} "
        f"({plan.classifier_path}): {ready} of {len(plan.targets)} document(s) "
        "to classify."
    )
    try:
        result = asyncio.run(run_classify(plan, StderrReporter()))
    except PlanError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(exc.exit_code, **base)
    _report(plan, result)
    fields = classify_fields(
        unit=plan.unit,
        classifier=plan.classifier.name,
        total=len(plan.targets),
        classified=result.count(CLASSIFIED),
        refused=result.count(REFUSED),
        blocked=result.count(BLOCKED),
        failed=result.count(FAILED),
        skipped=result.count(SKIPPED),
        asked=result.asked,
        reused=result.reused,
        outputs=result.databases,
        usage=result.usage,
    )
    if plan.dry_run:
        return summary.emit(ExitCode.OK, **fields, **_plan_fields(plan, result))
    return summary.emit(result.exit_code, **fields)


def _execute(args: argparse.Namespace, summary: Summary, unit: str) -> int:
    table = _TABLES[unit]
    flags = table.values(args)
    setup_logging()
    try:
        settings = load_settings(flags)
        resolution = _resolve(unit, settings, flags)
    except (SettingsError, SpecError) as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    spec: ClassifySpec = resolution.spec
    if not spec.run.dry_run:
        setup_logging(settings)
    warn_conditions(table, flags, settings, spec)
    try:
        plan = plan_classify(spec, settings, sources=resolution.sources, args=args)
    except PlanError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(
            exc.exit_code, **classify_fields(unit=unit, total=exc.items_total)
        )
    return _run(args, settings, spec, plan, summary)


def _execute_command(args: argparse.Namespace, unit: str) -> int:
    flags = _TABLES[unit].values(args)
    summary = Summary(
        sys.stdout if flags.get("json") else None,
        classify_fields(unit=unit),
        dry_run=bool(flags.get("dry_run")),
    )
    return guarded(summary, lambda: _execute(args, summary, unit))


def execute_chunks(args: argparse.Namespace) -> int:
    """Run the parsed classify chunks options and return the exit code."""
    return _execute_command(args, "chunk")


def execute_entries(args: argparse.Namespace) -> int:
    """Run the parsed classify entries options and return the exit code."""
    return _execute_command(args, "entry")


def execute_documents(args: argparse.Namespace) -> int:
    """Run the parsed classify documents options and return the exit code."""
    return _execute_command(args, "document")
