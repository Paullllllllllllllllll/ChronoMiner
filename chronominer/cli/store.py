"""The render and drop subcommands: the commands that need no model calls.

``render`` exits 0 when every file holds its rendering or was left alone
because it was edited, 1 when a file could not be replaced or a document is
held by another run, 2 for a missing input or a refused database. It writes
one JSON summary line with ``--json`` and nothing at all with ``--dry-run``.

``drop`` exits 0 when every selected document was dropped (in a dry run,
would be), 1 when one was refused because a live run leases it or an open
or unresolved batch holds its units, and 2 for a missing store, a
malformed selection or one that matches nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from chronominer.cli.command import (
    Summary,
    drop_fields,
    guarded,
    render_fields,
    setup_logging,
)
from chronominer.cli.console import (
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from chronominer.common.contract import ExitCode
from chronominer.common.store import (
    DocumentSelectionError,
    DropReport,
    VersionlessDatabaseError,
    select_documents,
)
from chronominer.output.drop import (
    KEPT_NOTE,
    DropInputError,
    drop_documents,
    drop_root,
    stored_documents,
)
from chronominer.output.render import (
    DocumentRender,
    find_databases,
    preview_database,
    render_database,
)
from chronominer.output.store import DATABASE_NAME, read_store
from chronominer.spec import (
    DROP_OPTIONS,
    RENDER_OPTIONS,
    DropSpec,
    RenderSpec,
    SpecError,
    resolve_drop,
    resolve_render,
)

__all__ = ["execute_drop", "execute_render"]


def _report_render(results: list[DocumentRender], dry_run: bool) -> None:
    prefix = "[DRY RUN] " if dry_run else ""
    verb = "would write" if dry_run else "wrote"
    for result in results:
        label = f"{result.database.parent.name}/{result.name}"
        if result.skipped is not None:
            print_error(f"{label}: {result.skipped}")
        for path in result.written:
            print_info(f"{prefix}{label}: {verb} {path}")
        for path in result.kept:
            print_warning(
                f"{label}: {path} was changed outside the tool and is left as it "
                "is; render with --force to replace it."
            )
        for warning in result.warnings:
            if "changed outside the tool" not in warning:
                print_warning(f"{label}: {warning}")
    written = sum(len(result.written) for result in results)
    unchanged = sum(len(result.unchanged) for result in results)
    kept = sum(len(result.kept) for result in results)
    print_info(
        f"{prefix}Render of {len(results)} document(s): {verb} {written} "
        f"file(s), {unchanged} unchanged, {kept} left as edited."
    )


def _execute_render(spec: RenderSpec, summary: Summary) -> int:
    setup_logging()
    path = spec.input.expanduser().resolve()
    if not path.exists():
        print_error(f"Invalid arguments: Input path does not exist: {path}")
        return summary.emit(ExitCode.USAGE)
    if path.is_file() and path.name != DATABASE_NAME:
        print_error(f"Invalid arguments: {path} is not a {DATABASE_NAME} file")
        return summary.emit(ExitCode.USAGE)
    databases = find_databases(path)
    for database in databases:
        try:
            read_store(database.parent)
        except VersionlessDatabaseError as exc:
            print_error(f"Invalid arguments: {exc}")
            return summary.emit(ExitCode.USAGE)
    print_header("RENDER", str(path))
    results: list[DocumentRender] = []
    for database in databases:
        if spec.dry_run:
            results.extend(
                preview_database(database, formats=spec.formats, force=spec.force)
            )
        else:
            results.extend(
                asyncio.run(
                    render_database(database, formats=spec.formats, force=spec.force)
                )
            )
    _report_render(results, spec.dry_run)
    failed = any(result.failed for result in results)
    return summary.emit(
        ExitCode.FAILURE if failed else ExitCode.OK,
        **render_fields(
            databases=len(databases),
            documents=len(results),
            written=sum(len(result.written) for result in results),
            unchanged=sum(len(result.unchanged) for result in results),
            kept=sum(len(result.kept) for result in results),
            stale=sum(len(result.stale) for result in results),
            skipped=sum(result.skipped is not None for result in results),
            outputs=[str(path) for result in results for path in result.written],
        ),
    )


def execute_render(args: argparse.Namespace) -> int:
    """Run the parsed render options and return the exit code."""
    flags = RENDER_OPTIONS.values(args)
    summary = Summary(
        sys.stdout if flags.get("json") else None,
        render_fields(),
        dry_run=bool(flags.get("dry_run")),
    )
    try:
        spec = resolve_render(flags).spec
    except SpecError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    return guarded(summary, lambda: _execute_render(spec, summary))


def _report_drop(report: DropReport) -> None:
    prefix = "[DRY RUN] " if report.dry_run else ""
    dropped, deleted = (
        ("would drop", "would delete") if report.dry_run else ("dropped", "deleted")
    )
    for name in report.dropped:
        print_info(f"{prefix}{dropped} {name}")
    for refusal in report.refused:
        print_error(f"refused {refusal.name or refusal.doc_id}: {refusal.reason}")
    for path in report.deleted_files:
        print_info(f"{prefix}{deleted} {path}")
    for kept in report.kept_files:
        print_warning(f"kept {kept.path}: {kept.reason}")
    if report.kept_files:
        print_warning(KEPT_NOTE)
    totals = (
        f"{prefix}Drop: {dropped} {len(report.dropped)} document(s), "
        f"{len(report.refused)} refused; {deleted} {len(report.deleted_files)} "
        f"file(s), {len(report.kept_files)} kept."
    )
    if report.refused:
        print_error(totals)
    elif report.dry_run:
        print_info(totals)
    else:
        print_success(totals)


def _execute_drop(spec: DropSpec, summary: Summary) -> int:
    setup_logging()
    try:
        root = drop_root(spec.input)
        documents = stored_documents(root)
        selection = select_documents(
            documents, select=spec.select, files=spec.files, select_all=spec.all
        )
    except (DropInputError, DocumentSelectionError) as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    if not selection.documents:
        print_error(f"Invalid arguments: {root / DATABASE_NAME} holds no document")
        return summary.emit(ExitCode.USAGE)
    print_header("DROP", str(root / DATABASE_NAME))
    for warning in selection.warnings:
        print_warning(warning)
    report = asyncio.run(
        drop_documents(root, selection.documents, dry_run=spec.dry_run)
    )
    _report_drop(report)
    fields = report.as_dict()
    del fields["dry_run"]
    return summary.emit(
        ExitCode.FAILURE if report.refused else ExitCode.OK, **drop_fields(**fields)
    )


def execute_drop(args: argparse.Namespace) -> int:
    """Run the parsed drop options and return the exit code."""
    flags = DROP_OPTIONS.values(args)
    summary = Summary(
        sys.stdout if flags.get("json") else None,
        drop_fields(),
        dry_run=bool(flags.get("dry_run")),
    )
    try:
        spec = resolve_drop(flags).spec
    except SpecError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    return guarded(summary, lambda: _execute_drop(spec, summary))
