"""Human-readable run reporting on stderr; stdout carries only the JSON line."""

from __future__ import annotations

import math

from chronominer.cli.console import (
    print_error,
    print_info,
    print_success,
    print_warning,
)
from chronominer.events import ChunkDone, ItemFinished, ItemStarted, Waiting
from chronominer.run import RunPlan, RunResult

__all__ = ["PROGRESS_STEPS", "StderrReporter"]

# Progress lines per item: one each time another 1/PROGRESS_STEPS of the
# pending chunks is done; failed chunks are always reported.
PROGRESS_STEPS = 10


class StderrReporter:
    """Write run events, the plan and the run summary to stderr."""

    def item_started(self, event: ItemStarted) -> None:
        """Name the item about to be processed."""
        print_info(f"Processing item {event.index}/{event.total}: {event.name}")

    def chunk_done(self, event: ChunkDone) -> None:
        """Report failed chunks, and progress in steps."""
        if event.status != "ok":
            print_warning(
                f"{event.item}: chunk {event.chunk + 1} {event.status} "
                f"({event.done}/{event.total})"
            )
            return
        step = max(1, math.ceil(event.total / PROGRESS_STEPS))
        if event.done % step and event.done != event.total:
            return
        line = f"{event.item}: {event.done}/{event.total} chunk(s) done"
        if event.already_complete:
            line += f" ({event.already_complete} from an earlier run)"
        print_info(line)

    def waiting(self, event: Waiting) -> None:
        """Report a wait."""
        seconds = f" {event.seconds:.0f}s" if event.seconds is not None else ""
        print_info(f"Waiting{seconds}: {event.reason}")

    def warning(self, message: str) -> None:
        """Write a warning."""
        print_warning(message)

    def item_finished(self, event: ItemFinished) -> None:
        """Report a failed item, or the outputs of a finished one."""
        if not event.success:
            print_error(f"Failed to process '{event.name}': {event.error}")
            return
        for output in event.outputs:
            print_success(f"Saved output for '{event.name}' -> {output}")

    def plan(self, run_plan: RunPlan) -> None:
        """Write the plan's warnings, the items left out and the resume mode.

        Unstored and blocked items are listed as skipped, refused items with
        their reason.
        """
        for message in run_plan.warnings:
            print_warning(message)
        for item in (*run_plan.unstored, *run_plan.blocked):
            print_warning(f"Skipping '{item.name}' ({item.state}): {item.reason}")
        for item in run_plan.refused:
            print_warning(item.reason)
        if run_plan.force:
            print_info("Resume mode: overwrite (all files will be reprocessed)")
        else:
            print_info("Resume mode: skip (use --force to reprocess all)")

    def dry_run(self, run_plan: RunPlan) -> None:
        """Write what the run would process, and whether it would batch."""
        print_info(
            f"[DRY RUN] {len(run_plan.discovered)} discovered; "
            f"{len(run_plan.to_process)} would be processed, "
            f"{len(run_plan.skipped)} skipped (already complete)."
        )
        for item in run_plan.to_process:
            print_info(f"[DRY RUN]   would process: {item.name} ({item.type})")
        for item in run_plan.render_only:
            print_info(f"[DRY RUN]   would render again: {item.name}")
        left_out = (*run_plan.unstored, *run_plan.blocked, *run_plan.refused)
        for item in left_out:
            print_info(f"[DRY RUN]   would skip ({item.state}): {item.name}")
        if run_plan.batch:
            print_info(
                f"[DRY RUN] Batch mode: {len(run_plan.to_process)} item(s) would "
                f"be submitted to the {run_plan.spec.model.provider} batch API."
            )

    def beginning(self, run_plan: RunPlan) -> None:
        """Write how many items the run processes."""
        if run_plan.skipped:
            print_info(
                f"Resume: skipping {len(run_plan.skipped)} already-processed item(s)"
            )
        print_info(f"Beginning processing of {len(run_plan.to_process)} item(s)...")

    def completion(self, result: RunResult) -> None:
        """Write the run's outcome, its databases and its tokens.

        Each blocked item is named with its reason, a lease with its expiry.
        An item another run's lease holds makes the run incomplete; one
        waiting for a batch is left to ``batch check``. Items that
        ``--pages`` left partial are counted.
        """
        for name, reason in zip(result.blocked, result.blocked_reasons, strict=True):
            print_warning(f"Not processed (blocked): '{name}': {reason}.")
        if result.leased:
            failures = (
                f" and {result.failed} failure(s) of {len(result.items)} item(s)"
                if result.failed
                else ""
            )
            print_warning(
                f"Processing incomplete: {len(result.leased)} item(s) leased by "
                f"another run{failures}. Run again once the lease has expired."
            )
        elif result.failed:
            print_warning(
                f"Processing complete with {result.failed} failure(s) "
                f"of {len(result.items)} item(s)."
            )
        else:
            print_success("Processing complete!")
        if result.partial:
            print_info(f"{result.partial} item(s) left partial by --pages.")
        for database in result.databases:
            print_info(f"Database: {database}")
        tokens = result.total_usage
        if tokens.total_tokens:
            print_info(
                f"Tokens: {tokens.input_tokens:,} input, "
                f"{tokens.output_tokens:,} output, {tokens.total_tokens:,} total."
            )
