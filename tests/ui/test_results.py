"""Result screens of batch check and cancel, repair and render."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console, RenderableType

from chronominer.batch.cancel import CancelOutcome, CancelReport, Listing
from chronominer.batch.check import CheckResult, SubmissionCheck
from chronominer.batch.repair import RepairResult
from chronominer.common.batch import BatchListing, BatchStatus
from chronominer.output.render import DocumentRender
from chronominer.spec import RepairSpec, resolve_repair
from chronominer.ui.results import (
    cancel_result,
    check_result,
    render_result,
    repair_result,
)

ESC = "\x1b"
DB = Path("/runs/book/chronominer.sqlite")


def _render(renderable: RenderableType) -> str:
    file = io.StringIO()
    console = Console(file=file, force_terminal=False, width=100, color_system=None)
    console.print(renderable)
    output = file.getvalue()
    assert ESC not in output
    return output


def _flat(output: str) -> str:
    """Return *output* with its whitespace runs collapsed, for wrapped lines."""
    return " ".join(output.split())


def test_check_result_lists_submissions_candidates_refusals_and_totals() -> None:
    result = CheckResult()
    result.add(
        SubmissionCheck(
            DB,
            "s1",
            "openai",
            "finalized",
            batch_id="b1",
            errors=2,
            outputs=(Path("/runs/book/a_output.json"),),
            warnings=("chunk 3 truncated",),
        )
    )
    result.add(SubmissionCheck(DB, "s2", "anthropic", "pending", "b2", "running"))
    result.add(
        SubmissionCheck(
            DB,
            "s3",
            "google",
            "unresolved",
            detail="2 candidates",
            candidates=(
                BatchListing(
                    "google",
                    "b9",
                    BatchStatus.COMPLETED,
                    created_at=datetime(2026, 10, 1, tzinfo=UTC),
                    request_count=4,
                ),
            ),
        )
    )
    result.refused[Path("/runs/old/chronominer.sqlite")] = "versionless"
    output = _flat(_render(check_result(result)))
    assert "book/chronominer.sqlite" in output
    for cell in ("s1", "openai", "b1", "finalized", "s2", "pending", "running"):
        assert cell in output
    assert "chronominer.sqlite s1: outputs a_output.json" in output
    assert (
        "2 chunk(s) failed in the batch; repair them with 'chronominer repair'."
    ) in output
    assert "warning: chunk 3 truncated" in output
    assert (
        "s3: candidate b9: created 2026-10-01T00:00:00+00:00, 4 request(s), completed"
    ) in output
    assert "--adopt BATCH_ID" in output
    assert "versionless" in output
    assert (
        "Batch check: 1 finalized, 1 pending, 0 failed, 1 unresolved, 0 abandoned."
    ) in output


def test_result_screens_show_bracketed_paths_and_ids_literally(tmp_path: Path) -> None:
    db = Path(r"C:\data\[draft]\chronominer.sqlite")
    result = CheckResult()
    result.add(
        SubmissionCheck(db, "[s1]", "openai", "failed", "[b1]", detail="bad [/x] tag")
    )
    check = _flat(_render(check_result(result)))
    assert "[draft]/chronominer.sqlite" in check
    assert "[s1]" in check and "[b1]" in check and "bad [/x] tag" in check
    outcome = CancelOutcome("[b2]", "openai", True, submission_id="[s2]")
    cancelled = _render(cancel_result(CancelReport(outcomes=[outcome])))
    assert "[b2]" in cancelled and "[s2]" in cancelled
    rendered = _render(render_result([DocumentRender(db, "[a]")], dry_run=False))
    assert "[draft]/[a]" in rendered
    spec = _repair_spec(tmp_path, input=tmp_path / "[draft] a.txt")
    repair = RepairResult(repaired=1, selected=1, mode="sync")
    assert "Repair of [draft] a.txt" in _render(repair_result(spec, repair))


def test_cancel_result_lists_listing_outcomes_and_totals() -> None:
    report = CancelReport(
        outcomes=[
            CancelOutcome("b1", "openai", True, submission_id="s1"),
            CancelOutcome("b2", "anthropic", False, error="gone"),
        ],
        listing=Listing(skipped=["google"], errors={"anthropic": "timeout"}),
        refused={Path("/runs/old/chronominer.sqlite"): "newer"},
        skipped=["s4"],
    )
    output = _flat(_render(cancel_result(report)))
    assert "google: key variable not set; not listed." in output
    assert "anthropic: listing failed: timeout" in output
    assert "Found 0 unfinished batch(es) to cancel." in output
    assert "newer" in output
    assert "Not cancelled, no provider batch: s4" in output
    assert "cancelled" in output and "failed: gone" in output
    assert "Cancelled: 1; failed: 2." in output


def _repair_spec(tmp_path: Path, **flags: object) -> RepairSpec:
    return resolve_repair({"input": tmp_path / "book.txt", **flags}).spec


def test_repair_result_reports_counts_database_and_outputs(tmp_path: Path) -> None:
    spec = _repair_spec(tmp_path)
    result = RepairResult(
        repaired=3,
        failed=1,
        selected=4,
        mode="batch",
        database=tmp_path / "chronominer.sqlite",
        outputs=(tmp_path / "book_output.json",),
    )
    output = _flat(_render(repair_result(spec, result)))
    assert "Targeting failed chunks" in output
    assert "Repair of book.txt" in output
    assert "batch 4 3 1" in _flat(output.replace("│", " "))
    assert "Stored in:" in output and "Rendered:" in output


def test_repair_result_dry_run_and_message(tmp_path: Path) -> None:
    spec = _repair_spec(tmp_path)
    dry = RepairResult(selected=5, dry_run=True)
    output = _flat(_render(repair_result(spec, dry)))
    assert "[DRY RUN] 5 chunk(s) would be repaired." in output
    message = RepairResult(message="Nothing to repair.")
    assert _flat(_render(repair_result(spec, message))) == "Nothing to repair."


def test_repair_result_names_the_schema_and_indices(tmp_path: Path) -> None:
    spec = _repair_spec(tmp_path, schema="WageRecords", indices="2,5")
    dry = RepairResult(selected=2, dry_run=True)
    output = _flat(_render(repair_result(spec, dry)))
    assert "Targeting failed chunks of schema WageRecords at indices 2, 5" in output


def test_render_result_lists_documents_files_and_totals() -> None:
    results = [
        DocumentRender(
            DB,
            "a",
            written=[Path("/runs/book/a_output.json")],
            unchanged=[Path("/runs/book/a_line_ranges.txt")],
            kept=[Path("/runs/book/a_output.csv")],
            warnings=["a_output.csv changed outside the tool", "slow disk"],
        ),
        DocumentRender(
            DB, "b", stale=[Path("/runs/book/b_output.json")], skipped="leased"
        ),
    ]
    output = _flat(_render(render_result(results, dry_run=False)))
    assert "book/a: wrote" in output
    assert "a_output.csv was changed outside the tool" in output
    assert "book/a: slow disk" in output
    assert output.count("changed outside the tool") == 1
    assert "leased" in output
    assert (
        "Render of 2 document(s): wrote 1 file(s), 1 unchanged, 1 left as edited."
    ) in output
    dry = _flat(_render(render_result(results[:1], dry_run=True)))
    assert "[DRY RUN] Render of 1 document(s): would write 1 file(s)" in dry
