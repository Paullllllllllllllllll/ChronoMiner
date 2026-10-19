"""Run screen: progress events, summary table and plan tables."""

from __future__ import annotations

import dataclasses
import io
import re
import sys
from pathlib import Path

from rich.console import Console, RenderableType

import chronominer.run as core
from chronominer.common.store import UnitKey, UnitRow
from chronominer.common.usage import Usage
from chronominer.discovery import SourceItem
from chronominer.events import (
    ChunkDone,
    ItemFinished,
    ItemStarted,
    RunEvents,
    Waiting,
)
from chronominer.output.store import Classification, ItemState
from chronominer.spec import (
    ResolvedSchema,
    resolve_ranges_adjust,
    resolve_ranges_generate,
)
from chronominer.ui.progress import (
    RichProgress,
    plan_table,
    ranges_plan_table,
    summary_table,
)
from tests.conftest import make_spec

ESC = "\x1b"


def _console() -> Console:
    return Console(
        file=io.StringIO(), width=100, color_system=None, force_terminal=False
    )


def _output(console: Console) -> str:
    file = console.file
    assert isinstance(file, io.StringIO)
    return file.getvalue()


def _render(renderable: RenderableType) -> str:
    console = _console()
    console.print(renderable)
    return _output(console)


def _drive(events: RunEvents) -> None:
    events.item_started(ItemStarted("alpha.pdf", "pdf", 1, 3))
    events.chunk_done(ChunkDone("alpha.pdf", 2, "ok", 1, 2, already_complete=2))
    events.waiting(Waiting("rate limit", 30.0))
    events.chunk_done(ChunkDone("alpha.pdf", 3, "ok", 2, 2, already_complete=2))
    events.item_finished(
        ItemFinished(
            "alpha.pdf", 1, 3, True, ("/out/alpha_output.json", "/out/alpha.csv")
        )
    )
    events.item_started(ItemStarted("ledger.txt", "text", 2, 3))
    events.warning("model refused chunk 1\nretrying later")
    events.chunk_done(ChunkDone("ledger.txt", 0, "failed", 1, 2))
    events.chunk_done(ChunkDone("ledger.txt", 1, "ok", 2, 2))
    events.item_finished(
        ItemFinished("ledger.txt", 2, 3, False, error="1 chunk failed")
    )
    events.item_started(ItemStarted("photo.jpg", "image", 3, 3))
    events.chunk_done(ChunkDone("photo.jpg", 0, "deferred", 1, 1))
    events.item_finished(ItemFinished("photo.jpg", 3, 3, True))


def test_non_terminal_console_prints_plain_lines() -> None:
    console = _console()
    with RichProgress(console) as progress:
        assert not progress.live
        _drive(progress)
    output = _output(console)
    assert ESC not in output
    lines = output.splitlines()
    assert "[1/3] alpha.pdf (PDF)" in lines
    assert "[2/3] ledger.txt (text)" in lines
    assert "[3/3] photo.jpg (image)" in lines
    assert "Waiting 30s: rate limit" in lines
    assert "Warning: model refused chunk 1" in lines
    assert "Warning: retrying later" in lines
    assert (
        "[1/3] alpha.pdf: complete; 2 chunk(s) ok, 0 failed, 0 deferred, "
        "2 from an earlier run"
    ) in lines
    assert "    outputs: /out/alpha_output.json, /out/alpha.csv" in lines
    assert "[2/3] ledger.txt: failed; 1 chunk(s) ok, 1 failed, 0 deferred" in lines
    assert "    error: 1 chunk failed" in lines
    assert "[3/3] photo.jpg: complete; 0 chunk(s) ok, 0 failed, 1 deferred" in lines


def test_item_without_chunks_prints_its_verdict() -> None:
    console = _console()
    with RichProgress(console) as progress:
        progress.item_started(ItemStarted("gamma.txt", "text", 1, 1))
        progress.item_finished(ItemFinished("gamma.txt", 1, 1, False))
    lines = _output(console).splitlines()
    assert "[1/1] gamma.txt (text)" in lines
    assert "[1/1] gamma.txt: failed" in lines


def test_partial_item_prints_partial() -> None:
    console = _console()
    with RichProgress(console) as progress:
        progress.item_started(ItemStarted("gamma.txt", "text", 1, 1))
        progress.item_finished(ItemFinished("gamma.txt", 1, 1, True, partial=True))
    assert "[1/1] gamma.txt: partial" in _output(console).splitlines()


def test_waiting_with_cancel_shows_the_wait_only() -> None:
    cancelled: list[bool] = []
    console = _console()
    with RichProgress(console) as progress:
        progress.waiting(Waiting("batch pending", 60.0, lambda: cancelled.append(True)))
    assert "Waiting 60s: batch pending" in _output(console).splitlines()
    assert not cancelled


def test_terminal_console_shows_bars_and_redirects_stderr() -> None:
    console = Console(
        file=io.StringIO(), width=100, force_terminal=True, color_system=None
    )
    with RichProgress(console) as progress:
        assert progress.live
        progress.item_started(ItemStarted("ledger.txt", "text", 1, 1))
        progress.chunk_done(ChunkDone("ledger.txt", 0, "ok", 1, 3, already_complete=2))
        progress.waiting(Waiting("rate limit", 12.0))
        screen = _render(progress._render())
        assert "Waiting 12s: rate limit" in screen
        assert "chunks, +2 from an earlier run" in screen
        sys.stderr.write("a log record\n")
        progress.item_finished(ItemFinished("ledger.txt", 1, 1, True))
    assert not progress.live
    output = _output(console)
    assert "a log record" in output
    assert "ledger.txt: complete; 1 chunk(s) ok" in output
    assert "Items" in output


def _result() -> core.RunResult:
    return core.RunResult(
        items=(
            core.ItemResult("alpha.pdf", True, ("/out/alpha_output.json",)),
            core.ItemResult("ledger.txt", False, error="1 chunk failed"),
            core.ItemResult("gamma.txt", True, ("/out/gamma_output.json",)),
        ),
        skipped=("delta.pdf",),
        seconds=12.34,
        usage={
            "extract": Usage(1200, 300, 1500, 100, 0),
            "adjust": Usage(800, 200, 1000, 0, 50),
        },
        databases=("/out/chronominer.sqlite",),
        blocked=("epsilon.pdf",),
        unstored=("old.txt",),
    )


def _row(output: str, first: str) -> list[str]:
    for line in output.splitlines():
        cells = [cell.strip() for cell in re.split(r"[│┃]", line.strip("│┃ "))]
        if cells and cells[0] == first:
            return cells
    raise AssertionError(f"no row starting with {first!r} in:\n{output}")


def test_summary_table_lists_items_chunks_outputs_and_usage() -> None:
    progress = RichProgress(_console())
    with progress:
        _drive(progress)
    output = _render(summary_table(_result(), progress))
    assert ESC not in output
    assert _row(output, "alpha.pdf") == ["alpha.pdf", "complete", "2", "0", "0"]
    assert _row(output, "ledger.txt") == [
        "ledger.txt",
        "failed",
        "1",
        "1",
        "0",
        "1 chunk failed",
    ]
    assert _row(output, "gamma.txt") == ["gamma.txt", "complete", "-", "-", "-"]
    assert "2 complete, 1 failed" in output
    assert "Outputs: /out/alpha_output.json, /out/gamma_output.json" in output
    assert "Skipped (already complete): delta.pdf" in output
    assert "Blocked (held by another run): epsilon.pdf" in output
    assert "Unstored (outputs without store rows): old.txt" in output
    assert "Databases written: /out/chronominer.sqlite" in output
    assert "Run time: 12.3s" in output
    assert _row(output, "extract") == [
        "extract",
        "1,200",
        "300",
        "100",
        "0",
        "1,500",
    ]
    assert _row(output, "adjust")[-1] == "1,000"
    totals = [
        line for line in output.splitlines() if line.strip("│ ").startswith("Total")
    ]
    assert len(totals) == 2
    assert _row(totals[0], "Total")[2:5] == ["3", "1", "0"]
    assert "2,000" in totals[1] and "2,500" in totals[1]


def test_summary_table_without_progress_usage_or_skips() -> None:
    result = core.RunResult(items=(core.ItemResult("alpha.pdf", True),))
    output = _render(summary_table(result))
    assert _row(output, "alpha.pdf")[2:5] == ["-", "-", "-"]
    assert "Token usage: none reported" in output
    assert "Skipped (already complete): none" in output
    assert "Blocked (held by another run): none" in output
    assert "Outputs: none" in output


def test_summary_table_shows_names_and_errors_literally() -> None:
    result = core.RunResult(
        items=(core.ItemResult("[red]draft.txt", False, error="bad [/x] tag"),)
    )
    output = _render(summary_table(result))
    assert "[red]draft.txt" in output
    assert "bad [/x] tag" in output


def _failed_chunk(idx: int, error_type: str | None, error: str | None) -> UnitRow:
    return UnitRow(
        doc_id=1,
        kind="chunk",
        idx=idx,
        status="failed",
        outcome="failed",
        source_index=idx,
        label=None,
        content=None,
        tool="chronominer",
        model="m",
        provider="google",
        run_id="r1",
        attempts=1,
        usage=None,
        error=error,
        error_type=error_type,
        provenance=None,
        response_id=None,
        stale=0,
        updated_at=None,
    )


def test_summary_names_each_failed_chunk_with_its_error_type_unclipped() -> None:
    long_name = "glasse_puddings_2p.txt-909ebc59/glasse_puddings_2p_extraction"
    filtered = "Response stopped by the provider's content filter: finish_reason=X"
    failures = (
        _failed_chunk(1, "content_filter", filtered),
        _failed_chunk(3, None, None),
    )
    result = core.RunResult(
        items=(
            core.ItemResult(
                "ledger.txt",
                False,
                error=f"1 of 4 chunk(s) failed for '{long_name}'",
                failures=failures,
            ),
        )
    )
    output = _render(summary_table(result))
    flat = " ".join(output.split())
    assert "…" not in output
    assert f"ledger.txt: chunk 2: content_filter: {filtered}" in flat
    assert "ledger.txt: chunk 4: error" in flat
    error_cells = [
        cells[-2].strip()
        for line in output.splitlines()
        if len(cells := line.split("│")) > 2
    ]
    assert long_name in "".join(error_cells)


def test_a_batch_run_shows_its_submitted_items_as_submitted() -> None:
    console = _console()
    with RichProgress(console) as progress:
        progress.item_started(ItemStarted("alpha.pdf", "pdf", 1, 3))
        progress.item_finished(ItemFinished("alpha.pdf", 1, 3, True, submitted=True))
        progress.item_started(ItemStarted("beta.pdf", "pdf", 2, 3))
        progress.item_finished(ItemFinished("beta.pdf", 2, 3, False))
        progress.item_started(ItemStarted("gamma.pdf", "pdf", 3, 3))
        progress.item_finished(ItemFinished("gamma.pdf", 3, 3, True))
    lines = _output(console).splitlines()
    assert "[1/3] alpha.pdf: submitted" in lines
    assert "[2/3] beta.pdf: failed" in lines
    assert "[3/3] gamma.pdf: complete" in lines
    result = core.RunResult(
        items=(
            core.ItemResult("alpha.pdf", True, submitted=True),
            core.ItemResult("beta.pdf", False, error="refused"),
        )
    )
    output = _render(summary_table(result, progress, batch=True))
    assert _row(output, "alpha.pdf")[:2] == ["alpha.pdf", "submitted"]
    assert _row(output, "beta.pdf")[:2] == ["beta.pdf", "failed"]
    assert "0 complete, 1 submitted, 1 failed" in output


def _planned(
    tmp_path: Path, name: str, kind: str, resume: Classification
) -> core.PlannedItem:
    return core.PlannedItem(
        SourceItem(tmp_path / name, kind),
        resume,
        tmp_path,
        name,
        tmp_path / f"{Path(name).stem}_output.json",
    )


def _plan(tmp_path: Path, *, batch: bool = False) -> core.RunPlan:
    spec = make_spec(tmp_path)
    spec = dataclasses.replace(spec, run=dataclasses.replace(spec.run, batch=batch))
    partial = Classification(
        ItemState.PARTIAL,
        "alpha.pdf",
        units=(UnitKey(1, "chunk", 3), UnitKey(1, "chunk", 4)),
        note="m2",
    )
    to_process = (
        _planned(tmp_path, "alpha.pdf", "pdf", partial),
        _planned(
            tmp_path, "ledger.txt", "text", Classification(ItemState.NONE, "ledger.txt")
        ),
    )
    return core.RunPlan(
        spec=spec,
        input_path=tmp_path,
        output_dir=None,
        discovered=tuple(item.source for item in to_process),
        to_process=to_process,
        skipped=(
            _planned(
                tmp_path,
                "done.pdf",
                "pdf",
                Classification(ItemState.COMPLETE, "done.pdf"),
            ),
            _planned(
                tmp_path,
                "stale.txt",
                "text",
                Classification(ItemState.COMPLETE, "stale.txt", render=True),
            ),
        ),
        schema=ResolvedSchema("WageRecords", tmp_path / "wages.json", {}),
        unstored=(
            _planned(
                tmp_path,
                "old.txt",
                "text",
                Classification(ItemState.UNSTORED, "old.txt", reason="no store rows"),
            ),
        ),
        blocked=(
            _planned(
                tmp_path,
                "held.jpg",
                "image",
                Classification(
                    ItemState.BLOCKED, "held.jpg", reason="leased by run r1"
                ),
            ),
        ),
        refused=(
            _planned(
                tmp_path,
                "other.pdf",
                "pdf",
                Classification(ItemState.REFUSED, "other.pdf", reason="dpi 300"),
            ),
        ),
    )


def test_plan_table_shows_resume_state_and_left_out_items(tmp_path: Path) -> None:
    run_plan = _plan(tmp_path)
    output = _render(plan_table(run_plan))
    assert ESC not in output
    assert "Plan: 2 to process, 2 already complete, 3 left out" in output
    assert _row(output, "1") == [
        "1",
        "alpha.pdf",
        "PDF",
        "partial",
        "2 chunk(s) to process; m2",
    ]
    assert _row(output, "2") == ["2", "ledger.txt", "text", "none"]
    lines = {
        cells[1]: cells[2:]
        for line in output.splitlines()
        if (cells := [c.strip() for c in re.split(r"[│┃]", line.strip("│┃ "))])
        and cells[0] == "-"
    }
    assert lines["done.pdf"] == ["PDF", "complete (skip)"]
    assert lines["stale.txt"] == ["text", "complete (render again)"]
    assert lines["old.txt"] == ["text", "unstored", "no store rows"]
    assert lines["held.jpg"] == ["image", "blocked", "leased by run r1"]
    assert lines["other.pdf"] == ["PDF", "refused", "dpi 300"]
    mode = run_plan.spec.chunking.mode
    assert f"Schema: WageRecords; text chunking: {mode}" in output
    assert "Batch:" not in output


def test_plan_table_names_the_batch(tmp_path: Path) -> None:
    run_plan = _plan(tmp_path, batch=True)
    output = " ".join(_render(plan_table(run_plan)).split())
    provider = run_plan.spec.model.provider
    assert (
        f"Batch: 2 item(s) will be submitted to the {provider} batch API; "
        "collect them with 'chronominer batch check'."
    ) in output


def test_plan_table_shows_names_and_reasons_literally(tmp_path: Path) -> None:
    run_plan = dataclasses.replace(
        _plan(tmp_path),
        to_process=(
            _planned(
                tmp_path,
                "[bold]a.pdf",
                "pdf",
                Classification(ItemState.PARTIAL, "[bold]a.pdf", note="[/i] m2"),
            ),
        ),
        refused=(
            _planned(
                tmp_path,
                "[red]b.pdf",
                "pdf",
                Classification(ItemState.REFUSED, "[red]b.pdf", reason="dpi [/x]"),
            ),
        ),
    )
    output = _render(plan_table(run_plan))
    assert "[bold]a.pdf" in output
    assert "[/i] m2" in output
    assert "[red]b.pdf" in output
    assert "dpi [/x]" in output


def test_ranges_plan_table_shows_file_names_literally(tmp_path: Path) -> None:
    text = tmp_path / "[red]ledger.txt"
    spec = resolve_ranges_generate({"input": tmp_path}).spec
    output = _render(ranges_plan_table(core.RangesPlan(spec, tmp_path, (text,))))
    assert "[red]ledger.txt" in output
    assert "[red]ledger_line_ranges.txt (write)" in output


def test_progress_descriptions_show_bracketed_names_literally() -> None:
    console = Console(
        file=io.StringIO(), width=100, force_terminal=True, color_system=None
    )
    with RichProgress(console) as progress:
        progress.item_started(ItemStarted("[red]scan.pdf", "pdf", 1, 1))
        assert "[red]scan.pdf" in _render(progress._render())


def test_ranges_plan_table_for_generate(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    (tmp_path / "a_line_ranges.txt").write_text("1-10\n", encoding="utf-8")
    spec = resolve_ranges_generate({"input": tmp_path, "chunk_tokens": 7500}).spec
    ranges_plan = core.RangesPlan(spec, tmp_path, (first, second))
    output = _render(ranges_plan_table(ranges_plan))
    assert ESC not in output
    assert "Plan (ranges generate): 2 file(s)" in output
    assert _row(output, "1") == ["1", "a.txt", "a_line_ranges.txt (replace)"]
    assert _row(output, "2") == ["2", "b.txt", "b_line_ranges.txt (write)"]
    assert f"Chunk size: 7,500 tokens ({spec.model} tokenizer)" in output


def test_ranges_plan_table_for_adjust(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    spec = resolve_ranges_adjust(
        {"input": tmp_path, "schema": "WageRecords", "pages": "2-3"}
    ).spec
    ranges_plan = core.RangesPlan(
        spec,
        tmp_path,
        (first, second),
        warnings=("Skipping 'c.txt': c_line_ranges.txt is missing",),
        schema=ResolvedSchema("WageRecords", tmp_path / "wages.json", {}),
        contexts={first: "Entries are dated.", second: None},
        response_format="json",
    )
    output = " ".join(_render(ranges_plan_table(ranges_plan)).split())
    raw = _render(ranges_plan_table(ranges_plan))
    assert "Plan (ranges adjust): 2 file(s)" in output
    assert _row(raw, "1") == ["1", "a.txt", "a_line_ranges.txt", "yes"]
    assert _row(raw, "2") == ["2", "b.txt", "b_line_ranges.txt", "none"]
    assert "Adjust schema: WageRecords" in output
    assert (
        f"Model: {spec.model.provider}:{spec.model.model}; response format: json; "
        f"context window: {spec.context_window} line(s)"
    ) in output
    assert "Line ranges selected: 2-3" in output
    assert "Warning: Skipping 'c.txt': c_line_ranges.txt is missing" in output
