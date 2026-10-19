"""Run screen of the guided UI: live progress bars, plan and summary tables.

:class:`RichProgress` implements :class:`~chronominer.events.RunEvents`
on a rich console. On a terminal it shows an overall bar for the items, a
chunk bar for the current item and a status line for waits; warnings and one
result line per item are printed above the bars, and so are log records
written to stderr while the display runs. On any other console it prints
plain lines only. It tallies the chunks of each item from the
:class:`~chronominer.events.ChunkDone` events, since the core's item results
carry no chunk counts; :func:`summary_table` shows them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from chronominer.common.usage import Usage
from chronominer.events import ChunkDone, ItemFinished, ItemStarted, Waiting
from chronominer.ranges.lines import ranges_path
from chronominer.run import PlannedItem, RangesPlan, RunPlan, RunResult
from chronominer.spec import RangesAdjustSpec

__all__ = [
    "ChunkTally",
    "RichProgress",
    "plan_table",
    "ranges_plan_table",
    "summary_table",
]

_KIND_LABELS = {
    "text": "text",
    "image": "image",
    "pdf": "PDF",
}


def _kind_label(kind: str) -> str:
    return _KIND_LABELS.get(kind, kind)


def _wait_line(event: Waiting) -> str:
    seconds = f" {event.seconds:.0f}s" if event.seconds is not None else ""
    return f"Waiting{seconds}: {event.reason}"


@dataclass
class ChunkTally:
    """The chunks of one item finished in this run, by status.

    *earlier* counts the chunks an earlier run completed.
    """

    ok: int = 0
    failed: int = 0
    deferred: int = 0
    earlier: int = 0

    def count(self, event: ChunkDone) -> None:
        """Count one finished chunk."""
        self.earlier = event.already_complete
        if event.status == "ok":
            self.ok += 1
        elif event.status == "deferred":
            self.deferred += 1
        else:
            self.failed += 1

    def line(self) -> str:
        """Return the counts as one phrase."""
        text = f"{self.ok} chunk(s) ok, {self.failed} failed, {self.deferred} deferred"
        if self.earlier:
            text += f", {self.earlier} from an earlier run"
        return text


class RichProgress:
    """Show run events on a rich console; use as a context manager."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console if console is not None else Console(stderr=True)
        self._progress = Progress(
            TextColumn("{task.description}", markup=False),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[note]}", markup=False),
            TimeElapsedColumn(),
            console=self.console,
            auto_refresh=False,
        )
        self._items_task: TaskID | None = None
        self._chunk_task: TaskID | None = None
        self._status: str = ""
        self._live: Live | None = None
        self._tallies: dict[str, ChunkTally] = {}

    @property
    def live(self) -> bool:
        """Whether the live display is running."""
        return self._live is not None

    @property
    def tallies(self) -> Mapping[str, ChunkTally]:
        """The chunk counts of each item that reported chunks, by item name."""
        return self._tallies

    def __enter__(self) -> Self:
        if self.console.is_terminal and not self.console.is_dumb_terminal:
            self._live = Live(
                console=self.console,
                get_renderable=self._render,
                refresh_per_second=8,
                redirect_stdout=False,
                redirect_stderr=True,
                transient=False,
            )
            self._live.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._status = ""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _render(self) -> RenderableType:
        bars = self._progress.get_renderable()
        if not self._status:
            return bars
        return Group(bars, Text(self._status, style="yellow"))

    def _print(self, text: Text) -> None:
        self.console.print(text, soft_wrap=True)

    def item_started(self, event: ItemStarted) -> None:
        """Start the chunk bar of the item and size the overall bar."""
        self._status = ""
        if self._items_task is None:
            self._items_task = self._progress.add_task(
                "Items", total=event.total, note=""
            )
        else:
            self._progress.update(self._items_task, total=event.total)
        if self._chunk_task is not None:
            self._progress.remove_task(self._chunk_task)
        self._chunk_task = self._progress.add_task(event.name, total=None, note="")
        if not self.live:
            self._print(
                Text(
                    f"[{event.index}/{event.total}] {event.name} "
                    f"({_kind_label(event.kind)})"
                )
            )

    def chunk_done(self, event: ChunkDone) -> None:
        """Count the chunk and advance the chunk bar to *done* of *total*."""
        self._status = ""
        self._tallies.setdefault(event.item, ChunkTally()).count(event)
        if self._chunk_task is None:
            return
        note = "chunks"
        if event.already_complete:
            note += f", +{event.already_complete} from an earlier run"
        self._progress.update(
            self._chunk_task, completed=event.done, total=event.total, note=note
        )

    def waiting(self, event: Waiting) -> None:
        """Show the wait on the status line, or print it."""
        line = _wait_line(event)
        if self.live:
            self._status = line
        else:
            self._print(Text(line))

    def warning(self, message: str) -> None:
        """Print a warning above the bars."""
        for line in message.splitlines() or [""]:
            self._print(Text(f"Warning: {line}", style="yellow"))

    def item_finished(self, event: ItemFinished) -> None:
        """Print the item's result line and advance the overall bar."""
        self._status = ""
        if self._chunk_task is not None:
            self._progress.remove_task(self._chunk_task)
            self._chunk_task = None
        if self._items_task is not None:
            self._progress.update(self._items_task, completed=event.index)
        if not event.success:
            verdict = "failed"
        elif event.submitted:
            verdict = "submitted"
        elif event.partial:
            verdict = "partial"
        else:
            verdict = "complete"
        line = Text(f"[{event.index}/{event.total}] {event.name}: ")
        line.append(verdict, style="green" if event.success else "red")
        tally = self._tallies.get(event.name)
        if tally is not None:
            line.append(f"; {tally.line()}")
        self._print(line)
        if event.error:
            self._print(Text(f"    error: {event.error}", style="red"))
        if event.outputs:
            self._print(Text(f"    outputs: {', '.join(event.outputs)}"))


def _usage_table(usage: Mapping[str, Usage], total: Usage) -> Table:
    table = Table(title="Token usage", title_justify="left")
    table.add_column("Role")
    for header in ("Input", "Output", "Cached", "Reasoning", "Total"):
        table.add_column(header, justify="right")

    def cells(value: Usage) -> list[str]:
        return [
            f"{value.input_tokens:,}",
            f"{value.output_tokens:,}",
            f"{value.cached_tokens:,}",
            f"{value.reasoning_tokens:,}",
            f"{value.total_tokens:,}",
        ]

    for role, value in usage.items():
        table.add_row(role, *cells(value))
    table.add_section()
    table.add_row("Total", *cells(total), style="bold")
    return table


def _listing(label: str, values: Iterable[str]) -> Text:
    names = list(values)
    return Text(f"{label}: {', '.join(names) if names else 'none'}")


def _failure_lines(result: RunResult) -> list[Text]:
    """Return one line per failed unit: item, unit and number, error type, error."""
    lines: list[Text] = []
    for item in result.items:
        for unit in item.failures:
            text = f"  {item.name}: {unit.kind} {unit.idx + 1}: "
            text += unit.error_type or "error"
            if unit.error:
                text += f": {' '.join(unit.error.split())}"
            lines.append(Text(text, style="red"))
    return [Text("Failed units:"), *lines] if lines else []


def summary_table(
    result: RunResult, progress: RichProgress | None = None, *, batch: bool = False
) -> RenderableType:
    """Return the run summary: items, outputs, run facts and token usage.

    The chunk counts come from *progress*; without it they show as ``-``.
    With *batch* the items whose chunks went to a batch show as submitted.
    """
    tallies = progress.tallies if progress is not None else {}
    table = Table(title="Run summary", title_justify="left")
    table.add_column("Item")
    table.add_column("Status")
    for header in ("Chunks ok", "Failed", "Deferred"):
        table.add_column(header, justify="right")
    table.add_column("Error", overflow="fold")
    totals = [0, 0, 0]
    for item in result.items:
        tally = tallies.get(item.name)
        if tally is None:
            counts: list[str] = ["-", "-", "-"]
        else:
            numbers = [tally.ok, tally.failed, tally.deferred]
            totals = [a + b for a, b in zip(totals, numbers, strict=True)]
            counts = [str(number) for number in numbers]
        if item.submitted:
            status = "submitted"
        elif item.partial:
            status = "partial"
        else:
            status = "complete"
        table.add_row(
            Text(item.name),
            Text(
                status if item.success else "failed",
                style="green" if item.success else "red",
            ),
            *counts,
            Text(item.error or ""),
        )
    table.add_section()
    submitted = f"{result.submitted} submitted, " if batch else ""
    if result.partial:
        submitted += f"{result.partial} partial, "
    table.add_row(
        "Total",
        f"{result.complete} complete, {submitted}{result.failed} failed",
        *(str(count) if tallies else "-" for count in totals),
        "",
        style="bold",
    )
    parts: list[RenderableType] = [table, *_failure_lines(result)]
    outputs = [path for item in result.items for path in item.outputs]
    parts.append(_listing("Outputs", outputs))
    parts += [
        _listing("Skipped (already complete)", result.skipped),
        _listing("Blocked (held by another run)", result.blocked),
        _listing("Unstored (outputs without store rows)", result.unstored),
        _listing("Databases written", result.databases),
        Text(f"Run time: {result.seconds:.1f}s"),
    ]
    if result.usage:
        parts.append(_usage_table(result.usage, result.total_usage))
    else:
        parts.append(Text("Token usage: none reported"))
    return Group(*parts)


def _note(planned: PlannedItem) -> str:
    """Return what the plan says about an item to process."""
    notes: list[str] = []
    if planned.resume.units:
        notes.append(f"{len(planned.resume.units)} chunk(s) to process")
    if planned.resume.note:
        notes.append(planned.resume.note)
    return "; ".join(notes)


def _settings_note(plan: RunPlan) -> Text:
    line = f"Schema: {plan.schema.name}"
    if "text" in plan.types or not plan.types:
        line += f"; text chunking: {plan.spec.chunking.mode}"
    return Text(line)


def _batch_note(plan: RunPlan) -> Text | None:
    if not plan.batch:
        return None
    return Text(
        f"Batch: {len(plan.to_process)} item(s) will be submitted to the "
        f"{plan.spec.model.provider} batch API; collect them with "
        "'chronominer batch check'."
    )


def plan_table(plan: RunPlan) -> RenderableType:
    """Return the items a plan processes and those it leaves out, with reasons."""
    left_out = (*plan.unstored, *plan.blocked, *plan.refused)
    title = (
        f"Plan: {len(plan.to_process)} to process, {len(plan.skipped)} already complete"
    )
    if left_out:
        title += f", {len(left_out)} left out"
    table = Table(title=title, title_justify="left")
    table.add_column("#", justify="right")
    table.add_column("Item")
    table.add_column("Type")
    table.add_column("State")
    table.add_column("Note")
    for index, planned in enumerate(plan.to_process, start=1):
        table.add_row(
            str(index),
            Text(planned.name),
            _kind_label(planned.type),
            planned.state,
            Text(_note(planned)),
        )
    if plan.skipped:
        table.add_section()
        for planned in plan.skipped:
            action = "render again" if planned.resume.render else "skip"
            table.add_row(
                "-",
                Text(planned.name),
                _kind_label(planned.type),
                f"complete ({action})",
                "",
            )
    if left_out:
        table.add_section()
        for planned in left_out:
            table.add_row(
                "-",
                Text(planned.name),
                _kind_label(planned.type),
                planned.state,
                Text(planned.reason),
            )
    parts: list[RenderableType] = [table, _settings_note(plan)]
    note = _batch_note(plan)
    if note is not None:
        parts.append(note)
    return Group(*parts)


def ranges_plan_table(plan: RangesPlan) -> RenderableType:
    """Return the files a ``ranges generate`` or ``ranges adjust`` plan processes.

    An adjust plan also names its schema, model and range selection; the
    warnings name the files left out.
    """
    table = Table(
        title=f"Plan ({plan.command}): {len(plan.files)} file(s)",
        title_justify="left",
    )
    table.add_column("#", justify="right")
    table.add_column("File")
    table.add_column("Line ranges")
    if plan.adjust:
        table.add_column("Context")
    for index, path in enumerate(plan.files, start=1):
        target = ranges_path(path)
        if plan.adjust:
            context = "yes" if plan.contexts.get(path) else "none"
            table.add_row(str(index), Text(path.name), Text(target.name), context)
        else:
            action = "replace" if target.is_file() else "write"
            table.add_row(
                str(index), Text(path.name), Text(f"{target.name} ({action})")
            )
    parts: list[RenderableType] = [table]
    spec = plan.spec
    if isinstance(spec, RangesAdjustSpec):
        schema = plan.schema.name if plan.schema is not None else spec.schema
        parts += [
            Text(f"Adjust schema: {schema}"),
            Text(
                f"Model: {spec.model.provider}:{spec.model.model}; response "
                f"format: {plan.response_format}; context window: "
                f"{spec.context_window} line(s)"
            ),
        ]
        if spec.pages:
            parts.append(Text(f"Line ranges selected: {spec.pages}"))
    else:
        parts.append(
            Text(f"Chunk size: {spec.chunk_tokens:,} tokens ({spec.model} tokenizer)")
        )
    parts += [Text(f"Warning: {message}", style="yellow") for message in plan.warnings]
    return Group(*parts)
