"""Line-range adjustment: align chunk boundaries with semantic units.

One orchestration serves ``ranges adjust`` and ``extract --chunking
adjust``. For each range of a text file's ``_line_ranges.txt`` the model
(role ``adjust``) judges a window of lines around the range start, marked
by :data:`BOUNDARY_SENTINEL`; :mod:`~chronominer.ranges.decision` turns the
answers into a decision, which is committed to the store as the range's
unit before the next range starts. The file is then rendered again from
the store.

Starting ranges and resume: a file that is the store's own last rendering,
or whose ranges equal the stored starting ranges, resumes the stored
decisions; any other file (generated anew or edited by hand) is read as the
new starting ranges. ``force`` and changed adjust settings (boundary type,
context, prompt, context window) start over from the file. A missing file
is generated first when ``chunk_tokens`` is given. A model change alone
resumes.

Errors: a transient model failure that outlasts the caller's retries
answers as unsure for that window; a range whose every call failed is
stored as failed and decided again by the next run; when every call of a
file failed, :class:`AdjustDegradedError` is raised after the file is
rendered. Any other model error is stored with the range and propagates;
the file is not rendered then.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from chronominer.common.capture import use_recorder
from chronominer.common.retry import ErrorKind, classify_error
from chronominer.common.selection import PageRange
from chronominer.common.store import Lease, Store, UnitKey, file_sha256, unit_context
from chronominer.common.structured import (
    ResponseFormat,
    StructuredRequest,
    StructuredResult,
)
from chronominer.events import (
    ChunkDone,
    ItemFinished,
    ItemStarted,
    NullEvents,
    RunEvents,
)
from chronominer.llm.prompts import boundary_prompt, system_content
from chronominer.output.store import KIND_RANGE
from chronominer.ranges.decision import (
    DEFAULT_CONTEXT_WINDOW,
    SEMANTIC_BOUNDARY_SCHEMA,
    DecidedRange,
    RangeResult,
    decide_range,
    final_ranges,
    unsure_payload,
)
from chronominer.ranges.generate import token_ranges
from chronominer.ranges.lines import (
    Range,
    clamp_ranges,
    ranges_path,
    read_lines,
    read_ranges,
)
from chronominer.ranges.store import (
    LINE_RANGES_FORMAT,
    AttemptCollector,
    RangeUnit,
    group_by_folder,
    lease_document,
    open_folder_store,
    plan_ranges,
    ranges_document,
    recorded_sha256,
    render_ranges,
    stored_ranges,
    write_failure,
    write_result,
)
from chronominer.settings import PromptCaching

__all__ = [
    "BOUNDARY_SENTINEL",
    "ROLE",
    "AdjustDegradedError",
    "AdjustOptions",
    "AdjustReport",
    "BoundaryCaller",
    "adjust_files",
    "adjust_ranges",
    "boundary_user_text",
]

logger = logging.getLogger(__name__)

ROLE: Final = "adjust"
"""The model-call role of boundary requests."""

BOUNDARY_SENTINEL: Final = "<<<CURRENT_CHUNK_START>>>"
"""Inserted above the line where the range starts; never part of a marker."""

_DEGRADING: Final = frozenset(
    {
        ErrorKind.RATE_LIMIT,
        ErrorKind.TIMEOUT,
        ErrorKind.CONNECTION,
        ErrorKind.SERVER_ERROR,
        ErrorKind.OTHER,
    }
)


class BoundaryCaller(Protocol):
    """Sends boundary requests; :class:`~chronominer.llm.caller.ModelCaller`.

    The caller retries transient errors itself and raises the last one.
    """

    provider: str

    async def call(
        self, messages: Sequence[Any], request: StructuredRequest, *, label: str
    ) -> StructuredResult: ...


class AdjustDegradedError(RuntimeError):
    """Every model call for a file failed; its ranges stayed as they were."""

    def __init__(self, name: str, degraded: int, total: int) -> None:
        self.name = name
        self.degraded = degraded
        self.total = total
        super().__init__(
            f"all {total} model call(s) for {name} failed; the line ranges were "
            "left unadjusted"
        )


@dataclass(frozen=True)
class AdjustOptions:
    """The settings of one adjustment.

    *boundary_type* names the semantic unit (the schema name); *context* is
    the resolved ``adjust`` context text. *pages* selects ranges by
    position; the others keep their bounds. *chunk_tokens* generates a
    missing ``_line_ranges.txt``; without it a missing file is an error.
    *prompt_caching* marks the system prompt for caching as extraction does.
    """

    boundary_type: str
    context: str | None = None
    response_format: ResponseFormat = "schema"
    context_window: int = DEFAULT_CONTEXT_WINDOW
    pages: PageRange | None = None
    force: bool = False
    chunk_tokens: int | None = None
    prompt_caching: PromptCaching = PromptCaching()


@dataclass
class AdjustReport:
    """The outcome of adjusting one file.

    *status* is ``adjusted``, ``complete`` (every selected range was already
    decided with these settings) or ``empty`` (no ranges). *decided* counts
    the ranges decided in this run, *calls* their model answers.
    """

    text_file: Path
    ranges_file: Path
    status: str
    ranges: tuple[Range, ...] = ()
    decided: int = 0
    calls: int = 0
    degraded: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Tally:
    """Model calls of a file and those that failed into an unsure answer."""

    total: int = 0
    degraded: int = 0


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def boundary_user_text(
    lines: Sequence[str],
    start: int,
    window: Range,
    failed_markers: Sequence[str] = (),
) -> str:
    """Return the user message: the window's lines and the failed markers.

    The sentinel goes above line *start* when the window shows it.
    """
    block: list[str] = []
    for number in range(window[0], window[1] + 1):
        if number == start:
            block.append(BOUNDARY_SENTINEL)
        block.append(lines[number - 1])
    text = "Input text:\n\n" + "\n".join(block) + "\n"
    failed = [marker.strip() for marker in failed_markers if marker.strip()]
    if failed:
        text += (
            "\nThe following semantic markers previously failed to match the "
            "source text. Do not reuse any of them. Provide a new 5-15 character "
            "substring that excludes these markers:\n"
            + "\n".join(f"- {marker}" for marker in failed)
            + "\n"
        )
    return text


def _degrades(exc: BaseException) -> bool:
    """Whether a failed call answers as unsure instead of failing the file."""
    return classify_error(exc) in _DEGRADING


@dataclass(frozen=True)
class _Context:
    """What every range of one file shares."""

    store: Store
    lease: Lease
    name: str
    caller: BoundaryCaller
    model: str
    options: AdjustOptions
    lines: Sequence[str]
    system: str
    settings: Mapping[str, Any]


def _settings(options: AdjustOptions, system: str) -> dict[str, Any]:
    """Return the adjust settings a decision is valid for."""
    return {
        "boundary_type": options.boundary_type,
        "context_window": options.context_window,
        "prompt_sha256": _sha256(system),
    }


async def adjust_ranges(
    store: Store,
    text_file: Path,
    *,
    caller: BoundaryCaller,
    model: str,
    options: AdjustOptions,
    document: str | None = None,
    events: RunEvents | None = None,
) -> AdjustReport:
    """Adjust the line ranges of *text_file* and render its ``_line_ranges.txt``.

    *store* is open with a run begun; *document* names the document that
    holds the range units, by default the text file's document in its own
    folder. *model* stamps units written without a model answer.

    Raises:
        FileNotFoundError: The line-ranges file is missing and
            ``options.chunk_tokens`` is None.
        AdjustDegradedError: Every model call failed.
        DocumentBusyError: Another live run holds the document.
    """
    sink = events or NullEvents()
    text_file = text_file.resolve()
    ranges_file = ranges_path(text_file)
    lines = read_lines(text_file)
    name = document or ranges_document(text_file)
    system = boundary_prompt(
        options.boundary_type, SEMANTIC_BOUNDARY_SCHEMA, options.context
    )
    report = AdjustReport(text_file, ranges_file, "adjusted")
    tally = _Tally()
    lease = await lease_document(store, text_file, name)
    try:
        context = _Context(
            store,
            lease,
            name,
            caller,
            model,
            options,
            lines,
            system,
            _settings(options, system),
        )
        units = await _starting_units(context, ranges_file, report)
        if not units:
            report.status = "empty"
            return report
        await _decide_pending(context, units, report, tally, sink)
        rendered = await render_ranges(store, lease, ranges_file)
        report.warnings.extend(rendered.warnings)
        for warning in rendered.warnings:
            sink.warning(warning)
        report.ranges = tuple(_current_ranges(store, lease.doc_id))
    finally:
        await store.release(lease)
    if tally.total and tally.degraded >= tally.total:
        raise AdjustDegradedError(name, tally.degraded, tally.total)
    return report


def _current_ranges(store: Store, doc_id: int) -> list[Range]:
    units = stored_ranges(store, doc_id)
    return final_ranges(
        [
            DecidedRange(
                unit.original, unit.result if unit.status != "pending" else None
            )
            for unit in units
        ]
    )


def _settings_changed(units: Sequence[RangeUnit], settings: Mapping[str, Any]) -> bool:
    return any(
        unit.decided and dict(unit.settings or {}) != dict(settings) for unit in units
    )


async def _starting_units(
    context: _Context, ranges_file: Path, report: AdjustReport
) -> list[RangeUnit]:
    """Return the range units to work from, replanning them when needed."""
    store, lease, options = context.store, context.lease, context.options
    provider = context.caller.provider
    if not ranges_file.exists():
        if options.chunk_tokens is None:
            raise FileNotFoundError(f"Line ranges file not found: {ranges_file}")
        stripped = [line.strip() for line in context.lines]
        generated = token_ranges(stripped, options.chunk_tokens, context.model)
        await plan_ranges(
            store,
            lease,
            generated,
            model=context.model,
            provider=provider,
            generated=True,
        )
        await render_ranges(store, lease, ranges_file, force=True)
        logger.info("Generated %s with %d range(s)", ranges_file.name, len(generated))
    file_ranges = clamp_ranges(read_ranges(ranges_file), len(context.lines))
    units = stored_ranges(store, lease.doc_id)
    current = file_sha256(ranges_file)
    recorded = recorded_sha256(store, lease.doc_id, ranges_file)
    own = recorded is not None and current == recorded
    same = [unit.original for unit in units] == file_ranges
    changed = _settings_changed(units, context.settings)
    if changed and not options.force:
        message = (
            f"The adjust settings of '{context.name}' changed; its ranges are "
            "adjusted again from the current file."
        )
        logger.warning(message)
        report.warnings.append(message)
    if units and (own or same) and not options.force and not changed:
        return units
    if not file_ranges:
        logger.warning("No line ranges in %s", ranges_file)
        return []
    await plan_ranges(store, lease, file_ranges, model=context.model, provider=provider)
    if current is not None and current != recorded:
        await store.register_output(
            lease, ranges_file, format=LINE_RANGES_FORMAT, sha256=current
        )
    return stored_ranges(store, lease.doc_id)


def _selected(units: Sequence[RangeUnit], pages: PageRange | None) -> set[int]:
    indices = [unit.idx for unit in units]
    if pages is None:
        return set(indices)
    return {indices[position] for position in pages.resolve(len(indices))}


async def _decide_pending(
    context: _Context,
    units: Sequence[RangeUnit],
    report: AdjustReport,
    tally: _Tally,
    events: RunEvents,
) -> None:
    """Decide the selected ranges that have no decision yet."""
    selected = _selected(units, context.options.pages)
    todo = [unit for unit in units if unit.idx in selected and not unit.decided]
    if not todo:
        report.status = "complete"
        return
    done = sum(unit.decided for unit in units)
    for position, unit in enumerate(todo, 1):
        await context.store.wait_capacity()
        result, degraded = await _decide_one(context, unit, tally)
        report.decided += 1
        report.calls += result.calls
        report.degraded += int(degraded)
        events.chunk_done(
            ChunkDone(
                item=context.name,
                chunk=unit.idx,
                status="failed" if degraded else "ok",
                done=position,
                total=len(todo),
                already_complete=done,
            )
        )
    if tally.degraded:
        message = (
            f"{context.name}: {tally.degraded} of {tally.total} model calls failed "
            "and were treated as unsure"
        )
        logger.warning(message)
        report.warnings.append(message)
        events.warning(message)


async def _decide_one(
    context: _Context, unit: RangeUnit, tally: _Tally
) -> tuple[RangeResult, bool]:
    """Decide one range and commit it; return the result and whether it degraded."""
    store, lease = context.store, context.lease
    collector = AttemptCollector(store)
    calls = degraded = 0
    request = StructuredRequest(
        provider=context.caller.provider,
        response_format=context.options.response_format,
        schema=SEMANTIC_BOUNDARY_SCHEMA,
        required=(),
    )
    system = SystemMessage(
        content=system_content(
            context.system,
            provider=context.caller.provider,
            model=context.model,
            caching=context.options.prompt_caching,
        )
    )

    async def ask(
        window: Range, window_index: int, failed: Sequence[str]
    ) -> Mapping[str, Any]:
        nonlocal calls, degraded
        calls += 1
        tally.total += 1
        text = boundary_user_text(context.lines, unit.original[0], window, failed)
        label = f"{context.name} range {unit.idx + 1} window {window_index}"
        try:
            answer = await context.caller.call(
                [system, HumanMessage(content=text)], request, label=label
            )
        except Exception as exc:
            if not _degrades(exc):
                raise
            degraded += 1
            tally.degraded += 1
            logger.error(
                "Boundary call for %s failed; treating it as unsure: %s", label, exc
            )
            return unsure_payload()
        if answer.data is None:
            logger.warning(
                "Boundary answer for %s is not a JSON object (%s); treating it as "
                "unsure",
                label,
                answer.error,
            )
            return unsure_payload()
        return answer.data

    key = UnitKey(lease.doc_id, KIND_RANGE, unit.idx)
    with unit_context(key), use_recorder(collector):
        try:
            result = await decide_range(
                ask,
                context.lines,
                unit.original,
                unit.idx + 1,
                context_window=context.options.context_window,
                boundary_type=context.options.boundary_type,
            )
        except Exception as exc:
            await write_failure(
                store,
                lease,
                unit.idx,
                unit.original,
                exc,
                model=context.model,
                provider=context.caller.provider,
                collector=collector,
            )
            raise
    failed_all = calls > 0 and degraded >= calls
    await write_result(
        store,
        lease,
        result,
        model=context.model,
        provider=context.caller.provider,
        collector=collector,
        settings=context.settings,
        degraded=failed_all,
    )
    return result, failed_all


async def adjust_files(
    text_files: Sequence[Path],
    *,
    caller: BoundaryCaller,
    model: str,
    options: AdjustOptions,
    events: RunEvents | None = None,
    run_spec: Mapping[str, Any] | None = None,
) -> list[tuple[Path, AdjustReport | BaseException]]:
    """Run ``ranges adjust``: adjust each file with the store of its folder.

    Each folder's ``chronominer.sqlite`` records one run. A file that fails
    is returned with its exception and the others go on.
    """
    sink = events or NullEvents()
    outcomes: list[tuple[Path, AdjustReport | BaseException]] = []
    total = len(text_files)
    for folder, files in group_by_folder(text_files).items():
        async with open_folder_store(folder) as store:
            await store.begin_run("ranges adjust", spec=run_spec)
            failed = False
            for text_file in files:
                index = len(outcomes) + 1
                sink.item_started(ItemStarted(text_file.name, "text", index, total))
                outcome = await _adjust_one(
                    store, text_file, caller, model, options, sink
                )
                ok = isinstance(outcome, AdjustReport)
                failed = failed or not ok
                sink.item_finished(
                    ItemFinished(
                        text_file.name,
                        index,
                        total,
                        ok,
                        outputs=(str(ranges_path(text_file)),) if ok else (),
                        error=None if ok else str(outcome),
                    )
                )
                outcomes.append((text_file, outcome))
            await store.end_run(1 if failed else 0)
    return outcomes


async def _adjust_one(
    store: Store,
    text_file: Path,
    caller: BoundaryCaller,
    model: str,
    options: AdjustOptions,
    events: RunEvents,
) -> AdjustReport | BaseException:
    try:
        return await adjust_ranges(
            store, text_file, caller=caller, model=model, options=options, events=events
        )
    except Exception as exc:  # noqa: BLE001 - reported per file; the others go on
        logger.error("Adjusting %s failed: %s", text_file.name, exc)
        return exc
