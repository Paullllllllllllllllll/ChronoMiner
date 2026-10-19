"""The ranges tasks of the guided UI: generate and adjust line ranges.

Generate asks the input, the tokens per chunk, the tokenizer model and the
run mode. Adjust asks the input with its range selection, the schema, the
context window and context, the model, the response format and the run
options. Each review previews the plan (the files and its warnings) and
withholds Run when the run cannot be planned. The runs mirror ``chronominer
ranges generate`` and ``chronominer ranges adjust``: plan, the extension's
startup, then the run with rich progress.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from rich.console import Console, RenderableType
from rich.text import Text

from chronominer import ext
from chronominer.common.contract import ExitCode, exit_code_for
from chronominer.common.options import OptionTable, Source
from chronominer.common.selection import parse_page_range
from chronominer.common.wizard import (
    BACK,
    DONE,
    Choice,
    Line,
    Nav,
    Part,
    Review,
    State,
    Step,
    StepContext,
)
from chronominer.output.store import database_path
from chronominer.run import PlanError, RangesPlan, plan_ranges, run_ranges
from chronominer.settings import (
    Settings,
    SettingsError,
    resolve_ranges_adjust,
    resolve_ranges_generate,
)
from chronominer.spec import (
    RANGES_ADJUST_OPTIONS,
    RANGES_GENERATE_OPTIONS,
    RangesAdjustSpec,
    RangesGenerateSpec,
    Resolution,
    condition_warnings,
    to_argv,
)
from chronominer.ui.progress import RichProgress, ranges_plan_table, summary_table
from chronominer.ui.questions import (
    MODEL_OPTIONS,
    ModelQuestions,
    Question,
    ResponseQuestions,
    chunk_tokens_question,
    context_text,
    fallback_text,
    model_line,
    support_of,
)
from chronominer.ui.session import (
    TASKS,
    Inventory,
    RunRequest,
    Session,
    TaskFlow,
    absolute,
    command_words,
    inventory_of,
    plural,
)

__all__ = ["ADJUST_PROGRAM", "GENERATE_PROGRAM", "adjust_flow", "generate_flow"]

GENERATE_PROGRAM: Final = ("chronominer", "ranges", "generate")
ADJUST_PROGRAM: Final = ("chronominer", "ranges", "adjust")

_INPUT_OPTIONS: Final = ("input", "recursive", "pages")
_PREVIEW_WIDTH: Final = 100
_GENERATE_FIRST: Final = "run 'chronominer ranges generate' first"

type Resolver = Callable[[Settings, Mapping[str, Any]], Resolution[Any]]


def _existing(text: str) -> str | None:
    if not text.strip():
        return "enter a file or a folder"
    if not absolute(text).exists():
        return "no such file or folder"
    return None


def _page_selection(text: str) -> str | None:
    if not text.strip():
        return None
    try:
        parse_page_range(text.strip())
    except ValueError as exc:
        return str(exc)
    return None


def _model_name(text: str) -> str | None:
    return None if text.strip() else "enter a model name"


def _input(state: State) -> Path | None:
    value = state.value("input")
    return None if value is None else absolute(value)


def _texts(inventory: Inventory) -> list[int]:
    return inventory.of_type("text")


def _ranged(inventory: Inventory) -> list[int]:
    return [index for index in _texts(inventory) if inventory.has_ranges(index)]


class _InputQuestions:
    """The questions of the input step: path, subfolders, a check, selection.

    For adjust only the text files with a ``_line_ranges.txt`` count; an
    input without any is refused with a pointer to ``ranges generate``.
    """

    def __init__(self, ctx: StepContext, *, adjust: bool) -> None:
        self.ctx = ctx
        self.state = ctx.state
        self.adjust = adjust

    def questions(self) -> list[Question]:
        return [self.path, self.recursive, self.found, self.pages]

    def _show_found(self, location: Path) -> None:
        inventory = inventory_of(self.state, location)
        texts = _texts(inventory)
        if not self.adjust:
            self.ctx.prompter.show(f"  found {plural(len(texts), 'text file')}")
            return
        ranged = _ranged(inventory)
        line = f"  found {plural(len(ranged), 'text file')} with line ranges"
        others = len(texts) - len(ranged)
        if ranged and others:
            line += f"; {plural(others, 'text file')} without them left out"
        self.ctx.prompter.show(line)

    def path(self) -> Nav:
        state = self.state
        current = _input(state)
        message = (
            "Text file or folder whose line ranges to adjust:"
            if self.adjust
            else "Text file or folder of text files:"
        )
        answer = self.ctx.prompter.path(
            message,
            default="" if current is None else str(current),
            validate=_existing,
        )
        if answer is BACK:
            return BACK
        location = absolute(answer)
        state.set("input", location)
        if not location.is_dir():
            state.clear("recursive")
        self._show_found(location)
        return DONE

    def recursive(self) -> Nav | None:
        location = _input(self.state)
        if location is None or not location.is_dir():
            return None
        before = bool(self.state.value("recursive"))
        result = self.ctx.select_option(
            "recursive",
            "Search the subfolders too?",
            [
                Choice(False, "No, this folder only"),
                Choice(True, "Yes, every subfolder"),
            ],
        )
        if result is DONE and bool(self.state.value("recursive")) != before:
            self._show_found(location)
        return result

    def _empty(self) -> str | None:
        """Return why the input holds nothing to process, or None."""
        location = _input(self.state)
        if location is None:
            return None
        inventory = inventory_of(self.state, location)
        if self.adjust and _ranged(inventory):
            return None
        if not self.adjust and _texts(inventory):
            return None
        if self.adjust and _texts(inventory):
            return f"  no text file here has a _line_ranges.txt; {_GENERATE_FIRST}"
        message = "  no text file here; choose another input"
        if location.is_dir() and not self.state.value("recursive"):
            message += " or search the subfolders"
        return message

    def found(self) -> Nav | None:
        """Ask the path again, in place, while the input holds nothing."""
        message = self._empty()
        if message is None:
            return None
        while message is not None:
            self.ctx.prompter.show(message)
            for question in (self.path, self.recursive):
                if question() is BACK:
                    return BACK
            message = self._empty()
        return DONE

    def pages(self) -> Nav | None:
        if "pages" not in self.state.table:
            return None
        return self.ctx.text_option(
            "pages",
            "Line ranges to adjust (5, first:5, last:5, 3-7 or 1,3,5-8; "
            "empty = every one):",
            validate=_page_selection,
            empty=None,
        )


def _input_step(table: OptionTable, *, adjust: bool) -> Step:
    return Step(
        "input",
        "Input",
        ask=lambda ctx: ctx.sequence(_InputQuestions(ctx, adjust=adjust).questions()),
        sets=tuple(name for name in _INPUT_OPTIONS if name in table),
    )


def _dry_run(ctx: StepContext, doing: str) -> Nav:
    return ctx.select_option(
        "dry_run",
        "Dry run:",
        [
            Choice(False, f"No, {doing}"),
            Choice(True, "Yes, only report the files"),
        ],
    )


def _ask_tokenizer(ctx: StepContext) -> Nav:
    return ctx.text_option(
        "model", "Tokenizer model (counts the chunk tokens):", validate=_model_name
    )


def _generate_steps(session: Session) -> list[Step]:
    return [
        _input_step(RANGES_GENERATE_OPTIONS, adjust=False),
        Step(
            "chunking",
            "Chunking",
            ask=lambda ctx: ctx.sequence(
                [chunk_tokens_question(ctx), lambda: _ask_tokenizer(ctx)]
            ),
            sets=("chunk_tokens", "model"),
        ),
        Step(
            "run",
            "Run options",
            ask=lambda ctx: _dry_run(ctx, "write the line ranges"),
            sets=("dry_run",),
        ),
    ]


def _has_store(state: State) -> bool:
    """Whether a database lies beside a text file with line ranges."""
    location = _input(state)
    if location is None:
        return False
    inventory = inventory_of(state, location)
    return any(
        database_path(inventory.items[index].path.parent).is_file()
        for index in _ranged(inventory)
    )


def _ask_adjust_run(ctx: StepContext) -> Nav:
    state = ctx.state

    def concurrency() -> Nav:
        return ctx.text_option("concurrency", "Parallel requests:")

    def force() -> Nav | None:
        if not _has_store(state):
            state.clear("force")
            return None
        return ctx.select_option(
            "force",
            "Stored decisions:",
            [
                Choice(False, "Keep them", "adjust only the undecided boundaries"),
                Choice(True, "Reset them", "adjust every boundary again"),
            ],
        )

    def dry_run() -> Nav:
        return _dry_run(ctx, "adjust the line ranges")

    questions: list[Question] = [concurrency, force, dry_run]
    return ctx.sequence(questions)


def _ask_context(ctx: StepContext, session: Session) -> Nav:
    def window() -> Nav:
        return ctx.text_option(
            "context_window", "Context window (lines around each boundary):"
        )

    context = ResponseQuestions(ctx, session).context_questions()
    return ctx.sequence([window, *context])


def _ask_schema(ctx: StepContext, session: Session) -> Nav:
    questions = ResponseQuestions(ctx, session)
    return ctx.sequence([questions.schema, questions.schema_file])


def _ask_response(ctx: StepContext, session: Session) -> Nav:
    answer = ResponseQuestions(ctx, session).response_format()
    return DONE if answer is None else answer


def _adjust_steps(session: Session) -> list[Step]:
    return [
        _input_step(RANGES_ADJUST_OPTIONS, adjust=True),
        Step(
            "schema",
            "Schema",
            ask=lambda ctx: _ask_schema(ctx, session),
            sets=("schema",),
        ),
        Step(
            "context",
            "Context",
            ask=lambda ctx: _ask_context(ctx, session),
            sets=("context_window", "context", "fallback_context"),
        ),
        Step(
            "model",
            "Model",
            ask=lambda ctx: ctx.sequence(ModelQuestions(ctx, session).questions()),
            sets=MODEL_OPTIONS,
        ),
        Step(
            "response",
            "Response",
            ask=lambda ctx: _ask_response(ctx, session),
            sets=("response_format",),
        ),
        Step(
            "run",
            "Run options",
            ask=_ask_adjust_run,
            sets=("concurrency", "force", "dry_run"),
        ),
    ]


def _resolve_generate(state: State, session: Session) -> Resolution[RangesGenerateSpec]:
    return resolve_ranges_generate(session.settings, state.answers)


def _resolve_adjust(state: State, session: Session) -> Resolution[RangesAdjustSpec]:
    return resolve_ranges_adjust(session.settings, state.answers)


def _marks(sources: Mapping[str, Source]) -> Callable[[str, str], Part]:
    def mark(target: str, text: str) -> Part:
        return Part(text, sources.get(target) is Source.SETTINGS)

    return mark


def _input_line(path: Path, recursive: bool) -> Line:
    parts = [str(absolute(path))]
    if recursive:
        parts.append("recursive")
    return Line("input", parts)


def _generate_lines(resolution: Resolution[RangesGenerateSpec]) -> list[Line]:
    spec, mark = resolution.spec, _marks(resolution.sources)
    return [
        _input_line(spec.input, spec.recursive),
        Line(
            "chunking",
            [
                mark("chunk_tokens", f"{spec.chunk_tokens:,} tokens per chunk"),
                mark("model", f"tokenizer {spec.model}"),
            ],
        ),
        Line("run", ["dry run" if spec.dry_run else "write the line ranges"]),
    ]


def _adjust_lines(
    resolution: Resolution[RangesAdjustSpec], session: Session
) -> list[Line]:
    spec, mark = resolution.spec, _marks(resolution.sources)
    input_line = _input_line(spec.input, spec.recursive)
    selection = f"line ranges {spec.pages}" if spec.pages else "every line range"
    chosen = spec.model.response_format
    if chosen is None:
        automatic = support_of(spec.model, session.settings).automatic_format
        response = f"automatic ({automatic})"
    else:
        response = chosen
    run: list[Part | str] = [
        mark("run.concurrency", f"{spec.run.concurrency} parallel requests")
    ]
    if spec.run.force:
        run.append("reset the stored decisions")
    if spec.run.dry_run:
        run.append("dry run")
    context: list[Part | str] = [
        mark("context_window", f"window {spec.context_window} lines"),
        context_text(spec.context, "context"),
    ]
    fallback = fallback_text(spec.fallback_context)
    if spec.context is None and fallback is not None:
        context.append(fallback)
    return [
        Line("input", [*input_line.parts, selection]),
        Line("schema", [mark("schema", spec.schema)]),
        Line("context", context),
        model_line(spec.model, resolution.sources, session),
        Line("response", [mark("model.response_format", response)]),
        Line("run", run),
    ]


def _rendered(renderable: RenderableType) -> list[str]:
    """Return the plain-text lines of a rich renderable."""
    buffer = io.StringIO()
    console = Console(file=buffer, width=_PREVIEW_WIDTH, highlight=False)
    console.print(renderable)
    return [line.rstrip() for line in buffer.getvalue().rstrip().splitlines()]


def _review(
    state: State,
    session: Session,
    table: OptionTable,
    resolve: Callable[[State, Session], Resolution[Any]],
) -> Review:
    try:
        resolution = resolve(state, session)
    except ValueError as exc:
        return Review([], withheld=str(exc))
    spec = resolution.spec
    if isinstance(spec, RangesAdjustSpec):
        lines = _adjust_lines(resolution, session)
    else:
        lines = _generate_lines(resolution)
    notes = [
        f"Warning: {message}"
        for message in condition_warnings(
            table, state.answers, session.settings.defaults, spec
        )
    ]
    try:
        ranges_plan = plan_ranges(
            spec, session.settings, sources=resolution.sources, args=session.args
        )
    except PlanError as exc:
        return Review(lines, notes=notes, withheld=str(exc))
    return Review(lines, notes=[*notes, *_rendered(ranges_plan_table(ranges_plan))])


def _command(
    state: State,
    session: Session,
    program: Sequence[str],
    table: OptionTable,
    resolve: Callable[[State, Session], Resolution[Any]],
) -> list[str]:
    spec = resolve(state, session).spec
    argv = to_argv(spec, session.settings.defaults)
    provider = spec.model.provider if isinstance(spec, RangesAdjustSpec) else None
    return command_words(program, argv, session, table, provider=provider)


def _error(request: RunRequest, message: str) -> None:
    request.console.print(Text(message, style="red"), soft_wrap=True)


def _report_dry_run(request: RunRequest, ranges_plan: RangesPlan) -> None:
    request.console.print(ranges_plan_table(ranges_plan))
    request.console.print(
        Text(
            f"[DRY RUN] {ranges_plan.command}: {len(ranges_plan.files)} text "
            "file(s) would be processed."
        )
    )


def _run(request: RunRequest, resolve: Resolver) -> int:
    settings = request.session.settings
    flags: dict[str, Any] = dict(request.answers)
    try:
        resolution = resolve(settings, flags)
    except ValueError as exc:
        _error(request, f"Error: {exc}")
        return int(ExitCode.USAGE)
    spec = resolution.spec
    try:
        ranges_plan = plan_ranges(
            spec, settings, sources=resolution.sources, args=request.args
        )
    except PlanError as exc:
        _error(request, f"Error: {exc}")
        return int(exc.exit_code)
    try:
        ext.startup(spec, settings=settings, args=request.args)
    except SettingsError as exc:
        _error(request, f"Error: {exc}")
        return int(ExitCode.USAGE)
    if ranges_plan.dry_run:
        _report_dry_run(request, ranges_plan)
        return int(ExitCode.OK)
    with RichProgress(request.console) as progress:
        try:
            result = asyncio.run(run_ranges(ranges_plan, progress))
        except PlanError as exc:
            _error(request, f"Error: {exc}")
            return int(exc.exit_code)
    request.console.print(summary_table(result, progress))
    return int(exit_code_for(failed=result.failed))


def generate_flow() -> TaskFlow:
    """Return the ranges generate task."""
    return TaskFlow(
        name="ranges-generate",
        title=TASKS["ranges-generate"],
        program=GENERATE_PROGRAM,
        table=RANGES_GENERATE_OPTIONS,
        steps=_generate_steps,
        review=lambda state, session: _review(
            state, session, RANGES_GENERATE_OPTIONS, _resolve_generate
        ),
        command=lambda state, session: _command(
            state,
            session,
            GENERATE_PROGRAM,
            RANGES_GENERATE_OPTIONS,
            _resolve_generate,
        ),
        resolve=_resolve_generate,
        run=lambda request: _run(request, resolve_ranges_generate),
    )


def adjust_flow() -> TaskFlow:
    """Return the ranges adjust task."""
    return TaskFlow(
        name="ranges-adjust",
        title=TASKS["ranges-adjust"],
        program=ADJUST_PROGRAM,
        table=RANGES_ADJUST_OPTIONS,
        steps=_adjust_steps,
        review=lambda state, session: _review(
            state, session, RANGES_ADJUST_OPTIONS, _resolve_adjust
        ),
        command=lambda state, session: _command(
            state, session, ADJUST_PROGRAM, RANGES_ADJUST_OPTIONS, _resolve_adjust
        ),
        resolve=_resolve_adjust,
        run=lambda request: _run(request, resolve_ranges_adjust),
    )
