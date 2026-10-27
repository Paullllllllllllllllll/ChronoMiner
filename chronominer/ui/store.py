"""The render and drop tasks of the guided UI.

:func:`render_flow` re-renders the views of the tool's databases at an
input path; :func:`drop_flow` deletes stored documents. Their runs report
like ``chronominer render`` and ``chronominer drop`` and return the same
exit codes.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from rich.console import Console
from rich.text import Text

from chronominer.common.contract import ExitCode
from chronominer.common.selection import is_numeric_selection, select_items
from chronominer.common.store import (
    DocumentRow,
    DocumentSelectionError,
    StoreError,
    VersionlessDatabaseError,
    select_documents,
)
from chronominer.common.wizard import (
    BACK,
    DONE,
    Choice,
    Line,
    Nav,
    Review,
    State,
    Step,
    StepContext,
)
from chronominer.output.drop import (
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
    OUTPUT_FORMATS,
    RENDER_OPTIONS,
    DropSpec,
    RenderSpec,
    Resolution,
    SpecError,
    resolve_drop,
    resolve_render,
    to_argv,
)
from chronominer.ui.results import drop_result, render_result
from chronominer.ui.session import (
    TASKS,
    RunRequest,
    Session,
    TaskFlow,
    absolute,
    command_words,
    plural,
)

__all__ = ["drop_flow", "render_flow"]

logger = logging.getLogger(__name__)

RENDER_PROGRAM: Final = ("chronominer", "render")
DROP_PROGRAM: Final = ("chronominer", "drop")

_STORED = "stored_documents"
_KEEP = "keep"
_CHOOSE = "choose"
_ALL = "all"
_NUMBERS = "numbers"
_SEARCH = "search"
_LIST = "list"
_LIST_LIMIT: Final = 50


def _path_text(text: str) -> str | None:
    return "enter a path" if not text.strip() else None


def _input_step(message: str, validate: Callable[[str], str | None]) -> Step:
    def ask(ctx: StepContext) -> Nav:
        current = ctx.state.value("input")
        answer = ctx.prompter.path(
            message,
            default="" if current is None else str(absolute(current)),
            validate=validate,
        )
        if answer is BACK:
            return BACK
        ctx.state.set("input", absolute(answer))
        return DONE

    return Step("input", "Input", ask=ask, sets=("input",))


def _flag_step(
    name: str, title: str, message: str, off: Choice[bool], on: Choice[bool]
) -> Step:
    def ask(ctx: StepContext) -> Nav:
        return ctx.select_option(name, message, [off, on])

    return Step(name, title, ask=ask, sets=(name,))


def _dry_run_step(write: str) -> Step:
    return _flag_step(
        "dry_run",
        "Dry run",
        "Dry run:",
        Choice(False, "No", write),
        Choice(True, "Yes", "show the plan, write nothing"),
    )


def _formats_step() -> Step:
    def ask(ctx: StepContext) -> Nav:
        state = ctx.state
        chosen: dict[str, str] = {}

        def mode() -> Nav:
            current = _KEEP if state.value("formats") is None else _CHOOSE
            answer = ctx.prompter.select(
                "Formats:",
                [
                    Choice(_KEEP, "As rendered before", "each document's formats"),
                    Choice(_CHOOSE, "Choose formats"),
                ],
                default=current,
            )
            if answer is BACK:
                return BACK
            chosen["mode"] = answer
            if answer == _KEEP:
                state.set("formats", None)
            return DONE

        def formats() -> Nav | None:
            if chosen.get("mode") != _CHOOSE:
                return None
            checked = state.value("formats") or OUTPUT_FORMATS[:1]
            while True:
                answer = ctx.prompter.checkbox(
                    "Formats to render besides _output.json:",
                    [Choice(fmt, fmt) for fmt in OUTPUT_FORMATS],
                    checked=tuple(checked),
                )
                if answer is BACK:
                    return BACK
                if answer:
                    state.set("formats", tuple(answer))
                    return DONE
                ctx.prompter.show("  choose at least one format")

        return StepContext.sequence([mode, formats])

    return Step("formats", "Formats", ask=ask, sets=("formats",))


def _command(
    program: tuple[str, ...],
    resolve: Callable[[State, Session], Resolution[Any]],
) -> Callable[[State, Session], list[str]]:
    def command(state: State, session: Session) -> list[str]:
        spec = resolve(state, session).spec
        return command_words(program, to_argv(spec), session, state.table)

    return command


def _error(console: Console, message: str) -> None:
    console.print(Text(message, style="red"))


def _guarded(console: Console, body: Callable[[], int]) -> int:
    """Run *body* with the exit codes of the CLI for refusals and crashes."""
    try:
        return body()
    except StoreError as exc:
        _error(console, f"Error:{exc}")
        return exc.exit_code
    except KeyboardInterrupt:
        _error(console, "Interrupted by the user.")
        return ExitCode.INTERRUPTED
    except Exception as exc:
        logger.exception("Command failed: %s", exc)
        _error(console, f"Unexpected error: {exc}")
        return ExitCode.FAILURE


def _resolve_render(state: State, session: Session) -> Resolution[RenderSpec]:
    return resolve_render(state.answers)


def _render_refusal(path: Path) -> str | None:
    """Return why render cannot run on *path*, or None."""
    if not path.exists():
        return f"Input path does not exist: {path}"
    if path.is_file() and path.name != DATABASE_NAME:
        return f"{path} is not a {DATABASE_NAME} file"
    return None


@dataclass(frozen=True)
class _RenderPreview:
    databases: list[Path]
    results: list[DocumentRender]
    refused: str | None = None


def _preview_render(path: Path, spec: RenderSpec) -> _RenderPreview:
    databases = find_databases(path)
    results: list[DocumentRender] = []
    for database in databases:
        try:
            results.extend(
                preview_database(database, formats=spec.formats, force=spec.force)
            )
        except (StoreError, sqlite3.Error, OSError) as exc:
            return _RenderPreview(databases, results, refused=str(exc))
    return _RenderPreview(databases, results)


def _render_review(state: State, session: Session) -> Review:
    try:
        spec = _resolve_render(state, session).spec
    except SpecError as exc:
        return Review([Line("input", [str(state.value("input"))])], withheld=str(exc))
    path = absolute(spec.input)
    formats = (
        f"_output.json, {', '.join(spec.formats)}"
        if spec.formats
        else "as rendered before"
    )
    lines = [
        Line("formats", [formats]),
        Line(
            "force",
            [
                "replace files changed outside the tool"
                if spec.force
                else "keep files changed outside the tool"
            ],
        ),
        Line("run", ["dry run" if spec.dry_run else "write the files"]),
    ]
    refusal = _render_refusal(path)
    if refusal is not None:
        return Review([Line("input", [str(path)]), *lines], withheld=refusal)
    preview = _preview_render(path, spec)
    found = (
        f"{plural(len(preview.databases), 'database')}, "
        f"{plural(len(preview.results), 'document')}"
    )
    lines.insert(0, Line("input", [str(path), found]))
    if preview.refused is not None:
        return Review(lines, withheld=preview.refused)
    written = sum(len(result.written) for result in preview.results)
    unchanged = sum(len(result.unchanged) for result in preview.results)
    kept = sum(len(result.kept) for result in preview.results)
    notes = [
        f"Would write {plural(written, 'file')}, {unchanged} unchanged, "
        f"{kept} left as edited."
    ]
    return Review(lines, notes=notes)


def _execute_render(spec: RenderSpec, console: Console) -> int:
    path = spec.input.expanduser().resolve()
    if not path.exists():
        _error(console, f"Error:Input path does not exist: {path}")
        return ExitCode.USAGE
    if path.is_file() and path.name != DATABASE_NAME:
        _error(console, f"Error:{path} is not a {DATABASE_NAME} file")
        return ExitCode.USAGE
    databases = find_databases(path)
    for database in databases:
        try:
            read_store(database.parent)
        except VersionlessDatabaseError as exc:
            _error(console, f"Error:{exc}")
            return ExitCode.USAGE
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
    console.print(render_result(results, spec.dry_run))
    failed = any(result.failed for result in results)
    return ExitCode.FAILURE if failed else ExitCode.OK


def _run_render(request: RunRequest) -> int:
    try:
        spec = resolve_render(dict(request.answers)).spec
    except SpecError as exc:
        _error(request.console, f"Error:{exc}")
        return ExitCode.USAGE
    return _guarded(request.console, lambda: _execute_render(spec, request.console))


def render_flow() -> TaskFlow:
    """Return the render task: input, formats, force and dry run."""

    def steps(session: Session) -> list[Step]:
        return [
            _input_step(f"Input (folder or {DATABASE_NAME}):", _path_text),
            _formats_step(),
            _flag_step(
                "force",
                "Force",
                "Files changed outside the tool:",
                Choice(False, "Keep them"),
                Choice(True, "Replace them"),
            ),
            _dry_run_step("render the files"),
        ]

    return TaskFlow(
        name="render",
        title=TASKS["render"],
        program=RENDER_PROGRAM,
        table=RENDER_OPTIONS,
        steps=steps,
        review=_render_review,
        command=_command(RENDER_PROGRAM, _resolve_render),
        resolve=_resolve_render,
        run=_run_render,
    )


def _store_path(text: str) -> str | None:
    if not text.strip():
        return "enter a path"
    try:
        drop_root(absolute(text))
    except DropInputError as exc:
        return str(exc)
    return None


@dataclass(frozen=True)
class _Stored:
    root: Path
    documents: tuple[DocumentRow, ...] = ()
    refused: str | None = None


def _stored(state: State) -> _Stored | None:
    """Return the documents of the store at the input, cached by root."""
    current = state.value("input")
    if current is None:
        return None
    try:
        root = drop_root(absolute(current))
    except DropInputError as exc:
        return _Stored(absolute(current), refused=str(exc))
    cached = state.data.get(_STORED)
    if isinstance(cached, _Stored) and cached.root == root:
        return cached
    try:
        stored = _Stored(root, tuple(stored_documents(root)))
    except (DropInputError, StoreError, sqlite3.Error, OSError) as exc:
        stored = _Stored(root, refused=str(exc))
    state.data[_STORED] = stored
    return stored


def _document_mode(state: State) -> str:
    if state.value("files"):
        return _LIST
    expr = state.value("select")
    if expr:
        return _NUMBERS if is_numeric_selection(str(expr)) else _SEARCH
    return _ALL


class _DocumentQuestions:
    """Ask which stored documents to drop.

    The choices are all (``--all``), by name search (``--select``), or by
    number, range or from the list (``--files``, the chosen documents'
    names).
    """

    def __init__(self, ctx: StepContext) -> None:
        self.ctx = ctx
        self.state = ctx.state
        self.how: str | None = None

    def questions(self) -> list[Callable[[], Nav | None]]:
        return [self.which, self.expression, self.pick]

    def names(self) -> list[str]:
        stored = _stored(self.state)
        return [] if stored is None else [doc.name for doc in stored.documents]

    def which(self) -> Nav:
        names = self.names()
        choices = [Choice(_ALL, f"All {len(names)}")]
        if names:
            choices += [
                Choice(_NUMBERS, "By number or range", "such as 1,3 or 2-5"),
                Choice(_SEARCH, "By name search", "every name holding the text"),
                Choice(_LIST, "Choose from the list"),
            ]
        mode = _document_mode(self.state)
        default = mode if any(choice.value == mode for choice in choices) else _ALL
        answer = self.ctx.prompter.select("Which documents?", choices, default=default)
        if answer is BACK:
            return BACK
        self.how = answer
        if answer == _ALL:
            self.state.set("all", True)
        return DONE

    def expression(self) -> Nav | None:
        if self.how not in (_NUMBERS, _SEARCH):
            return None
        names = self.names()
        numbers = self.how == _NUMBERS
        if numbers:
            for number, name in enumerate(names[:_LIST_LIMIT], start=1):
                self.ctx.prompter.show(f"  {number:>4}  {name}")
            if len(names) > _LIST_LIMIT:
                self.ctx.prompter.show(f"  ... and {len(names) - _LIST_LIMIT} more")
        current = self.state.value("select")
        if current is not None and is_numeric_selection(str(current)) != numbers:
            current = None

        def validate(text: str) -> str | None:
            if not text.strip():
                return "enter numbers or ranges" if numbers else "enter a name part"
            if numbers and not is_numeric_selection(text):
                return "enter numbers or ranges, such as 1,3 or 2-5"
            if not numbers and is_numeric_selection(text):
                return "digits alone are document numbers; choose By number or range"
            selection = select_items(names, text)
            if selection.unmatched:
                return (
                    f"no document {', '.join(selection.unmatched)} of 1 to {len(names)}"
                )
            if not selection.indices:
                return "no document matches"
            return None

        answer = self.ctx.prompter.text(
            "Documents (numbers or ranges):" if numbers else "Name search:",
            default="" if current is None else str(current),
            validate=validate,
        )
        if answer is BACK:
            return BACK
        picked = select_items(names, answer).indices
        if numbers:
            # Names, not positions: a document stored before the run starts
            # would shift the numbers onto other documents.
            self.state.set("files", tuple(names[index] for index in picked))
        else:
            self.state.set("select", answer.strip())
        self.ctx.prompter.show(f"  {plural(len(picked), 'document')} chosen")
        return DONE

    def pick(self) -> Nav | None:
        if self.how != _LIST:
            return None
        names = self.names()
        current = [name for name in self.state.value("files") or () if name in names]
        while True:
            answer = self.ctx.prompter.checkbox(
                "Documents:", [Choice(name, name) for name in names], checked=current
            )
            if answer is BACK:
                return BACK
            if answer:
                self.state.set("files", tuple(answer))
                return DONE
            self.ctx.prompter.show("  select at least one document")


def _ask_documents(ctx: StepContext) -> Nav:
    return StepContext.sequence(_DocumentQuestions(ctx).questions())


def _resolve_drop(state: State, session: Session) -> Resolution[DropSpec]:
    return resolve_drop(state.answers)


def _shown(names: list[str], limit: int = 5) -> str:
    text = ", ".join(names[:limit])
    if len(names) > limit:
        text += f" and {len(names) - limit} more"
    return text


def _drop_review(state: State, session: Session) -> Review:
    run = Line("run", ["dry run" if state.value("dry_run") else "drop and delete"])
    try:
        spec = _resolve_drop(state, session).spec
    except SpecError as exc:
        return Review(
            [Line("input", [str(state.value("input"))]), run], withheld=str(exc)
        )
    stored = _stored(state)
    if stored is None or stored.refused is not None:
        reason = "no input path" if stored is None else stored.refused
        return Review(
            [Line("input", [str(absolute(spec.input))]), run], withheld=reason
        )
    stock = plural(len(stored.documents), "document") + " stored"
    lines = [Line("input", [str(stored.root / DATABASE_NAME), stock])]
    try:
        selection = select_documents(
            stored.documents, select=spec.select, files=spec.files, select_all=spec.all
        )
    except DocumentSelectionError as exc:
        return Review([*lines, run], withheld=str(exc))
    if not selection.documents:
        return Review([*lines, run], withheld="the store holds no document")
    names = [document.name for document in selection.documents]
    lines += [Line("documents", [plural(len(names), "document"), _shown(names)]), run]
    try:
        report = asyncio.run(
            drop_documents(stored.root, selection.documents, dry_run=True)
        )
    except (StoreError, sqlite3.Error, OSError) as exc:
        return Review(lines, withheld=str(exc))
    notes = [
        *selection.warnings,
        f"Would drop {plural(len(report.dropped), 'document')} and delete "
        f"{plural(len(report.deleted_files), 'file')}; {len(report.refused)} "
        f"refused, {len(report.kept_files)} file(s) kept.",
        *(f"Refused {item.name}: {item.reason}" for item in report.refused),
        *(f"Kept {item.path}: {item.reason}" for item in report.kept_files),
    ]
    return Review(lines, notes=notes)


def _execute_drop(spec: DropSpec, console: Console) -> int:
    try:
        root = drop_root(spec.input)
        selection = select_documents(
            stored_documents(root),
            select=spec.select,
            files=spec.files,
            select_all=spec.all,
        )
    except (DropInputError, DocumentSelectionError) as exc:
        _error(console, f"Error:{exc}")
        return ExitCode.USAGE
    if not selection.documents:
        _error(console, f"Error:{root / DATABASE_NAME} holds no document")
        return ExitCode.USAGE
    report = asyncio.run(
        drop_documents(root, selection.documents, dry_run=spec.dry_run)
    )
    console.print(drop_result(report, selection.warnings))
    return ExitCode.FAILURE if report.refused else ExitCode.OK


def _run_drop(request: RunRequest) -> int:
    try:
        spec = resolve_drop(dict(request.answers)).spec
    except SpecError as exc:
        _error(request.console, f"Error:{exc}")
        return ExitCode.USAGE
    return _guarded(request.console, lambda: _execute_drop(spec, request.console))


def drop_flow() -> TaskFlow:
    """Return the drop task: store, documents and dry run."""

    def steps(session: Session) -> list[Step]:
        return [
            _input_step(f"Store (output folder or {DATABASE_NAME}):", _store_path),
            Step(
                "documents",
                "Documents",
                ask=_ask_documents,
                sets=("all", "select", "files"),
            ),
            _dry_run_step("drop the documents and delete their files"),
        ]

    return TaskFlow(
        name="drop",
        title=TASKS["drop"],
        program=DROP_PROGRAM,
        table=DROP_OPTIONS,
        steps=steps,
        review=_drop_review,
        command=_command(DROP_PROGRAM, _resolve_drop),
        resolve=_resolve_drop,
        run=_run_drop,
    )
