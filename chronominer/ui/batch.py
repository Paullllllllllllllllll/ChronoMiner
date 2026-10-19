"""The batch tasks of the guided UI: check and finalize, or cancel, batches.

:func:`check_flow` asks the folder, the schema filter, the batch IDs, what
to do with an unresolved submission and the formats (those of the
submitting run, or chosen ones); :func:`cancel_flow`
asks what to cancel: named batch IDs, every unfinished batch, or the open
submissions below a folder. The runs mirror ``chronominer batch check`` and
``batch cancel``, exit codes included. :func:`schema_filter` is the
``--schema`` question the check and repair tasks share.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, Final

from rich.text import Text

from chronominer import ext
from chronominer.batch.cancel import (
    CancelReport,
    cancel_batches,
    cancel_submissions,
    cancel_unfinished,
)
from chronominer.batch.check import (
    CheckTargetError,
    check_submissions,
    find_databases,
)
from chronominer.common.contract import ExitCode, exit_code_for
from chronominer.common.options import OptionTable
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
from chronominer.settings import (
    SettingsError,
    resolve_batch_cancel,
    resolve_batch_check,
)
from chronominer.spec import (
    BATCH_CANCEL_OPTIONS,
    BATCH_CHECK_OPTIONS,
    OUTPUT_FORMATS,
    BatchCancelSpec,
    BatchCheckSpec,
    CommandSpec,
    Resolution,
    condition_warnings,
    schema_names,
    to_argv,
)
from chronominer.ui.questions import OTHER
from chronominer.ui.results import cancel_result, check_result
from chronominer.ui.session import (
    TASKS,
    RunRequest,
    Session,
    TaskFlow,
    absolute,
    command_words,
    plural,
)

__all__ = [
    "CANCEL_PROGRAM",
    "CHECK_PROGRAM",
    "cancel_flow",
    "check_flow",
    "condition_notes",
    "schema_filter",
    "schema_text",
]

CHECK_PROGRAM: Final = ("chronominer", "batch", "check")
CANCEL_PROGRAM: Final = ("chronominer", "batch", "cancel")

_REPORT = "report"
_ADOPT = "adopt"
_ABANDON = "abandon"
_IDS = "ids"
_ALL = "all"
_DIR = "dir"


def _existing_folder(text: str) -> str | None:
    if not text.strip():
        return "enter a folder"
    if not absolute(text).is_dir():
        return "no such folder"
    return None


def _given(text: str) -> str | None:
    return None if text.strip() else "enter an ID"


def _named(text: str) -> str | None:
    return None if text.strip() else "enter a schema name or file"


def _databases(folder: Path) -> int:
    return len(find_databases(folder)) if folder.is_dir() else 0


def _error(request: RunRequest, message: str) -> None:
    request.console.print(Text(message, style="red"), soft_wrap=True)


def _is_folder(request: RunRequest, directory: Path) -> bool:
    """Return whether *directory* exists; print the refusal when it does not."""
    if directory.is_dir():
        return True
    _error(request, f"Error: Input folder does not exist: {directory}")
    return False


def _startup(request: RunRequest, spec: BatchCheckSpec | BatchCancelSpec) -> bool:
    """Start the extension; print the refusal and return False on one."""
    try:
        ext.startup(spec, settings=request.session.settings, args=request.args)
    except SettingsError as exc:
        _error(request, f"Error: {exc}")
        return False
    return True


def condition_notes(
    table: OptionTable, state: State, session: Session, spec: CommandSpec
) -> list[str]:
    """Return the CLI's condition warnings for the answers as review notes."""
    warnings = condition_warnings(table, state.answers, session.settings.defaults, spec)
    return [f"Warning: {warning}" for warning in warnings]


def schema_filter(ctx: StepContext, message: str) -> Nav:
    """Ask the ``--schema`` filter: every schema, a schema name or a file."""
    state = ctx.state
    chosen: dict[str, Any] = {}

    def pick() -> Nav:
        current = state.value("schema")
        names = list(schema_names())
        if current is not None and current not in names:
            names.insert(0, current)
        choices: list[Choice[str | None]] = [Choice(None, "Every schema")]
        choices += [Choice(name, name) for name in names]
        choices.append(Choice(OTHER, "Another schema name or file..."))
        answer = ctx.prompter.select(message, choices, default=current)
        if answer is BACK:
            return BACK
        chosen["schema"] = answer
        if answer is None:
            state.clear("schema")
        elif answer != OTHER:
            state.set("schema", answer)
        return DONE

    def typed() -> Nav | None:
        if chosen.get("schema") != OTHER:
            return None
        answer = ctx.prompter.text("Schema name or file:", validate=_named)
        if answer is BACK:
            return BACK
        text = answer.strip()
        state.set("schema", str(absolute(text)) if absolute(text).is_file() else text)
        return DONE

    return ctx.sequence([pick, typed])


def schema_text(schema: str | None) -> str:
    """Return the review text of a ``--schema`` filter."""
    return "every schema" if schema is None else f"schema {schema}"


def _ask_folder(ctx: StepContext) -> Nav:
    state = ctx.state
    answer = ctx.prompter.path(
        "Folder with batch submissions (subfolders included):",
        default=str(absolute(state.value("input"))),
        validate=_existing_folder,
        only_directories=True,
    )
    if answer is BACK:
        return BACK
    folder = absolute(answer)
    state.set("input", folder)
    ctx.prompter.show(f"  found {plural(_databases(folder), 'database')}")
    return DONE


def _ask_batch_ids(ctx: StepContext) -> Nav:
    return ctx.text_option(
        "batch_ids",
        "Batch IDs to finalize (comma-separated; empty = every batch):",
        empty=(),
    )


def _unresolved_action(state: State) -> str:
    if state.value("adopt") is not None:
        return _ADOPT
    if state.value("abandon") is not None:
        return _ABANDON
    return _REPORT


def _ask_unresolved(ctx: StepContext) -> Nav:
    state = ctx.state
    chosen: dict[str, str] = {}

    def action() -> Nav:
        answer = ctx.prompter.select(
            "Unresolved submissions:",
            [
                Choice(_REPORT, "Report them", "with their candidate batches"),
                Choice(_ADOPT, "Adopt a provider batch ID", "then finalize it"),
                Choice(_ABANDON, "Abandon a submission ID", "its chunks go pending"),
            ],
            default=_unresolved_action(state),
        )
        if answer is BACK:
            return BACK
        chosen["action"] = answer
        if answer == _REPORT:
            state.clear("adopt")
            state.clear("abandon")
        return DONE

    def target() -> Nav | None:
        name = chosen.get("action")
        if name not in (_ADOPT, _ABANDON):
            return None
        message = "Provider batch ID:" if name == _ADOPT else "Submission ID:"
        current = state.value(name)
        answer = ctx.prompter.text(message, default=current or "", validate=_given)
        if answer is BACK:
            return BACK
        state.set(name, answer.strip())
        return DONE

    return ctx.sequence([action, target])


_RUN = "run"
_CHOOSE = "choose"


def _ask_formats(ctx: StepContext) -> Nav:
    """Ask whether to render as the submitting run did, else which formats."""
    state = ctx.state
    chosen: dict[str, str] = {}

    def mode() -> Nav:
        answer = ctx.prompter.select(
            "Formats:",
            [
                Choice(_RUN, "As the submitting run", "each document's run formats"),
                Choice(_CHOOSE, "Choose formats"),
            ],
            default=_RUN if state.value("formats") is None else _CHOOSE,
        )
        if answer is BACK:
            return BACK
        chosen["mode"] = answer
        if answer == _RUN:
            state.clear("formats")
        return DONE

    def formats() -> Nav | None:
        if chosen.get("mode") != _CHOOSE:
            return None
        answer = ctx.prompter.checkbox(
            "Formats besides _output.json (none = only _output.json):",
            [Choice(name, name) for name in OUTPUT_FORMATS],
            checked=list(state.value("formats") or ()),
        )
        if answer is BACK:
            return BACK
        state.set("formats", tuple(answer))
        return DONE

    return ctx.sequence([mode, formats])


def _check_steps(session: Session) -> list[Step]:
    return [
        Step("folder", "Folder", ask=_ask_folder, sets=("input",)),
        Step(
            "schema",
            "Schema",
            ask=lambda ctx: schema_filter(ctx, "Check the documents of:"),
            sets=("schema",),
        ),
        Step("batches", "Batches", ask=_ask_batch_ids, sets=("batch_ids",)),
        Step(
            "unresolved", "Unresolved", ask=_ask_unresolved, sets=("adopt", "abandon")
        ),
        Step("formats", "Formats", ask=_ask_formats, sets=("formats",)),
    ]


def _resolve_check(state: State, session: Session) -> Resolution[BatchCheckSpec]:
    return resolve_batch_check(session.settings, state.answers)


def _unresolved_text(spec: BatchCheckSpec) -> str:
    if spec.adopt is not None:
        return f"adopt batch {spec.adopt}"
    if spec.abandon is not None:
        return f"abandon submission {spec.abandon}"
    return "report them"


def _formats_text(formats: tuple[str, ...] | None) -> str:
    if formats is None:
        return "the formats of each submitting run"
    return " + ".join(("_output.json", *formats))


def _check_review(state: State, session: Session) -> Review:
    try:
        spec = _resolve_check(state, session).spec
    except ValueError as exc:
        return Review([], withheld=str(exc))
    folder = absolute(spec.directory)
    count = _databases(folder)
    formats = _formats_text(spec.formats)
    lines = [
        Line("folder", [f"{folder}  ({plural(count, 'database')})"]),
        Line("schema", [schema_text(spec.schema)]),
        Line("batches", [", ".join(spec.batch_ids) or "every batch"]),
        Line("unresolved", [_unresolved_text(spec)]),
        Line("formats", [formats]),
    ]
    if not folder.is_dir():
        return Review(lines, withheld=f"Input folder does not exist: {folder}")
    notes = [
        f"Checks the open batch submissions of {plural(count, 'database')} below "
        f"{folder}; finished batches are finalized and rendered ({formats})."
    ]
    if spec.batch_ids:
        notes.append(
            "Only the named batches are finalized; unresolved submissions are "
            "reconciled in every check."
        )
    notes += condition_notes(BATCH_CHECK_OPTIONS, state, session, spec)
    return Review(lines, notes=notes)


def _check_command(state: State, session: Session) -> list[str]:
    spec = _resolve_check(state, session).spec
    argv = to_argv(spec, session.settings.defaults)
    return command_words(CHECK_PROGRAM, argv, session, BATCH_CHECK_OPTIONS)


def _run_check(request: RunRequest) -> int:
    settings = request.session.settings
    flags: dict[str, Any] = dict(request.answers)
    try:
        spec = resolve_batch_check(settings, flags).spec
    except ValueError as exc:
        _error(request, f"Error: {exc}")
        return int(ExitCode.USAGE)
    directory = spec.directory.expanduser().resolve()
    if not _is_folder(request, directory):
        return int(ExitCode.USAGE)
    if not _startup(request, spec):
        return int(ExitCode.USAGE)
    request.console.print(Text(f"Scanning '{directory}' for batch submissions..."))
    try:
        result = asyncio.run(
            check_submissions(
                directory,
                settings,
                formats=spec.formats,
                batch_ids=spec.batch_ids,
                adopt=spec.adopt,
                abandon=spec.abandon,
                schema=spec.schema,
                args=request.args,
            )
        )
    except CheckTargetError as exc:
        _error(request, f"Error: {exc}")
        return int(ExitCode.USAGE)
    request.console.print(check_result(result))
    if result.refused:
        return int(ExitCode.USAGE)
    return int(ExitCode.FAILURE if result.had_failure else ExitCode.OK)


def check_flow() -> TaskFlow:
    """Return the batch check task."""
    return TaskFlow(
        name="batch-check",
        title=TASKS["batch-check"],
        program=CHECK_PROGRAM,
        table=BATCH_CHECK_OPTIONS,
        steps=_check_steps,
        review=_check_review,
        command=_check_command,
        resolve=_resolve_check,
        run=_run_check,
    )


def _cancel_target(state: State) -> str:
    if state.value("all"):
        return _ALL
    if state.value("input") is not None:
        return _DIR
    return _IDS


def _ask_target(ctx: StepContext) -> Nav:
    state = ctx.state
    chosen: dict[str, str] = {}

    def target() -> Nav:
        answer = ctx.prompter.select(
            "Cancel:",
            [
                Choice(_IDS, "Named batch IDs"),
                Choice(_ALL, "Every unfinished batch", "of every provider"),
                Choice(_DIR, "The open submissions below a folder"),
            ],
            default=_cancel_target(state),
        )
        if answer is BACK:
            return BACK
        chosen["target"] = answer
        if answer == _ALL:
            state.set("all", True)
        return DONE

    def batch_ids() -> Nav | None:
        if chosen.get("target") != _IDS:
            return None
        option = state.table["batch_ids"]
        current = ",".join(state.value("batch_ids"))

        def validate(text: str) -> str | None:
            if not text.strip(" ,"):
                return "enter at least one batch ID"
            return None

        answer = ctx.prompter.text(
            "Batch IDs (comma-separated):", default=current, validate=validate
        )
        if answer is BACK:
            return BACK
        state.set("batch_ids", option.convert(answer.strip()))
        return DONE

    def folder() -> Nav | None:
        if chosen.get("target") != _DIR:
            return None
        current = state.value("input")
        answer = ctx.prompter.path(
            "Folder with the open submissions (subfolders included):",
            default="" if current is None else str(current),
            validate=_existing_folder,
            only_directories=True,
        )
        if answer is BACK:
            return BACK
        path = absolute(answer)
        state.set("input", path)
        ctx.prompter.show(f"  found {plural(_databases(path), 'database')}")
        return DONE

    return ctx.sequence([target, batch_ids, folder])


def _cancel_steps(session: Session) -> list[Step]:
    return [
        Step(
            "target",
            "What to cancel",
            ask=_ask_target,
            sets=("batch_ids", "all", "input"),
        )
    ]


def _resolve_cancel(state: State, session: Session) -> Resolution[BatchCancelSpec]:
    return resolve_batch_cancel(session.settings, state.answers)


def _cancel_review(state: State, session: Session) -> Review:
    try:
        spec = _resolve_cancel(state, session).spec
    except ValueError as exc:
        return Review([], withheld=str(exc))
    if spec.directory is not None:
        folder = absolute(spec.directory)
        count = _databases(folder)
        line = f"the open submissions below {folder}  ({plural(count, 'database')})"
        note = (
            f"Cancels the submitted batches recorded in {plural(count, 'database')} "
            f"below {folder}; their chunks return to pending."
        )
        withheld = None if folder.is_dir() else f"Input folder does not exist: {folder}"
    elif spec.batch_ids:
        line = ", ".join(spec.batch_ids)
        note = (
            f"Cancels {plural(len(spec.batch_ids), 'batch', 'batches')} with the "
            "provider each ID's shape names."
        )
        withheld = None
    else:
        line = "every unfinished batch of every provider"
        note = (
            "Lists the unfinished batches of every provider whose key variable is "
            "set and cancels each."
        )
        withheld = None
    notes = [note, *condition_notes(BATCH_CANCEL_OPTIONS, state, session, spec)]
    return Review([Line("target", [line])], notes=notes, withheld=withheld)


def _cancel_command(state: State, session: Session) -> list[str]:
    spec = _resolve_cancel(state, session).spec
    argv = to_argv(spec, session.settings.defaults)
    return command_words(CANCEL_PROGRAM, argv, session, BATCH_CANCEL_OPTIONS)


def _cancel(
    request: RunRequest, spec: BatchCancelSpec, directory: Path | None
) -> Coroutine[Any, Any, CancelReport]:
    settings = request.session.settings
    console = request.console
    args = request.args
    if directory is not None:
        console.print(
            Text(f"Cancelling the open batch submissions below '{directory}'...")
        )
        return cancel_submissions(directory, settings, args=args)
    if spec.batch_ids:
        console.print(f"Targeting {len(spec.batch_ids)} specific batch(es)...")
        return cancel_batches(spec.batch_ids, settings, args=args)
    console.print("Listing the unfinished batches of every provider...")
    return cancel_unfinished(settings, args=args)


def _run_cancel(request: RunRequest) -> int:
    settings = request.session.settings
    flags: dict[str, Any] = dict(request.answers)
    try:
        spec = resolve_batch_cancel(settings, flags).spec
    except ValueError as exc:
        _error(request, f"Error: {exc}")
        return int(ExitCode.USAGE)
    directory = None
    if spec.directory is not None:
        directory = spec.directory.expanduser().resolve()
        if not _is_folder(request, directory):
            return int(ExitCode.USAGE)
    if not _startup(request, spec):
        return int(ExitCode.USAGE)
    report = asyncio.run(_cancel(request, spec, directory))
    request.console.print(cancel_result(report))
    if report.refused:
        return int(ExitCode.USAGE)
    return int(exit_code_for(failed=report.failed))


def cancel_flow() -> TaskFlow:
    """Return the batch cancel task."""
    return TaskFlow(
        name="batch-cancel",
        title=TASKS["batch-cancel"],
        program=CANCEL_PROGRAM,
        table=BATCH_CANCEL_OPTIONS,
        steps=_cancel_steps,
        review=_cancel_review,
        command=_cancel_command,
        resolve=_resolve_cancel,
        run=_run_cancel,
    )
