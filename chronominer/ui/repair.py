"""The repair task of the guided UI: choose the failed chunks, repair them.

The steps follow the repair options: the documents (input, schema filter and
chunk indices), the model, the response format, the page images (asked only
when the repaired chunks are image or PDF pages) and the run mode. The
review previews the plan (the chunks to repair and its warnings) and
withholds Run when the repair cannot be planned; a dry run plans without the
key variable. The run mirrors ``chronominer repair``: plan, the extension's
startup on dry runs and when chunks are to repair, then the repair with rich
progress.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, Final

from rich.text import Text

from chronominer import ext
from chronominer.batch.repair import (
    RepairPlan,
    RepairPlanError,
    batch_refusal,
    plan_repair,
    run_repair,
)
from chronominer.common.batch import supports_batch
from chronominer.common.contract import ExitCode, exit_code_for
from chronominer.common.options import Source
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
from chronominer.settings import SettingsError, resolve_repair
from chronominer.spec import REPAIR_OPTIONS, RepairSpec, Resolution, to_argv
from chronominer.ui.batch import condition_notes, schema_filter, schema_text
from chronominer.ui.progress import RichProgress
from chronominer.ui.questions import (
    IMAGE_OPTIONS,
    MODEL_OPTIONS,
    ImageQuestions,
    ModelQuestions,
    Question,
    ResponseQuestions,
    images_line,
    model_line,
    resolution_of,
    support_of,
)
from chronominer.ui.results import repair_result
from chronominer.ui.session import (
    TASKS,
    RunRequest,
    Session,
    TaskFlow,
    absolute,
    command_words,
    plural,
)

__all__ = ["PROGRAM", "flow", "repairs_pages"]

PROGRAM: Final = ("chronominer", "repair")
_DOCUMENT_OPTIONS: Final = ("input", "schema", "indices")
_RUN_OPTIONS: Final = ("batch", "dry_run")
_PAGES: Final = "repair_pages"
_FORMAT_TEXTS: Final = {
    "schema": "schema (API-enforced)",
    "json": "json (prompted, validated)",
}


def _existing(text: str) -> str | None:
    if not text.strip():
        return "enter a path"
    if not absolute(text).exists():
        return "no such file or folder"
    return None


def _ask_documents(ctx: StepContext) -> Nav:
    state = ctx.state

    def path() -> Nav:
        answer = ctx.prompter.path(
            "Folder, chronominer.sqlite or file whose failed chunks to repair:",
            default=str(absolute(state.value("input"))),
            validate=_existing,
        )
        if answer is BACK:
            return BACK
        state.set("input", absolute(answer))
        return DONE

    def schema() -> Nav:
        return schema_filter(ctx, "Repair the documents of:")

    def indices() -> Nav:
        return ctx.text_option(
            "indices",
            "Chunk indices among the failed ones (0,5,12 or 3-7; empty = every one):",
            empty=None,
        )

    return ctx.sequence([path, schema, indices])


def _resolve(state: State, session: Session) -> Resolution[RepairSpec]:
    return resolve_repair(session.settings, state.answers)


def repairs_pages(state: State, session: Session) -> bool:
    """Whether the chunks the answers select are image or PDF pages.

    True also when the repair cannot be planned, so the page image options
    stay reachable. The answer is kept in ``state.data`` per spec.
    """
    try:
        spec = _resolve(state, session).spec
    except ValueError:
        return True
    probe = replace(spec, run=replace(spec.run, batch=False, dry_run=True))
    cached = state.data.get(_PAGES)
    if isinstance(cached, tuple) and cached[0] == probe:
        return bool(cached[1])
    try:
        plan = plan_repair(probe, session.settings, args=session.args)
    except RepairPlanError:
        pages = True
    else:
        pages = any(document.pages for document in plan.documents)
    state.data[_PAGES] = (probe, pages)
    return pages


def _batch_available(state: State, session: Session) -> bool:
    try:
        spec = resolution_of(state, session).spec
    except ValueError:
        return False
    return (
        supports_batch(spec.model.provider)
        and batch_refusal(spec, settings=session.settings, args=session.args) is None
    )


def _ask_run(ctx: StepContext, session: Session) -> Nav:
    state = ctx.state

    def mode() -> Nav | None:
        if not _batch_available(state, session):
            state.clear("batch")
            return None
        return ctx.select_option(
            "batch",
            "Run mode:",
            [
                Choice(False, "Synchronous", "the chunks are sent now"),
                Choice(True, "Batch", "through the provider's batch API"),
            ],
        )

    def dry_run() -> Nav:
        return ctx.select_option(
            "dry_run",
            "Dry run:",
            [
                Choice(False, "No, repair the chunks"),
                Choice(True, "Yes, only report what would be repaired"),
            ],
        )

    questions: list[Question] = [mode, dry_run]
    return ctx.sequence(questions)


def _steps(session: Session) -> list[Step]:
    return [
        Step("documents", "Documents", ask=_ask_documents, sets=_DOCUMENT_OPTIONS),
        Step(
            "model",
            "Model",
            ask=lambda ctx: ctx.sequence(ModelQuestions(ctx, session).questions()),
            sets=MODEL_OPTIONS,
        ),
        Step(
            "response",
            "Response",
            ask=lambda ctx: ctx.sequence(
                [ResponseQuestions(ctx, session).response_format]
            ),
            sets=("response_format",),
        ),
        Step(
            "images",
            "Page images",
            ask=lambda ctx: ctx.sequence(ImageQuestions(ctx, session).questions()),
            sets=IMAGE_OPTIONS,
            applies=lambda state: repairs_pages(state, session),
        ),
        Step(
            "run",
            "Run options",
            ask=lambda ctx: _ask_run(ctx, session),
            sets=_RUN_OPTIONS,
        ),
    ]


def _documents_line(spec: RepairSpec) -> Line:
    if spec.indices is None:
        chunks = "every failed chunk"
    else:
        chunks = "indices " + ",".join(str(index) for index in spec.indices)
    return Line(
        "documents", [str(absolute(spec.input)), schema_text(spec.schema), chunks]
    )


def _response_line(
    spec: RepairSpec, sources: dict[str, Source], session: Session
) -> Line:
    chosen = spec.model.response_format
    if chosen is None:
        automatic = support_of(spec.model, session.settings).automatic_format
        text = f"automatic ({automatic})"
    else:
        text = _FORMAT_TEXTS.get(chosen, chosen)
    settings = sources.get("model.response_format") is Source.SETTINGS
    return Line("response", [Part(text, settings)])


def _run_line(spec: RepairSpec) -> Line:
    parts = ["batch" if spec.run.batch else "sync"]
    if spec.run.dry_run:
        parts.append("dry run")
    return Line("run", parts)


def _plan_notes(plan: RepairPlan) -> list[str]:
    notes: list[str] = []
    if plan.message is not None:
        notes.append(plan.message)
    elif plan.documents:
        notes.append(
            f"Chunks to repair: {len(plan.units)} in "
            f"{plural(len(plan.documents), 'document')}"
        )
        for document in plan.documents:
            indices = ", ".join(str(index) for index in document.failures)
            notes.append(
                f"  '{document.name}' in {document.database}: indices {indices}"
            )
    notes += [f"Warning: {warning}" for warning in plan.warnings]
    return notes


def _review(state: State, session: Session) -> Review:
    try:
        resolution = _resolve(state, session)
    except ValueError as exc:
        return Review([], withheld=str(exc))
    spec, sources = resolution.spec, dict(resolution.sources)
    lines = [
        _documents_line(spec),
        model_line(spec.model, sources, session),
        _response_line(spec, sources, session),
    ]
    if repairs_pages(state, session):
        lines.append(images_line(spec.model, spec.images, sources))
    lines.append(_run_line(spec))
    conditions = condition_notes(REPAIR_OPTIONS, state, session, spec)
    try:
        plan = plan_repair(spec, session.settings, args=session.args)
    except RepairPlanError as exc:
        return Review(lines, notes=conditions, withheld=str(exc))
    return Review(lines, notes=[*_plan_notes(plan), *conditions])


def _command(state: State, session: Session) -> list[str]:
    spec = _resolve(state, session).spec
    argv = to_argv(spec, session.settings.defaults)
    return command_words(
        PROGRAM, argv, session, REPAIR_OPTIONS, provider=spec.model.provider
    )


def _error(request: RunRequest, message: str) -> None:
    request.console.print(Text(message, style="red"), soft_wrap=True)


def _run(request: RunRequest) -> int:
    settings = request.session.settings
    flags: dict[str, Any] = dict(request.answers)
    try:
        spec = resolve_repair(settings, flags).spec
    except ValueError as exc:
        _error(request, f"Error: {exc}")
        return int(ExitCode.USAGE)
    try:
        plan = plan_repair(spec, settings, args=request.args)
    except RepairPlanError as exc:
        _error(request, f"Error: {exc}")
        return int(exc.exit_code)
    for warning in plan.warnings:
        request.console.print(Text(f"Warning: {warning}", style="yellow"))
    if plan.dry_run or plan.to_repair:
        try:
            ext.startup(spec, settings=settings, args=request.args)
        except SettingsError as exc:
            _error(request, f"Error: {exc}")
            return int(ExitCode.USAGE)
    if plan.dry_run or not plan.to_repair:
        result = plan.outcome()
    else:
        with RichProgress(request.console) as progress:
            result = asyncio.run(run_repair(plan, progress))
    request.console.print(repair_result(spec, result))
    return int(exit_code_for(failed=result.failed))


def flow() -> TaskFlow:
    """Return the repair task."""
    return TaskFlow(
        name="repair",
        title=TASKS["repair"],
        program=PROGRAM,
        table=REPAIR_OPTIONS,
        steps=_steps,
        review=_review,
        command=_command,
        resolve=_resolve,
        run=_run,
    )
