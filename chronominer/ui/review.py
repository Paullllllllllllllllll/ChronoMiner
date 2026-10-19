"""Review screen of a new extraction: summary lines, plan preview, command.

The review resolves the answers like ``chronominer extract`` would, marks
values taken from the settings defaults, lists the condition warnings the
CLI would print and previews the plan with :func:`chronominer.run.plan` (no
API calls, no writes). Run is withheld while the plan fails (it checks the
input and its selection, the schema, the batch API and the model's image
input) and, unless the run is a dry run, while a key variable it needs is
unset; the review names each such variable. The equivalent command spells
out every value that differs from the code default.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

from chronominer import ext
from chronominer import run as core
from chronominer.common.options import Source
from chronominer.common.wizard import (
    Line,
    Part,
    Review,
    ReviewAction,
    State,
    access_line,
)
from chronominer.discovery import DATABASE_NAME
from chronominer.spec import (
    OPTIONS,
    RunSpec,
    SpecError,
    condition_warnings,
    resolve_schema,
    to_argv,
)
from chronominer.ui.questions import (
    images_line,
    model_line,
    resolution_of,
    response_line,
)
from chronominer.ui.session import (
    Session,
    absolute,
    command_words,
    inventory_of,
    plural,
)
from chronominer.ui.wizard import run_types, unit_word

__all__ = [
    "FORCE_ACTION",
    "PLAN_LIMIT",
    "PROGRAM",
    "build_review",
    "equivalent_command",
    "missing_key_notes",
    "plan_notes",
]

PROGRAM: Final = ("chronominer", "extract")
FORCE_ACTION: Final = "force"
PLAN_LIMIT: Final = 20

_Mark = Callable[[str, str], Part]

_STATE_TEXTS = {"none": "new"}
_IMAGE_TYPES = frozenset({"image", "pdf"})
_CHUNKED_TYPES = frozenset({"text", "pdf"})


def equivalent_command(state: State, session: Session) -> list[str]:
    """Return the words of the ``chronominer extract`` command.

    Raises:
        ValueError: The answers do not resolve, or a settings default cannot
            be undone on the command line.
    """
    spec = resolution_of(state, session).spec
    argv = to_argv(spec, session.settings.defaults)
    return command_words(PROGRAM, argv, session, OPTIONS, provider=spec.model.provider)


def _input_line(state: State, spec: RunSpec, mark: _Mark) -> Line:
    path = absolute(spec.input.path)
    inventory = inventory_of(state, path)
    parts = [
        Part(f"{path}  ({inventory.describe()})"),
        mark("input.type", f"type {spec.input.type}"),
    ]
    if spec.input.files:
        parts.append(Part(f"files {', '.join(spec.input.files)}"))
    elif spec.input.select is not None:
        parts.append(mark("input.select", f"items {spec.input.select}"))
    elif spec.input.all:
        parts.append(mark("input.all", "all items"))
    if spec.input.recursive:
        parts.append(mark("input.recursive", "recursive"))
    if run_types(state) & _CHUNKED_TYPES:
        unit = unit_word(state)
        pages = spec.input.pages
        parts.append(Part(f"all {unit}" if pages is None else f"{unit} {pages}"))
    return Line("input", parts)


def _output_line(spec: RunSpec, mark: _Mark) -> Line:
    output = spec.output
    if output.directory is None:
        parts = [mark("output.directory", "beside input")]
    else:
        parts = [
            mark("output.directory", str(absolute(output.directory))),
            mark("output.layout", f"layout {output.layout}"),
        ]
    formats = ", ".join(("_output.json", *output.formats))
    parts.append(mark("output.formats", f"formats {formats} (+ {DATABASE_NAME})"))
    return Line("output", parts)


def _schema_line(spec: RunSpec, mark: _Mark) -> Line:
    argument = spec.prompt.schema
    try:
        schema = resolve_schema(argument)
    except SpecError:
        return Line("schema", [mark("prompt.schema", argument)])
    return Line("schema", [mark("prompt.schema", schema.name), Part(str(schema.path))])


def _chunking_line(spec: RunSpec, mark: _Mark) -> Line:
    chunking = spec.chunking
    parts = [mark("chunking.mode", chunking.mode)]
    if chunking.mode == "auto":
        parts.append(
            mark("chunking.chunk_tokens", f"{chunking.chunk_tokens} tokens per chunk")
        )
    else:
        parts.append(Part("from _line_ranges.txt"))
    if chunking.mode == "adjust":
        parts.append(
            mark(
                "chunking.context_window",
                f"{plural(chunking.context_window, 'line')} around each boundary",
            )
        )
    return Line("chunking", parts)


def _access_line(spec: RunSpec, session: Session, adjust: bool) -> Line | None:
    """Return how the model calls of the run authenticate.

    The extraction calls use a batch client in a batch run; the boundary
    decisions of *adjust* chunking are synchronous calls.
    """
    model = spec.model

    def describe(kind: str) -> Callable[[], str]:
        return lambda: ext.describe_access(
            model.provider,
            model.endpoint,
            kind=kind,
            settings=session.settings,
            args=session.args,
        )

    routes = [("extract", describe("batch" if spec.run.batch else "chat"))]
    if adjust:
        routes.append(("adjust", describe("chat")))
    return access_line(routes)


def _run_line(spec: RunSpec, mark: _Mark) -> Line:
    run = spec.run
    parts = [mark("run.batch", "batch" if run.batch else "sync")]
    parts.append(mark("run.force", "force" if run.force else "resume"))
    if not run.batch:
        parts.append(mark("run.concurrency", f"concurrency {run.concurrency}"))
    if run.dry_run:
        parts.append(mark("run.dry_run", "dry run"))
    return Line("run", parts)


def plan_notes(plan: core.RunPlan) -> list[str]:
    """Return the plan preview: counts, one line per item, warnings."""
    left_out = (*plan.unstored, *plan.blocked, *plan.refused)
    head = f"Plan: {len(plan.to_process)} to process, {len(plan.skipped)} complete"
    if left_out:
        head += f", {len(left_out)} left out"
    entries = [
        f"  {item.name}: {_STATE_TEXTS.get(item.state, item.state)} ({item.type})"
        for item in plan.to_process
    ]
    entries += [
        f"  {item.name}: complete "
        f"({'rendered again' if item.resume.render else 'skipped'})"
        for item in plan.skipped
    ]
    entries += [f"  {item.name}: {item.state} ({item.reason})" for item in left_out]
    lines = [head, *entries[:PLAN_LIMIT]]
    if len(entries) > PLAN_LIMIT:
        lines.append(f"  ... and {len(entries) - PLAN_LIMIT} more")
    lines += [f"Warning: {warning}" for warning in plan.warnings]
    return lines


def _force(state: State) -> None:
    state.set("force", True)


def build_review(state: State, session: Session) -> Review:
    """Return the review screen of the answers in *state*."""
    try:
        resolved = resolution_of(state, session)
    except ValueError as exc:
        return Review(lines=(), withheld=str(exc))
    spec, sources = resolved.spec, resolved.sources

    def mark(target: str, text: str) -> Part:
        return Part(text, sources.get(target) is Source.SETTINGS)

    types = run_types(state)
    lines = [
        _input_line(state, spec, mark),
        _output_line(spec, mark),
        _schema_line(spec, mark),
    ]
    if "text" in types:
        lines.append(_chunking_line(spec, mark))
    lines.append(model_line(spec.model, sources, session))
    adjust = "text" in types and spec.chunking.mode == "adjust"
    access = _access_line(spec, session, adjust)
    if access is not None:
        lines.append(access)
    response = response_line(spec.model, spec.prompt, sources, session)
    lines.append(
        Line(
            response.label,
            [
                part
                for part in response.parts
                if not (isinstance(part, Part) and part.text == spec.prompt.schema)
            ],
        )
    )
    if types & _IMAGE_TYPES:
        lines.append(images_line(spec.model, spec.images, sources))
    lines.append(_run_line(spec, mark))

    notes: list[str] = []
    actions: list[ReviewAction] = []
    withheld: list[str] = []
    try:
        plan = core.plan(
            spec,
            session.settings,
            sources=sources,
            args=session.args,
            require_keys=False,
        )
    except core.PlanError as exc:
        notes.append(f"Plan error: {exc}")
        withheld.append("the plan failed; edit a step")
    else:
        notes += plan_notes(plan)
        if plan.skipped and not spec.run.force:
            actions.append(
                ReviewAction(
                    FORCE_ACTION,
                    f"Extract {plural(len(plan.skipped), 'complete item')} "
                    "again (--force)",
                    _force,
                )
            )
        if not spec.run.dry_run:
            key_notes, key_withheld = missing_key_notes(plan.missing_keys)
            notes += key_notes
            withheld += key_withheld
    warnings = condition_warnings(
        OPTIONS, state.answers, session.settings.defaults, spec
    )
    notes += [f"Warning: {warning}" for warning in warnings]
    if spec.run.dry_run:
        notes.append("Dry run: Run shows the plan only, with no API calls or writes.")
    return Review(
        lines=lines,
        notes=notes,
        actions=actions,
        withheld="; ".join(withheld) if withheld else None,
    )


def missing_key_notes(
    missing: Sequence[core.MissingKey],
) -> tuple[list[str], list[str]]:
    """Return the review notes and the withheld reason of unset key variables.

    One note names each variable with the calls that need it; Run is
    withheld until they are set.
    """
    needs: dict[str, list[str]] = {}
    for key in missing:
        needs.setdefault(key.variable, []).append(key.need)
    notes = [
        f"Missing API key: {variable} is not set ({', '.join(uses)})"
        for variable, uses in needs.items()
    ]
    withheld = [f"set {', '.join(needs)} in the environment"] if needs else []
    return notes, withheld
