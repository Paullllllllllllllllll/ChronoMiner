"""Core entry points: discover items, plan a run, run it.

:func:`plan` turns a resolved :class:`~chronominer.spec.RunSpec` and the
machine settings into a :class:`RunPlan` without API calls or writes: it
finds the text, image and PDF inputs, keeps the items ``--select``,
``--all`` or ``--files`` chooses (a folder of several items needs one of
them), classifies resume from the store of each output root, resolves the
schema and the prompt context of every item, and checks the key variable of
every provider the run needs, the provider's batch API for ``--batch`` and
the model's image input for image and PDF items. A dry run plans without
key variables and lists the unset ones as warnings. :func:`run` extracts
the plan's items on one event loop, each as the chunks of one store
document, and reports through
:class:`~chronominer.events.RunEvents`. :func:`plan_ranges` and
:func:`run_ranges` do the same for ``ranges generate`` and ``ranges
adjust``. None of them prints, prompts or exits: a plan that cannot run
raises a :class:`PlanError` that carries its exit code.

Every output root (the output folder, or each input's folder when writing
beside the input) holds one ``chronominer.sqlite``, the store of the run's
documents. Resume reads only the store: an item whose ``_output.json``
exists without units in the store is ``unstored`` and skipped, an item leased
by another live run or waiting for a batch is ``blocked`` and skipped, and
an item whose document records another input content, whose stored pages
used other image settings, or whose done chunks used another schema, is
``refused``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import logging
import os
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from chronominer import ext
from chronominer.batch import submit as batch_submit
from chronominer.batch.repair import batch_refusal
from chronominer.common.batch import supports_batch
from chronominer.common.context import ContextError
from chronominer.common.contract import ExitCode, exit_code_for
from chronominer.common.decisions import dialect_for
from chronominer.common.options import Source
from chronominer.common.selection import PageRange, parse_page_range, select_items
from chronominer.common.spec_view import resolved_view
from chronominer.common.sqlite import DEFAULT_BUSY_TIMEOUT, DEFAULT_WARN_INTERVAL
from chronominer.common.store import (
    Lease,
    LeaseKeeper,
    LeaseLostError,
    Store,
    StoreError,
    UnitRow,
    WriterOptions,
    file_sha256,
)
from chronominer.common.structured import ResponseFormat, fit_response_format
from chronominer.common.usage import Usage
from chronominer.discovery import (
    DiscoveryError,
    SourceItem,
    discover_items,
    item_paths,
)
from chronominer.events import ItemFinished, ItemStarted, NullEvents, RunEvents
from chronominer.extract import (
    BatchSubmitter,
    ExtractionMethod,
    Item,
    Job,
    NothingSelectedError,
    SubmitOutcome,
)
from chronominer.extract.images import resolve_image_settings, resume_images
from chronominer.extract.llm import Extractor, image_record
from chronominer.extract.units import DocumentHandle
from chronominer.llm.caller import CallEnv, ModelCaller
from chronominer.llm.capabilities import ensure_image_support
from chronominer.llm.options import (
    model_capabilities,
    output_limit,
    response_format_for,
)
from chronominer.llm.prompts import (
    IMAGE_PROMPT_PATH,
    TEXT_PROMPT_PATH,
    extraction_schema,
    item_context,
    item_context_image,
)
from chronominer.output.render import format_path, output_formats, render_item
from chronominer.output.store import (
    Classification,
    ItemState,
    classify,
    database_path,
    document_name,
    lease_reason,
    open_store,
)
from chronominer.ranges.adjust import ROLE as ADJUST_ROLE
from chronominer.ranges.adjust import AdjustOptions, adjust_ranges
from chronominer.ranges.generate import GenerateReport, generate_files
from chronominer.ranges.lines import ranges_path
from chronominer.ranges.store import group_by_folder, open_folder_store
from chronominer.settings import Settings, SettingsError
from chronominer.spec import (
    InputSpec,
    ModelSpec,
    RangesAdjustSpec,
    RangesGenerateSpec,
    ResolvedSchema,
    RunSpec,
    SpecError,
    effective_spec,
    resolve_schema,
)

__all__ = [
    "BUSY_TIMEOUT",
    "BUSY_WARN_INTERVAL",
    "ConfigurationError",
    "InputError",
    "ItemResult",
    "MissingKey",
    "NoItemsError",
    "PlanError",
    "PlannedItem",
    "RangesPlan",
    "RunPlan",
    "RunResult",
    "StoreRefusedError",
    "discover",
    "discover_inputs",
    "item_names",
    "plan",
    "plan_ranges",
    "run",
    "run_ranges",
    "writer_options",
]

logger = logging.getLogger(__name__)

_IMAGE_TYPES = ("image", "pdf")
"""Input types whose chunks are page images."""

BUSY_TIMEOUT: float = DEFAULT_BUSY_TIMEOUT
"""Seconds a store write or read waits on a busy database before it backs off."""

BUSY_WARN_INTERVAL: float = DEFAULT_WARN_INTERVAL
"""Seconds between two warnings about the same busy database."""


class PlanError(Exception):
    """The run cannot be planned.

    *items_total* is the number of items the run would have processed, for
    the run summary; ``exit_code`` is the process exit code of the error.
    """

    exit_code: ExitCode = ExitCode.USAGE

    def __init__(self, message: str, *, items_total: int = 0) -> None:
        super().__init__(message)
        self.items_total = items_total


class ConfigurationError(PlanError):
    """The spec and settings cannot serve the run (exit 2)."""


class StoreRefusedError(ConfigurationError):
    """An output root's database is versionless, newer or unreadable (exit 2).

    The database is left as it is.
    """


class InputError(PlanError):
    """The input or output path cannot serve the run (exit 2)."""


class NoItemsError(PlanError):
    """The input holds nothing to process, or ``--select`` matches nothing (exit 2)."""


@dataclass(frozen=True)
class MissingKey:
    """An unset key variable the run needs: *variable* of *provider* for *need*."""

    variable: str
    provider: str
    need: str

    @property
    def message(self) -> str:
        """The plan error, or the dry run's warning, that names the variable."""
        return (
            f"The API key variable {self.variable} of provider {self.provider} "
            f"is not set; {self.need} needs it."
        )


@dataclass(frozen=True)
class PlannedItem:
    """One discovered input: its resume state, store row and prompt context.

    *resume* is the item's classification from the store of *root*, where
    the item is the document *document*; *output* is its ``_output.json``.
    *context* and *context_image* are the item's resolved extraction
    context; *adjust_context* the context of its boundary decisions under
    ``--chunking adjust``.
    """

    source: SourceItem
    resume: Classification
    root: Path
    document: str
    output: Path
    context: str | None = None
    context_image: Path | None = None
    adjust_context: str | None = None

    @property
    def path(self) -> Path:
        """The item's input path."""
        return self.source.path

    @property
    def name(self) -> str:
        """The item's file name."""
        return self.source.name

    @property
    def type(self) -> str:
        """The item's input type: text, image or pdf."""
        return self.source.type

    @property
    def state(self) -> str:
        """The resume state: none, partial, complete, unstored, blocked or refused."""
        return self.resume.state.value

    @property
    def reason(self) -> str:
        """Why the item has its resume state."""
        return self.resume.reason


@dataclass(frozen=True)
class RunPlan:
    """What an extract run will do.

    *output_dir* None writes beside each input. *to_process* is in discovery
    order, *skipped* holds the complete items; a complete item whose
    rendered files are missing or stale is rendered again without model
    calls. *unstored* items have an ``_output.json`` without store rows,
    *blocked* items are leased by another run or wait for a batch,
    *refused* items were stored from another input content, with other
    image settings or with another schema; none of them is processed.
    *warnings* are notes for the user. *sources* maps spec fields to where
    their values came from. *missing_keys* are the unset key variables of a
    plan made without them (a dry run, or the review of the guided UI).
    """

    spec: RunSpec
    input_path: Path
    output_dir: Path | None
    discovered: tuple[SourceItem, ...]
    to_process: tuple[PlannedItem, ...]
    skipped: tuple[PlannedItem, ...]
    schema: ResolvedSchema
    warnings: tuple[str, ...] = ()
    settings: Settings = field(default_factory=Settings)
    sources: Mapping[str, Source] = field(default_factory=dict)
    pages: PageRange | None = None
    unstored: tuple[PlannedItem, ...] = ()
    blocked: tuple[PlannedItem, ...] = ()
    refused: tuple[PlannedItem, ...] = ()
    missing_keys: tuple[MissingKey, ...] = ()

    @property
    def schema_path(self) -> Path:
        """The resolved schema file."""
        return self.schema.path

    @property
    def batch(self) -> bool:
        """Whether chunks are submitted to the provider's batch API."""
        return self.spec.run.batch

    @property
    def force(self) -> bool:
        """Whether existing outputs are reprocessed instead of resumed."""
        return self.spec.run.force

    @property
    def dry_run(self) -> bool:
        """Whether the plan is only reported."""
        return self.spec.run.dry_run

    @property
    def types(self) -> tuple[str, ...]:
        """The input types of the discovered items, in order of appearance."""
        return tuple(dict.fromkeys(item.type for item in self.discovered))

    @property
    def render_only(self) -> tuple[PlannedItem, ...]:
        """The complete items whose rendered files are rendered again."""
        return tuple(item for item in self.skipped if item.resume.render)

    @property
    def not_processed(self) -> int:
        """The number of items skipped as complete, unstored or blocked."""
        return len(self.skipped) + len(self.unstored) + len(self.blocked)

    @property
    def leased(self) -> tuple[PlannedItem, ...]:
        """The blocked items another run's lease holds, not a batch."""
        return tuple(item for item in self.blocked if item.resume.leased_by)

    def effective_spec(self) -> dict[str, dict[str, Any]]:
        """Return every field's value and source as the store records them.

        The mapping has the shape ``{field: {"value": ..., "source": ...}}``.
        """
        return effective_spec(self.spec, self.sources)

    def resolved_spec(self) -> dict[str, dict[str, Any]]:
        """Return :meth:`effective_spec` as the dry run shows it.

        The unset output limit and response format, and for image inputs
        the unset image options, carry the values the requests use; a
        provider other than OpenAI shows no service tier
        (:func:`~chronominer.common.spec_view.resolved_view`).
        """
        view = self.effective_spec()
        model = self.spec.model
        try:
            caps = model_capabilities(model, self.settings)
            images: dict[str, Any] | None = None
            if set(self.types) - {"text"}:
                images, _family, _section = resolve_image_settings(
                    model.provider,
                    model.model,
                    self.spec.images,
                    self.settings.images.original_max_side_px,
                )
        except (ValueError, SettingsError):
            return view
        return resolved_view(
            view,
            model="model",
            provider=model.provider,
            limit=output_limit(model, caps),
            response_format=fit_response_format(
                model.provider,
                response_format_for(model, caps),
                extraction_schema(self.schema.schema),
            ),
            images=images,
        )


def _absolute(path: Path) -> Path:
    """Expand ``~`` and resolve *path* against the current directory."""
    return path.expanduser().resolve()


def _input_path(spec: RunSpec | RangesSpec) -> Path:
    path = _absolute(spec.input.path if isinstance(spec, RunSpec) else spec.input)
    if not path.exists():
        raise InputError(f"Input path does not exist: {path}")
    if not (path.is_file() or path.is_dir()):
        raise InputError(f"Input path is neither a file nor directory: {path}")
    return path


def _output_dir(spec: RunSpec) -> Path | None:
    if spec.output.directory is None:
        return None
    path = _absolute(spec.output.directory)
    if path.exists() and not path.is_dir():
        raise InputError(f"Output path exists but is not a directory: {path}")
    return path


_WANTED: Mapping[str, str] = {
    "auto": "text, image or PDF files",
    "text": "text files",
    "images": "images",
    "pdf": "PDF files",
}


def _found(inputs: InputSpec, input_path: Path) -> list[SourceItem]:
    """Return every item of the input type at *input_path*, or the named files.

    Raises:
        InputError: The input cannot serve the input type.
        NoItemsError: The input holds no item of the type.
    """
    try:
        items = discover_items(
            input_path,
            inputs.type,
            files=inputs.files,
            recursive=inputs.recursive,
        )
    except DiscoveryError as exc:
        raise InputError(str(exc)) from exc
    if not items:
        wanted = _WANTED.get(inputs.type, f"{inputs.type} files")
        raise NoItemsError(f"No {wanted} found to process in: {input_path}")
    return items


def item_names(items: Sequence[SourceItem], input_path: Path) -> list[str]:
    """Return the names ``--select`` numbers and searches: paths below the input."""
    names: list[str] = []
    for item in items:
        try:
            names.append(item.path.relative_to(input_path).as_posix())
        except ValueError:
            names.append(item.name)
    return names


def _chosen(
    inputs: InputSpec, input_path: Path, items: list[SourceItem]
) -> tuple[list[SourceItem], tuple[str, ...]]:
    """Return the items the selection keeps and the selection's warnings.

    ``--files`` names its items at discovery; ``--select`` keeps the
    numbered or searched ones. Without either, a folder of several items
    needs ``--all``.

    Raises:
        NoItemsError: ``--select`` matches no item.
        InputError: Several items and no selection.
    """
    if inputs.files:
        return items, ()
    expr = inputs.select
    if expr is not None:
        names = item_names(items, input_path)
        selection = select_items(names, expr)
        if not selection.indices:
            raise NoItemsError(
                f"--select {expr!r} matches none of the "
                f"{len(items)} item(s) in: {input_path}"
            )
        warnings = tuple(
            f"--select part {part!r} is not an item number from 1 to "
            f"{len(items)}; it is ignored."
            for part in selection.unmatched
        )
        return selection.pick(items), warnings
    if inputs.all or len(items) == 1:
        return items, ()
    raise InputError(
        f"{input_path} holds {len(items)} items; choose them with --all, "
        "--select EXPR (numbers, ranges or a name search) or --files LIST.",
        items_total=len(items),
    )


def discover_inputs(
    inputs: InputSpec, input_path: Path
) -> tuple[list[SourceItem], tuple[str, ...]]:
    """Return the items *inputs* select at *input_path* and the warnings.

    Raises:
        PlanError: The input cannot serve the input type, holds no item, or
            several items without a selection.
    """
    return _chosen(inputs, input_path, _found(inputs, input_path))


def discover(spec: RunSpec) -> list[SourceItem]:
    """Return the items the run of *spec* processes at its input.

    Raises:
        PlanError: The input is missing or holds nothing to process, or a
            folder of several items has no selection.
    """
    return discover_inputs(spec.input, _input_path(spec))[0]


def _schema(argument: str, total: int = 0) -> ResolvedSchema:
    """Return the schema *argument* names.

    Raises:
        ConfigurationError: No schema matches the argument.
    """
    try:
        return resolve_schema(argument)
    except SpecError as exc:
        raise ConfigurationError(str(exc), items_total=total) from exc


def _uses_adjust(spec: RunSpec, types: Sequence[str]) -> bool:
    return spec.chunking.mode == "adjust" and "text" in types


def _with_prompt(item: PlannedItem, spec: RunSpec) -> PlannedItem:
    """Return *item* with its contexts resolved.

    Raises:
        ConfigurationError: A given context file or image cannot be used.
    """
    prompt = spec.prompt
    try:
        context = item_context(
            item.path, prompt.context, fallback=prompt.fallback_context
        )
        image = item_context_image(item.path, prompt.context_image)
        adjust = (
            item_context(
                item.path, prompt.context, "adjust", fallback=prompt.fallback_context
            )
            if item.type == "text" and spec.chunking.mode == "adjust"
            else None
        )
    except ContextError as exc:
        raise ConfigurationError(str(exc)) from exc
    return replace(item, context=context, context_image=image, adjust_context=adjust)


def _missing_key(
    model: ModelSpec,
    settings: Settings,
    kind: str,
    args: argparse.Namespace | None,
    total: int,
) -> str | None:
    """Return the first key variable of *model*'s calls of *kind* when none is set.

    Raises:
        ConfigurationError: The settings name no key variable for the provider.
    """
    try:
        variables = ext.key_variables(
            model.provider, model.endpoint, kind=kind, settings=settings, args=args
        )
    except SettingsError as exc:
        raise ConfigurationError(str(exc), items_total=total) from exc
    if variables and not any(os.environ.get(name) for name in variables):
        return variables[0]
    return None


def _missing_keys(
    spec: RunSpec,
    settings: Settings,
    types: Sequence[str],
    total: int,
    args: argparse.Namespace | None = None,
) -> tuple[MissingKey, ...]:
    """Return the unset key variables of the extraction and adjust calls.

    The extraction calls use a batch client under ``--batch``; boundary
    decisions of ``--chunking adjust`` always use a chat client.

    Raises:
        ConfigurationError: The settings name no key variable for the
            provider.
    """
    needs = [
        (
            "batch" if spec.run.batch else "chat",
            "--batch" if spec.run.batch else "extraction",
        )
    ]
    if _uses_adjust(spec, types):
        needs.append(("chat", "--chunking adjust"))
    model = spec.model
    missing: list[MissingKey] = []
    for kind, reason in needs:
        variable = _missing_key(model, settings, kind, args, total)
        if variable is not None:
            missing.append(MissingKey(variable, model.provider, reason))
    decision = spec.decision
    if decision.mode != "off":
        dialect = dialect_for(decision.model, decision.provider)
        try:
            variables = ext.key_variables(
                dialect, kind="decision", settings=settings, args=args
            )
        except SettingsError as exc:
            raise ConfigurationError(str(exc), items_total=total) from exc
        if variables and not any(os.environ.get(name) for name in variables):
            needs_flag = f"--decisions {decision.mode}"
            missing.append(MissingKey(variables[0], dialect, needs_flag))
    return tuple(missing)


def _check_batch(
    spec: RunSpec,
    settings: Settings,
    args: argparse.Namespace | None,
    total: int,
) -> None:
    """Require a provider with a batch API and the extension's consent.

    Raises:
        ConfigurationError: The provider has no batch API or the extension
            refuses batch submission.
    """
    provider = spec.model.provider
    if not supports_batch(provider):
        raise ConfigurationError(
            f"--batch needs a provider with a batch API (openai, anthropic, "
            f"google), not {provider}",
            items_total=total,
        )
    refusal = batch_refusal(spec, settings=settings, args=args)
    if refusal is not None:
        raise ConfigurationError(refusal, items_total=total)


def _check_model(spec: RunSpec, settings: Settings, images: bool, total: int) -> None:
    """Require a known endpoint and, for page images, a vision model.

    Raises:
        ConfigurationError: The model does not accept image input
            (:class:`~chronominer.common.capabilities.CapabilityError`) or
            its custom endpoint is undefined.
    """
    try:
        caps = model_capabilities(spec.model, settings)
        if images:
            ensure_image_support(spec.model.model, caps)
    except ValueError as exc:
        raise ConfigurationError(str(exc), items_total=total) from exc


@dataclass(frozen=True)
class _Classifier:
    """Classifies the items of one plan from the stores of their roots.

    *image* returns the run's image-settings record of an image or PDF item
    from the item's stored record (None when it has none); it is None when
    no item has page images.
    """

    spec: RunSpec
    output_dir: Path | None
    input_root: Path
    image: Callable[[Mapping[str, Any] | None], Mapping[str, Any] | None] | None
    schema_sha256: str | None = None

    def locate(self, item: SourceItem) -> tuple[Path, str, Path]:
        """Return the output root, document name and ``_output.json`` of *item*."""
        output = item_paths(
            item.path,
            output_dir=self.output_dir,
            layout=self.spec.output.layout,
            input_root=self.input_root,
        ).output
        root = self.output_dir if self.output_dir is not None else item.path.parent
        return root, document_name(output, root), output

    def planned(self, item: SourceItem) -> PlannedItem:
        """Return *item* planned with its resume state; write nothing.

        Raises:
            StoreRefusedError: The root's database is refused.
        """
        root, name, output = self.locate(item)
        formats = output_formats(self.spec.output.formats)
        outputs = [format_path(output, fmt) for fmt in formats]
        try:
            state = classify(
                root,
                name,
                outputs=outputs,
                image_settings=self.image if item.type in _IMAGE_TYPES else None,
                model=self.spec.model.model,
                schema_sha256=self.schema_sha256,
                input_path=item.path,
            )
        except StoreError as exc:
            raise StoreRefusedError(str(exc)) from exc
        if self.spec.run.force and state.state is not ItemState.BLOCKED:
            state = Classification(
                ItemState.NONE, name, state.doc_id, reason="overwrite mode"
            )
        return PlannedItem(item, state, root, name, output)


def _schema_sha256(schema: ResolvedSchema) -> str | None:
    """Return the SHA-256 of the schema text, as the store's artifacts key it."""
    try:
        text = schema.path.read_text(encoding="utf-8")
    except OSError:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _image(
    spec: RunSpec, settings: Settings, items: Sequence[SourceItem]
) -> Callable[[Mapping[str, Any] | None], Mapping[str, Any] | None] | None:
    """Return the image-settings record of an item from its stored record.

    The image options the run leaves unset keep their stored values
    (:func:`~chronominer.extract.llm.resume_images`). None when no item is
    an image or PDF; the record is None when it cannot be resolved.
    """
    if not any(item.type in _IMAGE_TYPES for item in items):
        return None
    side = settings.images.original_max_side_px

    def record(stored: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        images = resume_images(spec.images, stored)
        try:
            return image_record(replace(spec, images=images), side)
        except ValueError:
            return None

    return record


def _in_order(
    groups: Mapping[ItemState, list[PlannedItem]], discovered: Sequence[SourceItem]
) -> list[PlannedItem]:
    """Return the items to process (none or partial) in discovery order."""
    chosen = {
        id(item.source): item
        for state in (ItemState.NONE, ItemState.PARTIAL)
        for item in groups[state]
    }
    return [chosen[id(item)] for item in discovered if id(item) in chosen]


def plan(
    spec: RunSpec,
    settings: Settings,
    *,
    sources: Mapping[str, Source] | None = None,
    args: argparse.Namespace | None = None,
    require_keys: bool | None = None,
) -> RunPlan:
    """Discover and classify the items of an extract run; write nothing.

    *sources* maps spec fields to where their values came from; the plan
    reports them with the effective spec. *args* holds the parsed command
    line, passed to the extension's batch consent and key variables. With
    *require_keys* (by default unless the spec is a dry run), an unset key
    variable is a :class:`ConfigurationError`; otherwise the plan lists it
    in ``missing_keys`` and, for a dry run, as a warning.

    Raises:
        PlanError: The run cannot proceed; the subclass names the reason and
            ``exit_code`` the exit code. A versionless, newer or unreadable
            database raises :class:`StoreRefusedError`.
    """
    output_dir = _output_dir(spec)
    input_path = _input_path(spec)
    logger.info("Input=%s, Output=%s", input_path, output_dir or "beside input")
    discovered, warnings = discover_inputs(spec.input, input_path)
    schema = _schema(spec.prompt.schema, len(discovered))

    classifier = _Classifier(
        spec,
        output_dir,
        input_path,
        _image(spec, settings, discovered),
        _schema_sha256(schema),
    )
    groups: dict[ItemState, list[PlannedItem]] = {state: [] for state in ItemState}
    seen: dict[tuple[Path, str], SourceItem] = {}
    for item in discovered:
        root, name, output = classifier.locate(item)
        other = seen.setdefault((root.resolve(), name.casefold()), item)
        if other is not item:
            raise InputError(
                f"{other.path} and {item.path} share the output {output}; "
                "rename one or run them separately.",
                items_total=len(discovered),
            )
        planned = classifier.planned(item)
        groups[planned.resume.state].append(planned)
    to_process = [_with_prompt(item, spec) for item in _in_order(groups, discovered)]
    logger.info(
        "%d item(s) to process, %d already complete",
        len(to_process),
        len(groups[ItemState.COMPLETE]),
    )
    try:
        pages = parse_page_range(spec.input.pages) if spec.input.pages else None
    except ValueError as exc:
        raise ConfigurationError(str(exc), items_total=len(to_process)) from exc
    warnings += _nothing_selected(pages, groups[ItemState.COMPLETE])

    run_plan = RunPlan(
        spec=spec,
        input_path=input_path,
        output_dir=output_dir,
        discovered=tuple(discovered),
        to_process=tuple(to_process),
        skipped=tuple(groups[ItemState.COMPLETE]),
        schema=schema,
        warnings=warnings,
        settings=settings,
        sources=dict(sources or {}),
        pages=pages,
        unstored=tuple(groups[ItemState.UNSTORED]),
        blocked=tuple(groups[ItemState.BLOCKED]),
        refused=tuple(groups[ItemState.REFUSED]),
    )
    total = len(to_process)
    if run_plan.batch:
        _check_batch(spec, settings, args, total)
    missing = _missing_keys(spec, settings, run_plan.types, total, args)
    if require_keys is None:
        require_keys = not spec.run.dry_run
    if missing and require_keys:
        raise ConfigurationError(missing[0].message, items_total=total)
    if missing:
        notes = (*warnings, *(key.message for key in missing))
        run_plan = replace(
            run_plan,
            missing_keys=missing,
            warnings=notes if spec.run.dry_run else warnings,
        )
    images = any(kind in _IMAGE_TYPES for kind in run_plan.types)
    _check_model(spec, settings, images, total)
    return run_plan


def _nothing_selected(
    pages: PageRange | None, complete: Sequence[PlannedItem]
) -> tuple[str, ...]:
    """Return a warning for each complete item of which *pages* selects no chunk."""
    if pages is None:
        return ()
    return tuple(
        f"Page range '{pages.describe()}' selects none of the "
        f"{planned.resume.total} chunks of '{planned.name}', which is complete. "
        "Skipping."
        for planned in complete
        if planned.resume.total > 0 and not pages.resolve(planned.resume.total)
    )


@dataclass(frozen=True)
class ItemResult:
    """The outcome of one item.

    *success* is False when the item raised or was refused; *error* then
    says why. *outputs* are the absolute paths of its rendered files.
    *submitted* marks an item whose chunks wait in a batch for ``batch
    check``; it counts neither complete nor failed. *partial* marks a
    successful ``--pages`` item whose document is still short of complete;
    it counts neither complete nor failed either. *failures* are the failed
    units of a failed item's document, with their error types and errors.
    """

    name: str
    success: bool
    outputs: tuple[str, ...] = ()
    error: str | None = None
    submitted: bool = False
    partial: bool = False
    failures: tuple[UnitRow, ...] = ()


@dataclass(frozen=True)
class RunResult:
    """The outcome of a run.

    *usage* holds the tokens of the run's model calls per role, where the
    responses reported them. *databases* are the database files written.
    *blocked* names the items not processed because another run held them
    or a batch holds their chunks, and *blocked_reasons* gives the reason of
    each, in the same order, a lease with its expiry; two items of the same
    name in different folders keep their own reasons. *leased* names those
    of them that another run's lease held; they make the run exit 1, while
    items waiting for a batch are submitted work. *unstored* names the items
    skipped because their outputs have no store rows, *unselected* those of
    which ``--pages`` selects no chunk.
    """

    items: tuple[ItemResult, ...]
    skipped: tuple[str, ...] = ()
    seconds: float = 0.0
    usage: Mapping[str, Usage] = field(default_factory=dict)
    databases: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()
    unstored: tuple[str, ...] = ()
    unselected: tuple[str, ...] = ()
    blocked_reasons: tuple[str, ...] = ()
    leased: tuple[str, ...] = ()

    @property
    def total_usage(self) -> Usage:
        """The tokens of all roles together."""
        total = Usage()
        for usage in self.usage.values():
            total = total + usage
        return total

    @property
    def complete(self) -> int:
        """The number of items that finished complete.

        Submitted items and items ``--pages`` left partial are left out.
        """
        return sum(
            1
            for item in self.items
            if item.success and not item.submitted and not item.partial
        )

    @property
    def partial(self) -> int:
        """The number of items whose documents ``--pages`` left partial."""
        return sum(1 for item in self.items if item.success and item.partial)

    @property
    def submitted(self) -> int:
        """The number of items whose chunks wait in a batch."""
        return sum(1 for item in self.items if item.success and item.submitted)

    @property
    def failed(self) -> int:
        """The number of items that failed."""
        return sum(1 for item in self.items if not item.success)

    @property
    def exit_code(self) -> ExitCode:
        """1 when an item failed or another run's lease held one, else 0."""
        return exit_code_for(failed=self.failed + len(self.leased))

    @property
    def outputs(self) -> list[str]:
        """The absolute paths written by the run, item by item, then databases."""
        return [path for item in self.items for path in item.outputs] + list(
            self.databases
        )


def writer_options(events: RunEvents) -> WriterOptions:
    """Return the writer options of a run: busy waits warn through *events*."""

    def on_busy(path: Path, waited: float) -> None:
        events.warning(
            f"The database {path} is busy (another program holds it open); "
            f"waited {waited:.0f} s. Results are kept in memory and written "
            "once it is free."
        )

    return WriterOptions(
        busy_timeout=BUSY_TIMEOUT, warn_interval=BUSY_WARN_INTERVAL, on_busy=on_busy
    )


def _read_text(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read %s for the store: %s", path, exc)
        return None


class _Stores:
    """The stores of a run: one per output root, opened on first use.

    Opening a store stamps the run, stores the prompt and schema texts,
    marks submissions left ``submitting`` as unresolved and starts the
    heartbeat that renews the run's leases.
    """

    def __init__(self, run_plan: RunPlan, events: RunEvents) -> None:
        self._plan = run_plan
        self._events = events
        self._stores: dict[Path, Store] = {}
        self._keepers: list[LeaseKeeper] = []

    @property
    def databases(self) -> tuple[str, ...]:
        """The database files of the opened stores, in opening order."""
        return tuple(str(store.path.resolve()) for store in self._stores.values())

    async def get(self, root: Path) -> Store:
        """Return the open store of output root *root*.

        Raises:
            StoreRefusedError: The root's database is refused.
        """
        key = root.resolve()
        store = self._stores.get(key)
        if store is not None:
            return store
        key.mkdir(parents=True, exist_ok=True)
        store = open_store(key, writer=writer_options(self._events))
        try:
            await store.open()
        except StoreError as exc:
            raise StoreRefusedError(str(exc)) from exc
        self._stores[key] = store
        await store.begin_run("extract", spec=self._plan.effective_spec())
        await self._artifacts(store)
        marked = await store.mark_stale_submissions()
        if marked:
            self._events.warning(
                f"{len(marked)} batch submission(s) in {database_path(key)} were "
                "left without a provider batch id and are now unresolved; their "
                "chunks stay blocked. Run 'chronominer batch check' to reconcile "
                "them."
            )
        keeper = store.heartbeat(on_lost=self._lost)
        keeper.start()
        self._keepers.append(keeper)
        return store

    def _lost(self, lease: Lease) -> None:
        self._events.warning(
            f"The lease on document {lease.doc_id} expired; its remaining chunks "
            "are not stored by this run."
        )

    async def _artifacts(self, store: Store) -> None:
        """Store the prompt templates and the schema text of the run."""
        types = self._plan.types
        prompts = [TEXT_PROMPT_PATH] if "text" in types else []
        if any(kind in _IMAGE_TYPES for kind in types):
            prompts.append(IMAGE_PROMPT_PATH)
        texts = [("prompt", _read_text(path)) for path in prompts]
        texts.append(("schema", _read_text(self._plan.schema_path)))
        for kind, text in texts:
            if text is not None:
                await store.store_artifact(kind, text)

    async def close(self, exit_code: int) -> None:
        """Stop the heartbeats, record the end of the run and close the stores."""
        for keeper in self._keepers:
            await keeper.stop()
        for store in self._stores.values():
            try:
                if store.run_id is not None:
                    await store.end_run(exit_code)
            except sqlite3.Error as exc:
                logger.warning("Could not record the run end in %s: %s", store, exc)
            try:
                await store.close()
            except sqlite3.Error as exc:
                logger.warning("Could not close %s: %s", store.path, exc)


async def _note(
    store: Store, doc_id: int | None, message: str, events: RunEvents
) -> None:
    events.warning(message)
    try:
        await store.add_note(message, doc_id=doc_id)
    except sqlite3.Error as exc:
        logger.warning("Could not record a note in %s: %s", store.path, exc)


def _input_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return file_sha256(path)
    except OSError as exc:
        logger.warning("Could not hash %s: %s", path, exc)
        return None


@dataclass(frozen=True)
class _Blocked:
    """An item whose document another run leased after planning."""

    reason: str


@dataclass
class _ItemRun:
    """One item of a run: its store document, lease, extraction and render.

    *held* is the reason the item could not be leased, once :meth:`lease`
    found another run holding it.
    """

    job: Job
    planned: PlannedItem
    stores: _Stores
    held: str | None = None

    async def lease(self) -> tuple[Store, int, Lease] | None:
        """Ensure the item's document and lease it; None when another holds it."""
        planned = self.planned
        store = await self.stores.get(planned.root)
        doc_id = await store.ensure_document(
            planned.document,
            input_path=planned.path,
            input_sha256=_input_sha256(planned.path),
            kind=planned.type,
        )
        lease = await store.acquire(doc_id)
        if lease is None:
            state = None
            with contextlib.suppress(StoreError, sqlite3.Error):
                state = store.resume_state(doc_id)
            self.held = lease_reason(
                state.lease_run_id if state is not None else None,
                state.lease_until if state is not None else None,
            )
            await _note(
                store,
                doc_id,
                f"Skipping '{planned.name}': another run holds its document "
                f"{planned.document} in {store.path} ({self.held}).",
                self.job.events,
            )
            return None
        return store, doc_id, lease

    async def render(
        self, store: Store, doc_id: int, lease: Lease, *, force: bool
    ) -> tuple[str, ...]:
        """Render the item's formats; return the files that hold the rendering."""
        report = await render_item(
            store, doc_id, self.job.formats, self.planned.output, lease, force=force
        )
        for message in report.warnings:
            self.job.events.warning(message)
        current = {*report.written, *report.unchanged}
        targets = [format_path(self.planned.output, name) for name in self.job.formats]
        return tuple(str(path) for path in targets if path in current)

    async def render_or_note(
        self, store: Store, doc_id: int, lease: Lease, *, force: bool
    ) -> tuple[str, ...]:
        """Render like :meth:`render`; note a lost lease or a file error instead."""
        name = self.planned.name
        try:
            return await self.render(store, doc_id, lease, force=force)
        except LeaseLostError as exc:
            message = f"'{name}' was not rendered: {exc}"
        except OSError as exc:
            target = exc.filename or ", ".join(
                str(format_path(self.planned.output, fmt)) for fmt in self.job.formats
            )
            message = (
                f"'{name}': {target} was not rendered ({exc}); its chunks are "
                "stored and a later render or run replaces the file."
            )
        await _note(store, doc_id, message, self.job.events)
        return ()

    async def extract(
        self,
        store: Store,
        doc_id: int,
        lease: Lease,
        extractor: ExtractionMethod,
        pages: PageRange | None,
    ) -> tuple[tuple[str, ...], str | None]:
        """Extract the item's chunks, then render; return the outputs and error."""
        planned, job = self.planned, self.job
        spec = job.spec
        if planned.resume.note:
            await _note(store, doc_id, planned.resume.note, job.events)
        if planned.context:
            await store.store_artifact("context", planned.context)
        handle = DocumentHandle(
            store,
            doc_id,
            planned.document,
            lease,
            model=spec.model.model,
            provider=spec.model.provider,
        )
        item = Item(
            path=planned.path,
            source=planned.type,
            pages=pages,
            context=planned.context,
            context_image=planned.context_image,
            adjust_context=planned.adjust_context,
            document=handle,
        )
        returned: tuple[str, ...] = ()
        error: str | None = None
        try:
            returned = tuple(str(path) for path in await extractor.extract(item))
        except LeaseLostError as exc:
            error = f"its lease was lost: {exc}"
            await _note(store, doc_id, f"'{planned.name}': {error}", job.events)
            return returned, error
        except NothingSelectedError as exc:
            await _note(store, doc_id, str(exc), job.events)
            raise
        except Exception as exc:
            logger.exception("Failed to process item %s: %s", planned.name, exc)
            error = str(exc) or type(exc).__name__
        force = spec.run.force or planned.resume.state is ItemState.NONE
        rendered = await self.render_or_note(store, doc_id, lease, force=force)
        outputs = tuple(dict.fromkeys((*returned, *rendered)))
        return outputs, error


async def _run_item(  # noqa: PLR0913 - the item's run context
    job: Job,
    extractor: ExtractionMethod,
    planned: PlannedItem,
    index: int,
    total: int,
    pages: PageRange | None,
    stores: _Stores,
    submitted: set[tuple[Path, int]],
) -> ItemResult | _Blocked:
    """Process one item; :class:`_Blocked` when another run holds its document.

    *submitted* holds the documents (database, id) whose chunks this run
    submitted to a batch.

    Raises:
        NothingSelectedError: ``--pages`` selects no chunk of the item.
    """
    events = job.events
    item_run = _ItemRun(job, planned, stores)
    leased = await item_run.lease()
    if leased is None:
        return _Blocked(item_run.held or "another run holds its document")
    store, doc_id, lease = leased
    events.item_started(ItemStarted(planned.name, planned.type, index, total))
    try:
        outputs, error = await item_run.extract(store, doc_id, lease, extractor, pages)
    finally:
        with contextlib.suppress(sqlite3.Error):
            await store.release(lease)
    success = error is None
    waiting = (store.path, doc_id) in submitted
    partial = (
        success
        and not waiting
        and pages is not None
        and _short_of_complete(store, doc_id)
    )
    events.item_finished(
        ItemFinished(
            planned.name,
            index,
            total,
            success,
            outputs,
            None,
            error,
            submitted=waiting,
            partial=partial,
        )
    )
    return ItemResult(
        planned.name,
        success,
        outputs,
        error,
        submitted=waiting,
        partial=partial,
        failures=() if success else _failed_units(store, doc_id),
    )


def _failed_units(store: Store, doc_id: int) -> tuple[UnitRow, ...]:
    """Return the failed units of the document *doc_id*; none when unreadable."""
    with contextlib.suppress(StoreError, sqlite3.Error):
        return tuple(store.failed_units(doc_id=doc_id))
    return ()


def _short_of_complete(store: Store, doc_id: int) -> bool:
    """Whether the document *doc_id* is still short of complete."""
    with contextlib.suppress(StoreError, sqlite3.Error):
        document = store.document(doc_id)
        return document is not None and document.status != "complete"
    return False


async def _render_only(
    job: Job, planned: PlannedItem, stores: _Stores
) -> _Blocked | None:
    """Render a complete item whose files are missing or stale; no model call.

    Return :class:`_Blocked` when another run holds the item's document.
    """
    item_run = _ItemRun(job, planned, stores)
    leased = await item_run.lease()
    if leased is None:
        return _Blocked(item_run.held or "another run holds its document")
    store, doc_id, lease = leased
    try:
        await item_run.render_or_note(store, doc_id, lease, force=False)
    finally:
        with contextlib.suppress(sqlite3.Error):
            await store.release(lease)
    return None


@dataclass
class _Held:
    """The blocked items of a run, their reasons and those leased elsewhere.

    It starts from the plan's blocked items; *reasons* holds the reason of
    each blocked item, in the order of *blocked*; *complete* holds the ids
    of complete items whose render found their document leased.
    """

    blocked: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    leased: list[str] = field(default_factory=list)
    complete: set[int] = field(default_factory=set)

    @classmethod
    def of(cls, run_plan: RunPlan) -> _Held:
        held = cls()
        for item in run_plan.blocked:
            held.blocked.append(item.name)
            held.reasons.append(item.reason)
            if item.resume.leased_by:
                held.leased.append(item.name)
        return held

    def add(self, planned: PlannedItem, outcome: _Blocked) -> None:
        """Record *planned* as held by another run's lease."""
        self.blocked.append(planned.name)
        self.leased.append(planned.name)
        self.reasons.append(outcome.reason)


async def _render_again(
    job: Job, run_plan: RunPlan, stores: _Stores, held: _Held
) -> None:
    """Render the plan's complete items whose files are missing or stale.

    An item another run holds goes to *held*.
    """
    for planned in run_plan.render_only:
        outcome = await _render_only(job, planned, stores)
        if outcome is not None:
            held.complete.add(id(planned))
            held.add(planned, outcome)


async def _note_skipped(job: Job, run_plan: RunPlan, stores: _Stores) -> None:
    """Record a note on the document of each blocked or unstored item."""
    for planned in (*run_plan.blocked, *run_plan.unstored):
        if planned.resume.doc_id is None:
            job.events.warning(f"Skipping '{planned.name}': {planned.reason}.")
            continue
        store = await stores.get(planned.root)
        await _note(
            store,
            planned.resume.doc_id,
            f"Skipping '{planned.name}': {planned.reason}.",
            job.events,
        )


@dataclass(frozen=True)
class _Unbuildable:
    """The outcome of a submission whose requests could not be built."""

    detail: str
    status: str = "refused"
    submission_ids: tuple[str, ...] = ()


async def _submit_batch(
    payloads: Sequence[Any],
    *,
    document: DocumentHandle,
    spec: RunSpec,
    settings: Settings,
    schema_path: Path | None,
    context: str | None = None,
    context_image: Path | None = None,
) -> SubmitOutcome:
    """Submit one document's chunks through the batch package.

    A submission whose requests cannot be built is reported as refused.
    """
    try:
        return await batch_submit.submit_batch(
            list(payloads),
            document=document,
            spec=spec,
            settings=settings,
            schema_path=schema_path,
            context=context,
            context_image=context_image,
        )
    except batch_submit.BatchSubmissionError as exc:
        return _Unbuildable(str(exc) or type(exc).__name__)


def _recording(submitted: set[tuple[Path, int]]) -> BatchSubmitter:
    """Return the run's submitter; it adds each submitted document to *submitted*."""

    async def submit(  # noqa: PLR0913 - the BatchSubmitter signature
        payloads: Sequence[Any],
        *,
        document: DocumentHandle,
        spec: RunSpec,
        settings: Settings,
        schema_path: Path | None,
        context: str | None = None,
        context_image: Path | None = None,
    ) -> SubmitOutcome:
        outcome = await _submit_batch(
            payloads,
            document=document,
            spec=spec,
            settings=settings,
            schema_path=schema_path,
            context=context,
            context_image=context_image,
        )
        if outcome.status == "submitted":
            submitted.add((document.store.path, document.doc_id))
        return outcome

    return submit


async def _open_extractor(
    run_plan: RunPlan, job: Job, stack: contextlib.AsyncExitStack
) -> ExtractionMethod:
    """Open the run's extractor.

    Raises:
        ConfigurationError: The model client cannot be built.
    """
    extractor = Extractor(job)
    try:
        await extractor.open()
    except Exception as exc:
        raise ConfigurationError(
            f"Cannot build the extraction client: {exc}",
            items_total=len(run_plan.to_process),
        ) from exc
    stack.push_async_callback(extractor.close)
    return extractor


async def run(run_plan: RunPlan, events: RunEvents | None = None) -> RunResult:
    """Extract the plan's items one after another on the running event loop.

    The run owns one :class:`~chronominer.llm.caller.CallEnv`. When the plan
    has items to process, it opens its extractor before any item starts or
    any file is written; otherwise it builds no model client. It opens one
    store per output root, stamped with the run. Each item's document is
    leased while its chunks are extracted and while the run core renders
    its formats from the store, also after a failed chunk; complete items
    with missing files are rendered without model calls. An item whose
    document another run holds is skipped with a note, as are unstored items
    and those of which ``--pages`` selects no chunk; an item another run's
    lease holds, also a complete one to render again, counts as blocked and
    the run records exit code 1 like a failure. A failed item is logged
    and counted; the run continues with the next one. Refused items count as
    failed. Cancellation propagates.

    Raises:
        ValueError: The plan is a dry run.
        ConfigurationError: The model client cannot be built; nothing has
            been written. :class:`StoreRefusedError` when a database became
            unusable after planning.
    """
    if run_plan.dry_run:
        raise ValueError("a dry-run plan is reported, not run")
    sink: RunEvents = events if events is not None else NullEvents()
    env = CallEnv.create(run_plan.settings, sink)
    started = time.monotonic()
    results: list[ItemResult] = []
    held = _Held.of(run_plan)
    unselected: list[str] = []
    submitted: set[tuple[Path, int]] = set()
    job = Job(
        spec=run_plan.spec,
        settings=run_plan.settings,
        events=sink,
        env=env,
        output_dir=run_plan.output_dir,
        input_root=run_plan.input_path,
        schema_path=run_plan.schema_path,
        batch=run_plan.batch,
        submit=_recording(submitted) if run_plan.batch else None,
    )
    stores = _Stores(run_plan, sink)
    exit_code: int = ExitCode.FAILURE
    try:
        async with contextlib.AsyncExitStack() as stack:
            extractor = (
                await _open_extractor(run_plan, job, stack)
                if run_plan.to_process
                else None
            )
            if run_plan.output_dir is not None:
                run_plan.output_dir.mkdir(parents=True, exist_ok=True)
            await _note_skipped(job, run_plan, stores)
            await _render_again(job, run_plan, stores, held)
            if extractor is not None:
                total = len(run_plan.to_process)
                for index, planned in enumerate(run_plan.to_process, start=1):
                    try:
                        result = await _run_item(
                            job,
                            extractor,
                            planned,
                            index,
                            total,
                            run_plan.pages,
                            stores,
                            submitted,
                        )
                    except NothingSelectedError:
                        unselected.append(planned.name)
                        continue
                    if isinstance(result, _Blocked):
                        held.add(planned, result)
                    else:
                        results.append(result)
            for planned in run_plan.refused:
                sink.warning(planned.reason)
                results.append(ItemResult(planned.name, False, (), planned.reason))
        failed = any(not result.success for result in results)
        exit_code = ExitCode.FAILURE if failed or held.leased else ExitCode.OK
    except (asyncio.CancelledError, KeyboardInterrupt):
        exit_code = ExitCode.INTERRUPTED
        raise
    finally:
        await stores.close(exit_code)
    return RunResult(
        items=tuple(results),
        skipped=tuple(
            item.name for item in run_plan.skipped if id(item) not in held.complete
        ),
        seconds=time.monotonic() - started,
        usage={role: env.usage.get(role) for role in env.usage.roles},
        databases=stores.databases,
        blocked=tuple(held.blocked),
        unstored=tuple(item.name for item in run_plan.unstored),
        unselected=tuple(unselected),
        blocked_reasons=tuple(held.reasons),
        leased=tuple(held.leased),
    )


type RangesSpec = RangesGenerateSpec | RangesAdjustSpec


@dataclass(frozen=True)
class RangesPlan:
    """What a ``ranges generate`` or ``ranges adjust`` run will do.

    *files* are the text inputs to process, in discovery order; each writes
    its ``_line_ranges.txt`` and ``chronominer.sqlite`` beside it. For
    adjust, *schema* names the boundary type, *contexts* holds each file's
    adjust context and *response_format* the format of the boundary calls;
    files without a ``_line_ranges.txt`` are left out with a warning.
    """

    spec: RangesSpec
    input_path: Path
    files: tuple[Path, ...]
    warnings: tuple[str, ...] = ()
    settings: Settings = field(default_factory=Settings)
    sources: Mapping[str, Source] = field(default_factory=dict)
    schema: ResolvedSchema | None = None
    contexts: Mapping[Path, str | None] = field(default_factory=dict)
    pages: PageRange | None = None
    response_format: ResponseFormat = "schema"

    @property
    def adjust(self) -> bool:
        """Whether the plan adjusts ranges instead of generating them."""
        return isinstance(self.spec, RangesAdjustSpec)

    @property
    def command(self) -> str:
        """The command name recorded with the run."""
        return "ranges adjust" if self.adjust else "ranges generate"

    @property
    def dry_run(self) -> bool:
        """Whether the plan is only reported."""
        if isinstance(self.spec, RangesAdjustSpec):
            return self.spec.run.dry_run
        return self.spec.dry_run

    def effective_spec(self) -> dict[str, dict[str, Any]]:
        """Return ``{field: {"value": ..., "source": ...}}`` for every field."""
        return effective_spec(self.spec, self.sources)


def _text_files(input_path: Path, recursive: bool) -> list[Path]:
    """Return the text inputs at *input_path*.

    Raises:
        InputError: The input is not a text file.
        NoItemsError: The folder holds no text file.
    """
    try:
        items = discover_items(input_path, "text", recursive=recursive)
    except DiscoveryError as exc:
        raise InputError(str(exc)) from exc
    if not items:
        raise NoItemsError(f"No text files found to process in: {input_path}")
    return [item.path for item in items]


def _adjust_checks(
    spec: RangesAdjustSpec,
    settings: Settings,
    files: list[Path],
    args: argparse.Namespace | None,
) -> tuple[
    ResolvedSchema,
    dict[Path, str | None],
    ResponseFormat,
    PageRange | None,
    MissingKey | None,
]:
    """Return the schema, contexts, response format and selection of an adjust.

    The last value is the unset key variable of a dry run, if any.

    Raises:
        ConfigurationError: Unknown schema, unset key variable (not for a
            dry run), undefined endpoint, unusable context file or invalid
            range selection.
    """
    total = len(files)
    schema = _schema(spec.schema, total)
    missing = _adjust_key(spec, settings, args, total)
    if missing is not None and not spec.run.dry_run:
        raise ConfigurationError(missing.message, items_total=total)
    try:
        caps = model_capabilities(spec.model, settings)
        response_format = response_format_for(spec.model, caps)
        pages = parse_page_range(spec.pages) if spec.pages else None
    except ValueError as exc:
        raise ConfigurationError(str(exc), items_total=total) from exc
    try:
        contexts = {
            path: item_context(
                path, spec.context, "adjust", fallback=spec.fallback_context
            )
            for path in files
        }
    except ContextError as exc:
        raise ConfigurationError(str(exc), items_total=total) from exc
    return schema, contexts, response_format, pages, missing


def _adjust_key(
    spec: RangesAdjustSpec,
    settings: Settings,
    args: argparse.Namespace | None,
    total: int,
) -> MissingKey | None:
    """Return the unset key variable of the boundary calls, if any.

    Raises:
        ConfigurationError: The settings name no key variable for the
            provider.
    """
    variable = _missing_key(spec.model, settings, "chat", args, total)
    if variable is None:
        return None
    return MissingKey(variable, spec.model.provider, "ranges adjust")


def plan_ranges(
    spec: RangesSpec,
    settings: Settings,
    *,
    sources: Mapping[str, Source] | None = None,
    args: argparse.Namespace | None = None,
) -> RangesPlan:
    """Discover the text files of a ranges run and check it; write nothing.

    ``ranges generate`` needs no model call. ``ranges adjust`` checks the
    schema, the key variable of the model's provider, its endpoint, the
    adjust context of each file and the range selection; a file without a
    ``_line_ranges.txt`` is left out with a warning naming ``ranges
    generate``.

    Raises:
        PlanError: The run cannot proceed; the subclass names the reason and
            ``exit_code`` the exit code.
    """
    input_path = _input_path(spec)
    files = _text_files(input_path, spec.recursive)
    if not isinstance(spec, RangesAdjustSpec):
        return RangesPlan(
            spec,
            input_path,
            tuple(files),
            settings=settings,
            sources=dict(sources or {}),
        )
    schema, contexts, response_format, pages, missing = _adjust_checks(
        spec, settings, files, args
    )
    ready = [path for path in files if ranges_path(path).is_file()]
    warnings = tuple(
        f"Skipping '{path.name}': {ranges_path(path).name} is missing; run "
        "'chronominer ranges generate' first."
        for path in files
        if path not in ready
    )
    if missing is not None:
        warnings += (missing.message,)
    if not ready:
        raise NoItemsError(
            f"No text file in {input_path} has a line-ranges file; run "
            "'chronominer ranges generate' first.",
            items_total=len(files),
        )
    return RangesPlan(
        spec,
        input_path,
        tuple(ready),
        warnings=warnings,
        settings=settings,
        sources=dict(sources or {}),
        schema=schema,
        contexts={path: contexts[path] for path in ready},
        pages=pages,
        response_format=response_format,
    )


def _generate_result(report: GenerateReport) -> ItemResult:
    outputs = (str(report.ranges_file),) if report.ok else ()
    return ItemResult(report.text_file.name, report.ok, outputs, report.error)


async def _run_generate(
    ranges_plan: RangesPlan, spec: RangesGenerateSpec, events: RunEvents
) -> list[ItemResult]:
    reports = await generate_files(
        list(ranges_plan.files),
        chunk_tokens=spec.chunk_tokens,
        model=spec.model,
        events=events,
        run_spec=ranges_plan.effective_spec(),
    )
    return [_generate_result(report) for report in reports]


async def _adjust_item(
    store: Store,
    text_file: Path,
    position: tuple[int, int],
    caller: ModelCaller,
    options: AdjustOptions,
    events: RunEvents,
) -> ItemResult:
    """Adjust one file's ranges and report it; *position* is (index, total)."""
    index, total = position
    name = text_file.name
    events.item_started(ItemStarted(name, "text", index, total))
    outputs: tuple[str, ...] = ()
    error: str | None = None
    try:
        await adjust_ranges(
            store,
            text_file,
            caller=caller,
            model=caller.spec.model,
            options=options,
            events=events,
        )
        outputs = (str(ranges_path(text_file)),)
    except Exception as exc:  # noqa: BLE001 - one failed file must not stop the run
        logger.error("Adjusting %s failed: %s", name, exc)
        error = str(exc) or type(exc).__name__
    success = error is None
    events.item_finished(
        ItemFinished(name, index, total, success, outputs, None, error)
    )
    return ItemResult(name, success, outputs, error)


async def _run_adjust(
    ranges_plan: RangesPlan,
    spec: RangesAdjustSpec,
    env: CallEnv,
    events: RunEvents,
) -> tuple[list[ItemResult], list[str]]:
    """Adjust each file with the store of its folder; one run per folder."""
    schema = ranges_plan.schema
    assert schema is not None
    try:
        caller = env.caller(ADJUST_ROLE, spec.model)
    except Exception as exc:
        raise ConfigurationError(
            f"Cannot build the adjust client: {exc}",
            items_total=len(ranges_plan.files),
        ) from exc
    results: list[ItemResult] = []
    databases: list[str] = []
    try:
        for folder, files in group_by_folder(list(ranges_plan.files)).items():
            async with open_folder_store(folder) as store:
                databases.append(str(store.path.resolve()))
                await store.begin_run(
                    ranges_plan.command, spec=ranges_plan.effective_spec()
                )
                failed = False
                for text_file in files:
                    options = AdjustOptions(
                        boundary_type=schema.name,
                        context=ranges_plan.contexts.get(text_file),
                        response_format=ranges_plan.response_format,
                        context_window=spec.context_window,
                        pages=ranges_plan.pages,
                        force=spec.run.force,
                        prompt_caching=env.settings.prompt_caching,
                    )
                    result = await _adjust_item(
                        store,
                        text_file,
                        (len(results) + 1, len(ranges_plan.files)),
                        caller,
                        options,
                        events,
                    )
                    failed = failed or not result.success
                    results.append(result)
                await store.end_run(ExitCode.FAILURE if failed else ExitCode.OK)
    finally:
        await caller.aclose()
    return results, databases


async def run_ranges(
    ranges_plan: RangesPlan, events: RunEvents | None = None
) -> RunResult:
    """Generate or adjust the line ranges of the plan's text files.

    Each file's ranges are recorded in ``chronominer.sqlite`` beside it and
    its ``_line_ranges.txt`` is rendered from them. A file that fails is
    reported and the others go on.

    Raises:
        ValueError: The plan is a dry run.
        ConfigurationError: The adjust client cannot be built.
    """
    if ranges_plan.dry_run:
        raise ValueError("a dry-run plan is reported, not run")
    sink: RunEvents = events if events is not None else NullEvents()
    for message in ranges_plan.warnings:
        sink.warning(message)
    started = time.monotonic()
    spec = ranges_plan.spec
    env = CallEnv.create(ranges_plan.settings, sink)
    if isinstance(spec, RangesAdjustSpec):
        results, databases = await _run_adjust(ranges_plan, spec, env, sink)
    else:
        results = await _run_generate(ranges_plan, spec, sink)
        folders = group_by_folder(list(ranges_plan.files))
        databases = [str(database_path(folder).resolve()) for folder in folders]
    return RunResult(
        items=tuple(results),
        seconds=time.monotonic() - started,
        usage={role: env.usage.get(role) for role in env.usage.roles},
        databases=tuple(databases),
    )
