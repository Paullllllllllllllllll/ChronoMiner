"""Repair: extract the failed chunks of stored documents again.

:func:`plan_repair` finds the documents of ``--input``: every document of
the ``chronominer.sqlite`` databases at or below a folder, the documents of
one database file, or the document of one file (the nearest database at or
above it whose outputs registry lists the file, else the nearest one that
records the file as a document's input). ``--schema`` keeps the documents
extracted with that schema and *files* those whose name, input file or
rendered file names match. It selects each document's failed chunk units
(with ``--skipped`` the units the decision pre-pass skipped instead),
narrowed by ``--indices`` (0-based chunk indices), leaves out the units an
open or unresolved batch submission holds and the documents whose input
changed since extraction (counted as failed), picks the image settings the
pages were rendered with, resolves each document's schema, context and
context image as its extraction run had them and checks provider access. It
writes nothing and raises :class:`RepairPlanError` (exit 2) for usage and
configuration errors, a database the store refuses included.

:func:`run_repair` leases each document, builds its chunks again from the
recorded source (the line range of a text, the page of a PDF, the image
file) and extracts them: synchronously through
:func:`~chronominer.extract.units.run_unit_stream`, after resetting the
units (their responses stay), or through the provider's batch API with
:func:`~chronominer.batch.submit.submit_units`, a wait on the submissions
and :func:`~chronominer.batch.finalize.finalize_submission`. Every unit is
written with its entries in a transaction that checks the lease; the
document's ``_output.json`` and CSV are then rendered again from the store,
and a finalized batch's provider files are deleted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import time
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from chronominer import ext
from chronominer.batch.bodies import build_plan
from chronominer.batch.clients import BatchClients
from chronominer.batch.finalize import (
    FINALIZED_OUTCOME,
    DocumentSchema,
    delete_provider_files,
    document_schema,
    finalize_submission,
    render_documents,
)
from chronominer.batch.submit import (
    BatchSubmissionError,
    UnitRequest,
    chunk_request,
    submit_units,
)
from chronominer.batch.wait import wait_for_submissions
from chronominer.common.batch import supports_batch
from chronominer.common.contract import ExitCode, exit_code_for
from chronominer.common.sqlite import read
from chronominer.common.store import (
    LeaseLostError,
    Store,
    StoreError,
    Submission,
    UnitResult,
    UnitRow,
    WriterOptions,
    file_sha256,
)
from chronominer.events import NullEvents, RunEvents
from chronominer.extract.chunks import TextChunk, custom_id, extract_payload
from chronominer.extract.decide import SKIPPED_OUTCOME
from chronominer.extract.extraction import ChunkExtractor
from chronominer.extract.images import (
    PageImage,
    detect_model_type,
    load_image,
    resolve_image_settings,
)
from chronominer.extract.pdf import render_pdf_page
from chronominer.extract.units import (
    DocumentHandle,
    EntrySchema,
    UnitsFailedError,
    run_unit_stream,
)
from chronominer.llm.caller import CallEnv
from chronominer.llm.capabilities import ensure_image_support
from chronominer.llm.options import model_capabilities
from chronominer.llm.prompts import ContextError, item_context, item_context_image
from chronominer.output.render import JSON_FORMAT, find_databases
from chronominer.output.store import (
    DATABASE_NAME,
    KIND_CHUNK,
    OUTPUT_SUFFIX,
    database_path,
    open_store,
    output_path,
    read_store,
)
from chronominer.ranges.generate import chunk_texts
from chronominer.ranges.lines import read_lines
from chronominer.settings import Settings, SettingsError
from chronominer.spec import (
    CONTEXT_NONE,
    DEFAULT_CONCURRENCY,
    ContextChoice,
    ContextText,
    ImageSpec,
    ModelSpec,
    RepairSpec,
    RunSpec,
    SpecError,
    effective_spec,
    resolve_schema,
)

__all__ = [
    "PAGE_KINDS",
    "ImageSettings",
    "RepairDocument",
    "RepairPlan",
    "RepairPlanError",
    "RepairResult",
    "batch_refusal",
    "locate_document",
    "page_image",
    "plan_repair",
    "repair_extraction",
    "repair_image_settings",
    "run_repair",
]

logger = logging.getLogger(__name__)

ROLE = "extract"
LABEL = "batch-repair"

PAGE_KINDS = frozenset({"image", "pdf"})
"""Document kinds whose chunks are page images."""

_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_OUTPUT_SUFFIXES = (OUTPUT_SUFFIX, "_output.csv")


class RepairPlanError(Exception):
    """The repair cannot be planned: a usage or configuration error (exit 2)."""

    exit_code: ExitCode = ExitCode.USAGE


@dataclass(frozen=True)
class ImageSettings:
    """The image settings the repaired pages are rendered with.

    *recorded* tells whether they are the document's recorded settings; only
    then is their detail sent with each page.
    """

    config: Mapping[str, Any]
    model_type: str
    recorded: bool

    @property
    def detail(self) -> str | None:
        """Return the resolved image detail."""
        value = self.config.get("resolved_detail")
        return str(value) if value else None

    def request_options(self) -> dict[str, str]:
        """Return the per-page options of a synchronous request."""
        if not self.recorded or not self.detail:
            return {}
        return {"image_detail": self.detail}


@dataclass(frozen=True)
class RepairDocument:
    """A stored document a repair works on, with what its chunks need.

    *root* is the folder of its database; *kind* its input type (text,
    image or pdf); *output* its ``_output.json``, rendered with *formats*;
    *input_sha256* the hash of its input when it was extracted. *units* are
    the selected chunk units; *schema*, *context* and *context_image* are
    those of its extraction run; *images* the settings its pages are
    rendered with.
    """

    root: Path
    doc_id: int
    name: str
    kind: str | None
    input_path: Path | None
    image_settings: Mapping[str, Any] | None
    output: Path
    formats: tuple[str, ...]
    input_sha256: str | None = None
    lease_run_id: str | None = None
    lease_until: float | None = None
    units: tuple[UnitRow, ...] = ()
    schema: DocumentSchema = field(default_factory=DocumentSchema)
    context: str | None = None
    context_image: Path | None = None
    images: ImageSettings | None = None

    @property
    def database(self) -> Path:
        """Return the database file of the document."""
        return database_path(self.root)

    @property
    def pages(self) -> bool:
        """Whether the document's chunks are page images."""
        return self.kind in PAGE_KINDS

    @property
    def failures(self) -> list[int]:
        """Return the indices of the selected units."""
        return [unit.idx for unit in self.units]


@dataclass
class RepairResult:
    """The outcome of one repair.

    *selected* counts the units targeted; *repaired* those extracted with an
    answer afterwards and *failed* the others. *database* is the last
    database written and *outputs* the files that hold the new rendering;
    *message* explains a repair that ended before any chunk was sent.
    """

    repaired: int = 0
    failed: int = 0
    selected: int = 0
    mode: str = "sync"
    dry_run: bool = False
    database: Path | None = None
    outputs: tuple[Path, ...] = ()
    message: str | None = None


@dataclass
class RepairPlan:
    """What a repair will do, decided before anything is written.

    *documents* hold the selected units; *message* explains a plan with
    nothing to repair. *warnings* are for the front end to show. *args*
    holds the parsed command line, if any. *refused* counts the failed
    chunks left out because their input changed since extraction; a repair
    counts them as failed.
    """

    spec: RepairSpec
    settings: Settings
    documents: list[RepairDocument] = field(default_factory=list)
    message: str | None = None
    warnings: tuple[str, ...] = ()
    args: argparse.Namespace | None = None
    refused: int = 0

    @property
    def document(self) -> RepairDocument | None:
        """Return the first document of the plan, or None."""
        return self.documents[0] if self.documents else None

    @property
    def units(self) -> list[UnitRow]:
        """Return the selected units of every document."""
        return [unit for document in self.documents for unit in document.units]

    @property
    def failures(self) -> list[tuple[str, int]]:
        """Return the document name and index of each selected unit."""
        return [
            (document.name, unit.idx)
            for document in self.documents
            for unit in document.units
        ]

    @property
    def mode(self) -> str:
        """``batch`` or ``sync``."""
        return "batch" if self.spec.run.batch else "sync"

    @property
    def dry_run(self) -> bool:
        """Whether the repair only reports what it would do."""
        return self.spec.run.dry_run

    @property
    def to_repair(self) -> bool:
        """Whether any unit is selected for repair."""
        return self.message is None and bool(self.units)

    def outcome(self) -> RepairResult:
        """Return the result of the plan before any chunk is sent."""
        refused = 0 if self.dry_run else self.refused
        return RepairResult(
            failed=refused,
            selected=(len(self.units) if self.to_repair else 0) + refused,
            mode=self.mode,
            dry_run=self.dry_run and self.to_repair,
            message=self.message,
        )


def _path_key(path: Path, root: Path) -> str:
    """Return *path* as the outputs registry of *root*'s database keys it."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _roots_above(path: Path) -> Iterator[Path]:
    """Yield the folders at or above *path* that hold the tool's database."""
    start = path if path.is_dir() else path.parent
    for folder in (start, *start.parents):
        if database_path(folder).is_file():
            yield folder


def _decoded(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return dict(value) if isinstance(value, Mapping) else None


def _registered(store: Store, root: Path, doc_id: int) -> dict[str, Path]:
    """Return the document's registered outputs by format."""
    found: dict[str, Path] = {}
    for row in store.outputs(doc_id):
        path = Path(row.path)
        found.setdefault(row.format, path if path.is_absolute() else root / path)
    return found


def _document(store: Store, root: Path, doc_id: int) -> RepairDocument | None:
    row = store.document(doc_id)
    if row is None:
        return None
    registered = _registered(store, root, doc_id)
    output = registered.get(JSON_FORMAT)
    if output is None:
        output = output_path(root, row.name)
    input_path = Path(row.input_path) if row.input_path else None
    kind = row.kind
    if kind is None and input_path is not None:
        kind = _kind_of(input_path)
    return RepairDocument(
        root=root,
        doc_id=row.doc_id,
        name=row.name,
        kind=kind,
        input_path=input_path,
        image_settings=_decoded(row.image_settings),
        output=output,
        formats=tuple(registered) or (JSON_FORMAT,),
        input_sha256=row.input_sha256,
        lease_run_id=row.lease_run_id,
        lease_until=row.lease_until,
    )


def _kind_of(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in (".txt", ".md"):
        return "text"
    return "image"


def _by_output(store: Store, root: Path, path: Path) -> RepairDocument | None:
    key = _path_key(path, root)
    row = read(
        store.path,
        lambda connection: connection.execute(
            "SELECT doc_id FROM outputs WHERE path = ? ORDER BY doc_id", (key,)
        ).fetchone(),
    )
    if row is None:
        return None
    return _document(store, root, int(row[0]))


def _by_input(store: Store, root: Path, path: Path) -> RepairDocument | None:
    resolved = path.resolve()
    for row in store.documents():
        if row.input_path and Path(row.input_path).resolve() == resolved:
            return _document(store, root, row.doc_id)
    return None


def _read_store(root: Path) -> Store | None:
    try:
        return read_store(root)
    except StoreError as exc:
        raise RepairPlanError(str(exc)) from exc


def locate_document(path: Path) -> tuple[Store, RepairDocument] | None:
    """Return the store and stored document of the file *path*; never write.

    The nearest database at or above *path* whose outputs registry lists
    *path* wins; else the nearest one with a document whose input *path* is.

    Raises:
        RepairPlanError: A database on the way is refused by the store
            (versionless, newer or unreadable).
    """
    stores: list[tuple[Path, Store]] = []
    for root in _roots_above(path):
        store = _read_store(root)
        if store is None:
            continue
        found = _by_output(store, root, path)
        if found is not None:
            return store, found
        stores.append((root, store))
    for root, store in stores:
        found = _by_input(store, root, path)
        if found is not None:
            return store, found
    return None


def _file_key(name: str) -> str:
    """Return a ``--files`` name without the suffix of a rendered file."""
    base = Path(name).name
    for suffix in _OUTPUT_SUFFIXES:
        if base.endswith(suffix) and len(base) > len(suffix):
            return base[: -len(suffix)]
    return base


def _names_of(document: RepairDocument) -> set[str]:
    names = {document.name, Path(document.name).name}
    if document.input_path is not None:
        names.update({document.input_path.name, document.input_path.stem})
    return names


def _candidates(path: Path) -> list[tuple[Store, RepairDocument]]:
    """Return the stored documents ``--input`` names, in database order.

    Raises:
        RepairPlanError: Nothing records *path*, or a database is refused.
    """
    if path.is_file() and path.name != DATABASE_NAME:
        located = locate_document(path)
        if located is None:
            raise RepairPlanError(
                f"{path} is not recorded in a {DATABASE_NAME} at or above it. "
                "Repair takes a folder, a database, a file the tool rendered or "
                "the input it extracted."
            )
        return [located]
    databases = find_databases(path)
    if not databases:
        raise RepairPlanError(f"No {DATABASE_NAME} at or below {path}.")
    found: list[tuple[Store, RepairDocument]] = []
    for database in databases:
        root = database.parent
        store = _read_store(root)
        if store is None:
            continue
        for row in store.documents():
            document = _document(store, root, row.doc_id)
            if document is not None:
                found.append((store, document))
    return found


def _schema_name(argument: str) -> str:
    try:
        return resolve_schema(argument).name
    except SpecError:
        return argument


def _filtered(
    found: Sequence[tuple[Store, RepairDocument]],
    spec: RepairSpec,
    files: Sequence[str],
) -> list[tuple[Store, RepairDocument]]:
    """Keep the documents of ``--schema`` and *files*.

    Raises:
        RepairPlanError: A name of *files* matches no document.
    """
    kept = list(found)
    if spec.schema is not None:
        wanted = _schema_name(spec.schema)
        kept = [
            (store, document)
            for store, document in kept
            if document_schema(store, document.doc_id).name == wanted
        ]
    if files:
        keys = {_file_key(name): name for name in files}
        matched = {
            key
            for _store, document in kept
            for key in keys
            if key in _names_of(document)
        }
        missing = [name for key, name in keys.items() if key not in matched]
        if missing:
            raise RepairPlanError(
                f"Specified files not found: {', '.join(sorted(missing))}"
            )
        kept = [
            (store, document)
            for store, document in kept
            if _names_of(document) & set(keys)
        ]
    return kept


@dataclass
class _Selection:
    """What the unit selection found across the documents."""

    failed: int = 0
    matched: int = 0
    held: int = 0
    refused: int = 0
    warnings: list[str] = field(default_factory=list)


def _select(
    spec: RepairSpec, store: Store, document: RepairDocument, tally: _Selection
) -> tuple[UnitRow, ...]:
    """Return the selected chunk units of *document*.

    A document whose input changed since extraction is refused: its units
    are left out with a warning and counted in ``tally.refused``.
    """
    failures = [
        unit
        for unit in store.units(document.doc_id, kind=KIND_CHUNK)
        if (
            unit.outcome == SKIPPED_OUTCOME if spec.skipped else unit.status == "failed"
        )
    ]
    tally.failed += len(failures)
    if spec.indices is not None:
        wanted = set(spec.indices)
        failures = [unit for unit in failures if unit.idx in wanted]
    tally.matched += len(failures)
    blocked = set(store.blocked_units(document.doc_id))
    held = [unit for unit in failures if unit.key in blocked]
    tally.held += len(held)
    units = tuple(unit for unit in failures if unit.key not in blocked)
    source, recorded = document.input_path, document.input_sha256
    if units and source is not None and recorded is not None:
        try:
            current = file_sha256(source)
        except OSError:
            current = None
        if current is not None and current != recorded:
            tally.refused += len(units)
            tally.warnings.append(
                f"The input {source} of '{document.name}' changed since it was "
                f"extracted; its {len(units)} failed chunk(s) are not repaired. "
                "Run 'chronominer extract --force' on it to extract it again."
            )
            return ()
    return units


_HELD_TEXT = (
    "{count} selected chunk(s) await an open or unresolved batch submission; "
    "run 'chronominer batch check' to finalize or settle it before repairing "
    "them."
)


def _message(spec: RepairSpec, tally: _Selection, selected: int) -> str | None:
    if not tally.failed:
        return "No failed chunks detected matching your criteria."
    if spec.indices is not None and not tally.matched:
        return "None of the specified indices matched detected failures."
    if not selected and tally.held:
        return _HELD_TEXT.format(count=tally.held)
    if not selected:
        return "Nothing to repair: the inputs of the failed chunks changed."
    return None


def repair_image_settings(
    document: RepairDocument,
    model: ModelSpec,
    images: ImageSpec | None,
    original_max_side_px: int | None = None,
) -> tuple[ImageSettings, str | None]:
    """Return the image settings of the repaired pages and a warning, if any.

    The document's recorded settings apply when they were made for the
    repair's model; otherwise the current settings of the model do, with
    the ``images`` setting *original_max_side_px*.

    Raises:
        ValueError: A current image setting is invalid.
    """
    recorded = document.image_settings
    if recorded is not None and recorded.get("model_name", model.model) == model.model:
        config = {key: value for key, value in recorded.items() if key != "model_type"}
        model_type = str(
            recorded.get("model_type") or detect_model_type(model.provider, model.model)
        )
        return ImageSettings(config, model_type, True), None
    if recorded is None:
        warning = (
            f"No recorded image settings for '{document.name}'; using the current ones."
        )
    else:
        warning = (
            f"Repair model {model.model} differs from the recorded model "
            f"{recorded.get('model_name')}; using the current image settings."
        )
    logger.warning(warning)
    config, model_type, _section = resolve_image_settings(
        model.provider, model.model, images, original_max_side_px
    )
    return ImageSettings(config, model_type, False), warning


def _custom_id(document: RepairDocument, unit: UnitRow, kind: str) -> str:
    provenance = unit.provenance if isinstance(unit.provenance, Mapping) else {}
    stored = provenance.get("custom_id")
    if stored:
        return str(stored)
    source = document.input_path or Path(document.name)
    return custom_id(source, kind, unit.idx)


def page_image(
    document: RepairDocument, unit: UnitRow, images: ImageSettings
) -> PageImage:
    """Render the page or image of *unit* again from the recorded source.

    A page that cannot be rendered carries ``render_error``, so it is
    recorded as a failed chunk.
    """
    label = unit.label or f"page_{unit.idx + 1:04d}"
    config = dict(images.config)
    source = document.input_path
    try:
        if source is None or not source.is_file():
            raise FileNotFoundError(f"Recorded source missing: {source}")
        if source.suffix.lower() == ".pdf":
            index = unit.source_index
            page_index = index if isinstance(index, int) else unit.idx
            page = render_pdf_page(source, page_index, config, images.model_type)
        else:
            page = load_image(source, unit.idx, config, images.model_type)
    except Exception as exc:  # noqa: BLE001 - a failed page is recorded
        logger.error("Re-render failed for %s of '%s': %s", label, document.name, exc)
        return PageImage(
            index=unit.idx,
            image_name=label,
            source_file=str(source or ""),
            render_error=str(exc),
            image_provenance={"render_error": str(exc)},
            custom_id=_custom_id(document, unit, "page"),
        )
    return replace(
        page,
        index=unit.idx,
        image_name=label,
        custom_id=_custom_id(document, unit, "page"),
    )


def text_chunk(document: RepairDocument, unit: UnitRow) -> TextChunk:
    """Cut the line range of *unit* again from the recorded text.

    A chunk whose text or range cannot be read carries ``render_error``, so
    it is recorded as a failed chunk.
    """
    match = _RANGE_RE.match(str(unit.source_index or ""))
    start, end = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    chunk_id = _custom_id(document, unit, "text")
    source = document.input_path
    try:
        if match is None:
            raise ValueError(f"Chunk {unit.idx} has no line range")
        if source is None or not source.is_file():
            raise FileNotFoundError(f"Recorded source missing: {source}")
        (text,) = chunk_texts(read_lines(source), [(start, end)])
    except (OSError, ValueError) as exc:
        logger.error("Chunk %d of '%s' not rebuilt: %s", unit.idx, document.name, exc)
        return TextChunk(unit.idx, start, end, "", chunk_id, render_error=str(exc))
    return TextChunk(unit.idx, start, end, text, chunk_id)


def chunk_payload(document: RepairDocument, unit: UnitRow) -> Any:
    """Return the chunk payload of *unit*: a page image or a text chunk."""
    if document.pages:
        if document.images is None:
            raise ValueError(f"No image settings for '{document.name}'")
        return page_image(document, unit, document.images)
    return text_chunk(document, unit)


def batch_refusal(
    spec: RunSpec | RepairSpec,
    *,
    settings: Settings,
    args: argparse.Namespace | None = None,
) -> str | None:
    """Return why the extension refuses batch submission, or None.

    The reason is the message of the
    :class:`~chronominer.settings.SettingsError` that
    :func:`chronominer.ext.batch_allowed` raises, else a generic one when it
    returns False.
    """
    try:
        allowed = ext.batch_allowed(spec, settings=settings, args=args)
    except SettingsError as exc:
        return str(exc)
    if allowed:
        return None
    task = "repair" if isinstance(spec, RepairSpec) else "run"
    return f"Batch submission is not available for this {task}."


def _check_access(
    spec: RepairSpec,
    settings: Settings,
    args: argparse.Namespace | None,
    pages: bool,
) -> str | None:
    """Require the model's key variable, image input and, for batch, a batch API.

    A dry run plans without the key variable: the message naming an unset
    one is returned for the plan's warnings.

    Raises:
        RepairPlanError: The repair cannot reach its provider, or the model
            does not accept the page images it resends.
    """
    model = spec.model
    if spec.run.batch:
        if not supports_batch(model.provider):
            raise RepairPlanError(
                f"Batch repair needs a provider with a batch API (openai, "
                f"anthropic, google), not {model.provider}"
            )
        refusal = batch_refusal(spec, settings=settings, args=args)
        if refusal is not None:
            raise RepairPlanError(refusal)
    try:
        variables = ext.key_variables(
            model.provider,
            model.endpoint,
            kind="batch" if spec.run.batch else "chat",
            settings=settings,
            args=args,
        )
    except SettingsError as exc:
        raise RepairPlanError(str(exc)) from exc
    missing: str | None = None
    if variables and not any(os.environ.get(name) for name in variables):
        missing = (
            f"The API key variable {variables[0]} of provider {model.provider} is "
            "not set; repair needs it."
        )
        if not spec.run.dry_run:
            raise RepairPlanError(missing)
    if pages:
        try:
            ensure_image_support(model.model, model_capabilities(model, settings))
        except ValueError as exc:
            raise RepairPlanError(str(exc)) from exc
    return missing


def _run_choices(
    store: Store, doc_id: int
) -> tuple[ContextText, ContextText, ContextChoice]:
    """Return the context choices of the document's extraction run.

    The choices are ``--context``, ``--fallback-context`` and
    ``--context-image``, in that order. The latest ``extract`` run that
    wrote a unit of the document counts; without one, the context is looked
    up and no context image is sent.
    """

    def query(connection: sqlite3.Connection) -> Any:
        return connection.execute(
            "SELECT spec FROM runs WHERE command = 'extract' AND run_id IN "
            "(SELECT run_id FROM units WHERE doc_id = ? UNION "
            "SELECT first_run_id FROM documents WHERE doc_id = ?) "
            "ORDER BY started_at DESC LIMIT 1",
            (doc_id, doc_id),
        ).fetchone()

    row = read(store.path, query)
    spec = _decoded(row[0]) if row is not None else None
    if spec is None:
        return None, None, CONTEXT_NONE
    image = _value(spec, "prompt.context_image")
    image_choice: ContextChoice = None
    if image == CONTEXT_NONE:
        image_choice = CONTEXT_NONE
    elif image is not None:
        image_choice = Path(image)
    return (
        _value(spec, "prompt.context"),
        _value(spec, "prompt.fallback_context"),
        image_choice,
    )


def _value(spec: Mapping[str, Any], key: str) -> str | None:
    """Return the stored value of spec field *key* as text, or None."""
    entry = spec.get(key)
    value = entry.get("value") if isinstance(entry, Mapping) else None
    return None if value is None else str(value)


def _schema_of(
    store: Store, document: RepairDocument, spec: RepairSpec
) -> DocumentSchema:
    """Return the document's schema: stored, else by name, else ``--schema``.

    Raises:
        RepairPlanError: No schema can be found.
    """
    stored = document_schema(store, document.doc_id)
    if stored.schema is not None and stored.name:
        return stored
    for argument in (stored.name, spec.schema):
        if not argument:
            continue
        try:
            resolved = resolve_schema(argument)
        except SpecError:
            continue
        return DocumentSchema(resolved.name, resolved.schema)
    raise RepairPlanError(
        f"The schema of '{document.name}' is not stored; name it with --schema."
    )


def _prepare(
    store: Store, document: RepairDocument, spec: RepairSpec, settings: Settings
) -> tuple[RepairDocument, str | None]:
    """Return the document with its schema, prompt inputs and image settings.

    Raises:
        RepairPlanError: The schema, context or image settings cannot be
            resolved.
    """
    schema = _schema_of(store, document, spec)
    context_choice, fallback, image_choice = _run_choices(store, document.doc_id)
    anchor = document.input_path or document.output
    try:
        context = item_context(anchor, context_choice, fallback=fallback)
        context_image = item_context_image(anchor, image_choice)
    except ContextError as exc:
        raise RepairPlanError(str(exc)) from exc
    images: ImageSettings | None = None
    warning: str | None = None
    if document.pages:
        try:
            images, warning = repair_image_settings(
                document,
                spec.model,
                spec.images,
                settings.images.original_max_side_px,
            )
        except ValueError as exc:
            raise RepairPlanError(str(exc)) from exc
    prepared = replace(
        document,
        schema=schema,
        context=context,
        context_image=context_image,
        images=images,
    )
    return prepared, warning


def _lease_warning(document: RepairDocument, now: float) -> str | None:
    holder, until = document.lease_run_id, document.lease_until
    if holder is None or until is None or until <= now:
        return None
    return (
        f"'{document.name}' is leased by run {holder}; the repair is refused "
        "while that run holds it."
    )


def plan_repair(
    spec: RepairSpec,
    settings: Settings,
    *,
    args: argparse.Namespace | None = None,
    files: Sequence[str] = (),
) -> RepairPlan:
    """Find the documents of *spec*, select their failed chunks, check the repair.

    Writes nothing. Schema, context, image settings and provider access are
    checked only when a chunk is selected, dry runs included. *files* (else
    the spec's ``files``, when it has them) keeps the documents whose name,
    input file or rendered file names match.
    *args* holds the parsed command line, passed to the extension's batch
    consent and key variables and kept on the plan for its batch clients.

    Raises:
        RepairPlanError: The input is missing or not recorded in a store,
            a database is refused, a name of *files* matches no document,
            the schema, context or image settings cannot be resolved, or the
            provider cannot be reached (no key, no batch API).
    """
    path = spec.input.expanduser().resolve()
    if not path.exists():
        raise RepairPlanError(f"Input path does not exist: {path}")
    names = tuple(files) or tuple(getattr(spec, "files", None) or ())
    found = _filtered(_candidates(path), spec, names)
    tally = _Selection()
    selected: list[tuple[Store, RepairDocument]] = []
    for store, document in found:
        units = _select(spec, store, document, tally)
        if units:
            selected.append((store, replace(document, units=units)))
    message = _message(spec, tally, len(selected))
    warnings = list(tally.warnings)
    if message is None and tally.held:
        warnings.append(_HELD_TEXT.format(count=tally.held) + " They are left out.")
    plan = RepairPlan(spec, settings, message=message, args=args, refused=tally.refused)
    if message is not None:
        plan.warnings = tuple(warnings)
        return plan
    now = time.time()
    for store, document in selected:
        prepared, warning = _prepare(store, document, spec, settings)
        for text in (warning, _lease_warning(prepared, now)):
            if text is not None:
                warnings.append(text)
        plan.documents.append(prepared)
    missing = _check_access(
        spec, settings, args, any(document.pages for document in plan.documents)
    )
    if missing is not None:
        warnings.append(missing)
    plan.warnings = tuple(warnings)
    return plan


class _DetailedExtractor:
    """Sends every page with the recorded image detail of the repair."""

    def __init__(self, inner: ChunkExtractor, options: Mapping[str, str]) -> None:
        self.inner = inner
        self.options = dict(options)

    async def extract_text(self, text: str, *, label: str | None = None) -> Any:
        return await self.inner.extract_text(text, label=label)

    async def extract_image_from_base64(
        self, image_base64: str, mime_type: str, *, label: str | None = None
    ) -> Any:
        return await self.inner.extract_image_from_base64(
            image_base64, mime_type, label=label, **self.options
        )


def _source_index(payload: Any) -> int | str:
    if isinstance(payload, TextChunk):
        return payload.source_index
    if isinstance(payload.page_index, int):
        return int(payload.page_index)
    return str(payload.image_name)


async def _stream(payloads: Sequence[Any]) -> AsyncIterator[Any]:
    for payload in payloads:
        yield payload


@dataclass
class _Session:
    """One repair of one leased document."""

    plan: RepairPlan
    document: RepairDocument
    store: Store
    handle: DocumentHandle
    events: RunEvents
    clients: BatchClients | None = None

    @property
    def idxs(self) -> list[int]:
        return [unit.idx for unit in self.document.units]

    def warn(self, message: str) -> None:
        logger.warning(message)
        self.events.warning(message)

    async def sync(self, payloads: Sequence[Any]) -> bool:
        """Extract *payloads* one call each; return True (units were written)."""
        plan, document = self.plan, self.document
        await self.handle.reset(self.idxs)
        extractor = ChunkExtractor(
            plan.spec.model,
            CallEnv.create(plan.settings, self.events),
            schema=document.schema.schema or {},
            schema_name=document.schema.name or "",
            images=plan.spec.images,
            context=document.context,
            role=ROLE,
        )
        options = document.images.request_options() if document.images else {}
        client: Any = _DetailedExtractor(extractor, options) if options else extractor
        try:
            extractor.update_context_image(document.context_image)
            await run_unit_stream(
                self.handle,
                _stream(payloads),
                client,
                extract_payload,
                concurrency=DEFAULT_CONCURRENCY,
                delay=plan.settings.concurrency.delay_between_tasks,
                entries=EntrySchema(document.schema.schema, document.schema.name),
            )
        except UnitsFailedError as exc:
            logger.info("Repair of '%s': %s", self.handle.name, exc)
        finally:
            await extractor.close()
        return True

    async def _record_chunks(self, payloads: Sequence[Any]) -> None:
        """Return the submitted units to pending with their new provenance."""
        handle = self.handle
        await handle.reset(self.idxs)
        for payload in payloads:
            provenance = payload.provenance()
            if not provenance:
                continue
            await self.store.write_unit(
                handle.lease,
                UnitResult(
                    kind=handle.kind,
                    idx=payload.index,
                    status="pending",
                    model=handle.model,
                    provider=handle.provider,
                    label=payload.image_name,
                    source_index=_source_index(payload),
                    provenance=provenance,
                    attempts=0,
                ),
            )

    async def batch(self, payloads: Sequence[Any]) -> list[Submission] | None:
        """Submit *payloads*, wait for the batches and finalize them.

        Returns the finalized submissions, or None when no unit was written:
        a refused submission leaves the units as they were, an unresolved or
        unfinished one leaves them to ``batch check``.
        """
        plan, handle, document = self.plan, self.handle, self.document
        clients = self.clients = BatchClients(plan.settings, ROLE, args=plan.args)
        images = document.images
        try:
            body = build_plan(
                plan.spec.model,
                plan.settings,
                schema=document.schema.schema or {},
                context=document.context,
                context_image=document.context_image,
                detail=images.detail if images else None,
                images=plan.spec.images,
            )
        except (OSError, ValueError) as exc:
            self.warn(f"Cannot build the repair batch requests: {exc}")
            return None
        detail = images.detail if images is not None and images.recorded else None
        units = [
            UnitRequest(
                handle.key(payload.index), chunk_request(payload, detail=detail)
            )
            for payload in payloads
        ]
        try:
            outcome = await submit_units(
                self.store, units, plan=body, clients=clients, label=LABEL
            )
        except BatchSubmissionError as exc:
            self.warn(f"Failed to submit the repair batch: {exc}")
            return None
        if outcome.status == "refused":
            self.warn(
                "The repair batch was refused "
                f"({outcome.detail or 'no detail'}); the chunks are left as they "
                "were."
            )
            return None
        if outcome.status == "unresolved":
            self.warn(
                f"The repair batch is unresolved ({outcome.detail or 'no detail'}); "
                "its chunks stay blocked until 'chronominer batch check' settles it."
            )
            return None
        await self._record_chunks(payloads)
        waited = await wait_for_submissions(
            self.store, outcome.submission_ids, clients, self.events
        )
        ids = ", ".join(outcome.submission_ids)
        if waited.error is not None or waited.running:
            reason = (
                f"their status could not be read ({waited.error})"
                if waited.error is not None
                else "they did not finish within the timeout"
            )
            self.warn(
                f"The repair batches were not finalized: {reason}. Submissions "
                f"{ids} stay in the store; run 'chronominer batch check' to "
                "finalize them."
            )
            return None
        finalized: list[Submission] = []
        env = CallEnv.create(plan.settings, self.events)
        for submission_id in outcome.submission_ids:
            submission = self.store.submission(submission_id)
            result = await finalize_submission(self.store, submission, clients)
            if result.outcome == FINALIZED_OUTCOME:
                env.record(ROLE, result.usage)
                finalized.append(submission)
            else:
                self.warn(
                    f"Repair submission {submission_id} not finalized "
                    f"({result.outcome}): {result.detail}"
                )
        return finalized or None

    async def render(self) -> tuple[Path, ...]:
        outputs, warnings = await render_documents(
            self.store, [self.document.doc_id], self.document.formats
        )
        for message in warnings:
            self.warn(message)
        return tuple(outputs)

    def count(self) -> int:
        """Return how many selected units hold an answer now."""
        wanted = set(self.idxs)
        return sum(
            1
            for unit in self.store.units(self.handle.doc_id, kind=self.handle.kind)
            if unit.idx in wanted
            and unit.status == "done"
            and unit.outcome != SKIPPED_OUTCOME
        )


def _writer_options(events: RunEvents) -> WriterOptions:
    def on_busy(path: Path, waited: float) -> None:
        events.warning(
            f"The database {path} is busy (another program holds it open); "
            f"waited {waited:.0f} s. Results are kept in memory and written "
            "once it is free."
        )

    return WriterOptions(on_busy=on_busy)


async def _repair_document(
    store: Store,
    plan: RepairPlan,
    document: RepairDocument,
    result: RepairResult,
    events: RunEvents,
) -> None:
    """Repair the selected units of one document; add to *result*."""
    selected = len(document.units)
    lease = await store.acquire(document.doc_id)
    if lease is None:
        message = (
            f"Skipping '{document.name}': another run holds its document in "
            f"{store.path}; repair it once that run has ended."
        )
        events.warning(message)
        await store.add_note(message, doc_id=document.doc_id)
        result.message = message
        result.failed += selected
        return
    blocked = set(store.blocked_units(document.doc_id))
    units = tuple(unit for unit in document.units if unit.key not in blocked)
    if len(units) < selected:
        events.warning(
            f"{selected - len(units)} chunk(s) of '{document.name}' were submitted "
            "to a batch since the repair was planned and are left out; run "
            "'chronominer batch check'."
        )
    model = plan.spec.model
    session = _Session(
        plan,
        replace(document, units=units),
        store,
        DocumentHandle(
            store,
            document.doc_id,
            document.name,
            lease,
            model=model.model,
            provider=model.provider,
        ),
        events,
    )
    if not units:
        result.failed += selected
        return
    payloads = [chunk_payload(session.document, unit) for unit in units]
    repaired = 0
    try:
        if plan.spec.run.batch:
            finalized = await session.batch(payloads)
            written = finalized is not None
        else:
            finalized = None
            written = await session.sync(payloads)
        if written:
            result.outputs = (*result.outputs, *await session.render())
        if session.clients is not None:
            for submission in finalized or ():
                await delete_provider_files(session.clients, submission)
        repaired = session.count()
    except LeaseLostError as exc:
        session.warn(
            f"The lease on '{document.name}' was lost; the repair stopped: {exc}"
        )
    result.repaired += repaired
    result.failed += selected - repaired


async def _repair_database(
    root: Path,
    plan: RepairPlan,
    documents: Sequence[RepairDocument],
    result: RepairResult,
    events: RunEvents,
) -> None:
    async with open_store(root, writer=_writer_options(events)) as store:
        await store.begin_run("repair", spec=effective_spec(plan.spec))
        exit_code = int(ExitCode.FAILURE)
        failed = result.failed
        try:
            for submission_id in await store.mark_stale_submissions():
                events.warning(
                    f"Submission {submission_id} in {store.path} was left without "
                    "a provider batch id; it is unresolved. Run 'chronominer "
                    "batch check' to reconcile it."
                )
            async with store.heartbeat():
                for document in documents:
                    await _repair_document(store, plan, document, result, events)
            exit_code = int(exit_code_for(failed=result.failed - failed))
        finally:
            try:
                await store.end_run(exit_code)
            except sqlite3.Error as exc:
                logger.warning("Could not record the run end in %s: %s", store, exc)


def _by_root(documents: Iterable[RepairDocument]) -> dict[Path, list[RepairDocument]]:
    grouped: dict[Path, list[RepairDocument]] = {}
    for document in documents:
        grouped.setdefault(document.root, []).append(document)
    return grouped


async def run_repair(plan: RepairPlan, events: RunEvents | None = None) -> RepairResult:
    """Repair the units *plan* selected; a plan with nothing to do returns its result.

    Errors raised here are failures of the repair, not usage errors.
    """
    result = plan.outcome()
    if not plan.to_repair or plan.dry_run:
        return result
    sink = events if events is not None else NullEvents()
    for root, documents in _by_root(plan.documents).items():
        result.database = database_path(root)
        await _repair_database(root, plan, documents, result, sink)
    return result


async def repair_extraction(
    spec: RepairSpec,
    settings: Settings,
    events: RunEvents | None = None,
    *,
    files: Sequence[str] = (),
) -> RepairResult:
    """Plan and run the repair of the documents of *spec*.

    Raises:
        RepairPlanError: The repair cannot be planned (see :func:`plan_repair`).
    """
    plan = plan_repair(spec, settings, files=files)
    sink = events if events is not None else NullEvents()
    for warning in plan.warnings:
        sink.warning(warning)
    return await run_repair(plan, sink)
