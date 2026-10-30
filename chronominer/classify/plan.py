"""Plan a classify command without model calls or writes.

:func:`plan_classify` resolves the classifier (``--classifier``: a file path,
a file name in the repository's ``classifiers/`` folder without ``.json``, or
the ``name`` of a classifier there) and checks that its unit is the
command's, then finds the documents to classify and gives each a state:

- ``classify chunks`` and ``classify documents`` discover the input items as
  ``extract`` does and locate each one's document in the store of its output
  root. An item whose stored document records another input content is
  ``refused``; a text item under ``--chunking line-ranges`` that the store
  does not hold needs its ``{stem}_line_ranges.txt`` (else exit 2).
- ``classify entries`` takes the documents of the stores at or below its
  input whose run metadata names the classifier's schema. A document with
  pending chunks is ``skipped`` (its extraction is not finished), one that
  another live run leases ``blocked``.

A dry run plans without the decision model's key variable and lists an unset
one as a warning; otherwise an unset key is a configuration error.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from chronominer import ext
from chronominer.common.classifiers import (
    Classifier,
    load_classifier,
    resolve_classifier,
)
from chronominer.common.decisions import dialect_for
from chronominer.common.options import Source
from chronominer.common.selection import PageRange, parse_page_range
from chronominer.common.sqlite import read
from chronominer.common.store import DocumentRow, StoreError, UnitRow, file_sha256
from chronominer.discovery import SourceItem, item_paths
from chronominer.output.entries import entry_fields
from chronominer.output.render import find_databases
from chronominer.output.store import (
    DATABASE_NAME,
    KIND_CHUNK,
    document_metadata,
    document_name,
    lease_reason,
    read_store,
)
from chronominer.ranges.lines import ranges_path
from chronominer.run import (
    ConfigurationError,
    InputError,
    MissingKey,
    NoItemsError,
    StoreRefusedError,
    discover_inputs,
)
from chronominer.settings import Settings, SettingsError
from chronominer.spec import (
    CLASSIFIERS_DIR,
    ClassifyEntriesSpec,
    ClassifyItemsSpec,
    ClassifySpec,
    ImageSpec,
    SpecError,
    effective_spec,
    resolve_schema,
)

__all__ = [
    "BLOCKED",
    "FINISHED",
    "READY",
    "REFUSED",
    "ROLE",
    "SKIPPED",
    "ClassifyPlan",
    "ClassifyTarget",
    "image_conflicts",
    "plan_classify",
    "stored_images",
]

ROLE: Final = "classify"
"""The usage role of the classify commands' decision-model calls."""

READY: Final = "ready"
REFUSED: Final = "refused"
BLOCKED: Final = "blocked"
SKIPPED: Final = "skipped"

_COMMANDS: Final = {
    "chunk": "classify chunks",
    "entry": "classify entries",
    "document": "classify documents",
}
FINISHED: Final = frozenset({"complete", "failed"})
"""Document statuses without pending chunks: the extraction is finished."""


@dataclass(frozen=True)
class ClassifyTarget:
    """One document to classify and its state.

    *name* is the input's file name (the stored document's name for
    entries), *document* the store's document name in output root *root*
    and *kind* the input type. *stored* holds the chunk units the store
    holds for the document and *image_settings* the image-settings record
    of its stored pages, both empty when the store does not hold it.
    *seen_doc_id* is the stored document's id at planning (None without
    one); unleased writes refuse to land once it is gone or replaced.
    """

    name: str
    document: str
    root: Path
    kind: str
    state: str = READY
    reason: str = ""
    path: Path | None = None
    input_sha256: str | None = None
    doc_id: int | None = None
    stored: tuple[UnitRow, ...] = ()
    image_settings: Mapping[str, Any] | None = None
    seen_doc_id: int | None = None


@dataclass(frozen=True)
class ClassifyPlan:
    """What a classify command will do; *targets* in order, with their states."""

    spec: ClassifySpec
    classifier: Classifier
    classifier_path: Path
    input_path: Path
    targets: tuple[ClassifyTarget, ...]
    settings: Settings = field(default_factory=Settings)
    sources: Mapping[str, Source] = field(default_factory=dict)
    pages: PageRange | None = None
    warnings: tuple[str, ...] = ()
    missing_keys: tuple[MissingKey, ...] = ()
    schema: str | None = None
    """The schema name an entry classifier's documents were extracted with."""

    @property
    def unit(self) -> str:
        """The unit the classifier judges: chunk, entry or document."""
        return self.classifier.unit

    @property
    def command(self) -> str:
        """The command name recorded with the run."""
        return _COMMANDS[self.unit]

    @property
    def dry_run(self) -> bool:
        """Whether the plan is only reported."""
        return self.spec.run.dry_run

    @property
    def force(self) -> bool:
        """Whether stored answers are asked again instead of reused."""
        return self.spec.classify.force

    def having(self, state: str) -> tuple[ClassifyTarget, ...]:
        """Return the targets in *state*."""
        return tuple(target for target in self.targets if target.state == state)

    def effective_spec(self) -> dict[str, dict[str, Any]]:
        """Return ``{field: {"value": ..., "source": ...}}`` for every field."""
        return effective_spec(self.spec, self.sources)


def _classifier(value: str, unit: str) -> tuple[Classifier, Path]:
    """Return the classifier *value* names and its file.

    Raises:
        ConfigurationError: No classifier matches, the file is invalid, or
            it judges another unit than *unit*.
    """
    try:
        path = resolve_classifier(value, CLASSIFIERS_DIR)
        classifier = load_classifier(path)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc
    if classifier.unit != unit:
        command = _COMMANDS.get(classifier.unit)
        hint = f"; use '{command}'" if command else ""
        raise ConfigurationError(
            f"classifier {classifier.name!r} judges unit {classifier.unit}, but "
            f"'{_COMMANDS[unit]}' classifies unit {unit}{hint}"
        )
    return classifier, path


def _check_classifier(classifier: Classifier, settings: Settings) -> str | None:
    """Check the classifier against the tool; return its schema's name, if any.

    Raises:
        ConfigurationError: A document classifier selects more pages than a
            request carries, or an entry classifier names fields its schema
            lacks.
    """
    budget = settings.decisions.pages_per_request
    selection = classifier.selection
    if selection is not None and selection.first_pages > budget:
        raise ConfigurationError(
            f"classifier {classifier.name!r} selects {selection.first_pages} "
            f"pages; a request carries at most decisions.pages_per_request "
            f"({budget})"
        )
    if classifier.schema is None:
        return None
    try:
        schema = resolve_schema(classifier.schema)
    except SpecError:
        return classifier.schema
    known = {entry.name for entry in entry_fields(schema.schema)}
    missing = [name for name in classifier.fields or () if name not in known]
    if missing:
        raise ConfigurationError(
            f"classifier {classifier.name!r} sends fields that schema "
            f"{schema.name} lacks: {', '.join(missing)}"
        )
    return schema.name


def _missing_key(
    spec: ClassifySpec, settings: Settings, args: argparse.Namespace | None
) -> MissingKey | None:
    """Return the unset key variable of the decision model, if any.

    Raises:
        ConfigurationError: The settings name no key variable for it.
    """
    choice = spec.classify
    dialect = dialect_for(choice.model, choice.provider)
    try:
        variables = ext.key_variables(
            dialect, kind="decision", settings=settings, args=args
        )
    except SettingsError as exc:
        raise ConfigurationError(str(exc)) from exc
    if variables and not any(os.environ.get(name) for name in variables):
        return MissingKey(variables[0], dialect, "classify")
    return None


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve()


def _input(path: Path) -> Path:
    path = _absolute(path)
    if not path.exists():
        raise InputError(f"Input path does not exist: {path}")
    return path


def _recorded(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return dict(value) if isinstance(value, Mapping) else None


_IMAGE_KEYS: Final = {
    "dpi": ("--dpi", "target_dpi"),
    "detail": ("--image-detail", "resolved_detail"),
    "format": ("--image-format", "payload_format"),
    "jpeg_quality": ("--jpeg-quality", "jpeg_quality"),
    "grayscale": ("--grayscale", "grayscale_conversion"),
}
"""Each image option: its flag and its key in an image-settings record."""


def stored_images(recorded: Mapping[str, Any]) -> ImageSpec:
    """Return the image options a document's stored pages were rendered with."""
    values = {name: recorded.get(key) for name, (_flag, key) in _IMAGE_KEYS.items()}
    if values["detail"] is None:
        values["detail"] = recorded.get("detail")
    return ImageSpec(**values)


def image_conflicts(given: ImageSpec, recorded: Mapping[str, Any]) -> list[str]:
    """Return the given image options that differ from the stored pages' ones."""
    stored = stored_images(recorded)
    found = []
    for name, (flag, _key) in _IMAGE_KEYS.items():
        mine, theirs = getattr(given, name), getattr(stored, name)
        if mine is not None and theirs is not None and mine != theirs:
            found.append(f"{flag} {mine} (stored: {theirs})")
    return found


def _item_target(
    spec: ClassifyItemsSpec, item: SourceItem, input_path: Path
) -> ClassifyTarget:
    """Return the target of one input item, read from its root's store.

    Raises:
        StoreRefusedError: The root's database is refused.
        ConfigurationError: A text item under ``line-ranges`` that the store
            does not hold has no line-ranges file.
    """
    output_dir = spec.output.directory
    output_dir = _absolute(output_dir) if output_dir is not None else None
    output = item_paths(
        item.path,
        output_dir=output_dir,
        layout=spec.output.layout,
        input_root=input_path,
    ).output
    root = output_dir if output_dir is not None else item.path.parent
    name = document_name(output, root)
    sha = file_sha256(item.path)
    target = ClassifyTarget(
        item.name, name, root, item.type, path=item.path, input_sha256=sha
    )
    try:
        store = read_store(root)
    except StoreError as exc:
        raise StoreRefusedError(str(exc)) from exc
    row = store.document(name) if store is not None else None
    if store is not None and row is not None:
        if row.input_sha256 is not None and row.input_sha256 != sha:
            return _refused(target, row.input_sha256)
        stored = tuple(store.units(row.doc_id, kind=KIND_CHUNK))
        pages = row.units_done + row.units_failed > 0
        recorded = _recorded(row.image_settings) if pages else None
        target = ClassifyTarget(
            item.name,
            name,
            root,
            item.type,
            path=item.path,
            input_sha256=sha,
            stored=stored,
            image_settings=recorded,
            seen_doc_id=row.doc_id,
        )
        conflicts = image_conflicts(spec.images, recorded) if recorded else []
        if item.type != "text" and conflicts:
            reason = (
                f"{name}: the image options {', '.join(conflicts)} differ from "
                "the settings its stored pages were rendered with; drop them to "
                "classify the stored pages"
            )
            return replace(target, state=REFUSED, reason=reason, stored=())
    needs_sidecar = (
        item.type == "text"
        and spec.chunking.mode == "line-ranges"
        and not target.stored
        and not ranges_path(item.path).is_file()
    )
    if needs_sidecar:
        raise ConfigurationError(
            f"{ranges_path(item.path).name} is missing beside {item.name}; write "
            "it with 'chronominer ranges generate' or 'chronominer ranges "
            "adjust' first, or classify with --chunking auto"
        )
    return target


def _refused(target: ClassifyTarget, stored_sha256: str) -> ClassifyTarget:
    reason = (
        f"{target.document}: the store holds this document with input SHA-256 "
        f"{stored_sha256[:12]}..., the classified file has "
        f"{str(target.input_sha256)[:12]}...; process the changed file first"
    )
    return ClassifyTarget(
        target.name,
        target.document,
        target.root,
        target.kind,
        state=REFUSED,
        reason=reason,
        path=target.path,
        input_sha256=target.input_sha256,
    )


def _item_targets(
    spec: ClassifyItemsSpec, input_path: Path
) -> tuple[list[ClassifyTarget], tuple[str, ...]]:
    items, warnings = discover_inputs(spec.input, input_path)
    targets: list[ClassifyTarget] = []
    seen: dict[tuple[Path, str], Path] = {}
    for item in items:
        target = _item_target(spec, item, input_path)
        other = seen.setdefault((target.root.resolve(), target.document), item.path)
        if other != item.path:
            raise InputError(
                f"{other} and {item.path} share the document {target.document}; "
                "rename one or classify them separately.",
                items_total=len(items),
            )
        targets.append(target)
    return targets, warnings


def _schema_names(connection: sqlite3.Connection) -> dict[int, Any]:
    """Return the ``schema_name`` of each document's run metadata, by doc_id."""
    ids = [int(row[0]) for row in connection.execute("SELECT doc_id FROM documents")]
    return {
        doc_id: document_metadata(connection, doc_id)[0].get("schema_name")
        for doc_id in ids
    }


def _entry_targets(
    input_path: Path, schema: str | None, now: float
) -> list[ClassifyTarget]:
    """Return the documents of the stores at or below *input_path* of *schema*.

    Raises:
        NoItemsError: No store, or no document extracted with the schema.
        StoreRefusedError: A database is refused.
    """
    if input_path.is_file() and input_path.name != DATABASE_NAME:
        raise InputError(f"{input_path} is not a {DATABASE_NAME} file")
    databases = find_databases(input_path)
    if not databases:
        raise NoItemsError(f"No {DATABASE_NAME} at or below {input_path}")
    targets: list[ClassifyTarget] = []
    for database in databases:
        root = database.parent
        try:
            store = read_store(root)
        except StoreError as exc:
            raise StoreRefusedError(str(exc)) from exc
        if store is None:
            continue
        schemas = read(database, _schema_names)
        for row in store.documents():
            if schemas.get(row.doc_id) != schema:
                continue
            target = ClassifyTarget(
                row.name,
                row.name,
                root,
                row.kind or "text",
                path=Path(row.input_path) if row.input_path else None,
                input_sha256=row.input_sha256,
                doc_id=row.doc_id,
            )
            targets.append(_entry_state(target, row, now))
    if not targets:
        raise NoItemsError(
            f"No document below {input_path} was extracted with schema {schema}"
        )
    return targets


def _entry_state(
    target: ClassifyTarget, row: DocumentRow, now: float
) -> ClassifyTarget:
    """Return *target* skipped while it has pending chunks, blocked while leased."""
    state, reason = READY, ""
    if row.status not in FINISHED:
        state, reason = SKIPPED, f"its extraction is not finished ({row.status})"
    elif row.lease_run_id is not None and (row.lease_until or 0) > now:
        state, reason = BLOCKED, lease_reason(row.lease_run_id, row.lease_until)
    if state == READY:
        return target
    return ClassifyTarget(
        target.name,
        target.document,
        target.root,
        target.kind,
        state=state,
        reason=reason,
        path=target.path,
        input_sha256=target.input_sha256,
        doc_id=target.doc_id,
    )


def plan_classify(
    spec: ClassifySpec,
    settings: Settings,
    *,
    sources: Mapping[str, Source] | None = None,
    args: argparse.Namespace | None = None,
    require_keys: bool | None = None,
    now: float | None = None,
) -> ClassifyPlan:
    """Resolve the classifier and find the documents to classify; write nothing.

    With *require_keys* (by default unless the spec is a dry run) an unset
    key variable of the decision model is a :class:`ConfigurationError`;
    otherwise the plan lists it. *now* is the time leases are judged at.

    Raises:
        PlanError: The command cannot proceed; the subclass names the reason
            and ``exit_code`` the exit code (2).
    """
    unit = spec.unit if isinstance(spec, ClassifyItemsSpec) else "entry"
    classifier, path = _classifier(spec.classify.classifier, unit)
    schema = _check_classifier(classifier, settings)
    pages: PageRange | None = None
    warnings: tuple[str, ...] = ()
    if isinstance(spec, ClassifyEntriesSpec):
        input_path = _input(spec.input)
        moment = time.time() if now is None else now
        targets = _entry_targets(input_path, schema, moment)
    else:
        input_path = _input(spec.input.path)
        targets, warnings = _item_targets(spec, input_path)
        try:
            pages = parse_page_range(spec.input.pages) if spec.input.pages else None
        except ValueError as exc:
            raise ConfigurationError(str(exc), items_total=len(targets)) from exc
    missing = _missing_key(spec, settings, args)
    if require_keys is None:
        require_keys = not spec.run.dry_run
    if missing is not None and require_keys:
        raise ConfigurationError(missing.message, items_total=len(targets))
    if missing is not None and spec.run.dry_run:
        warnings += (missing.message,)
    return ClassifyPlan(
        spec=spec,
        classifier=classifier,
        classifier_path=path,
        input_path=input_path,
        targets=tuple(targets),
        settings=settings,
        sources=dict(sources or {}),
        pages=pages,
        warnings=warnings,
        missing_keys=(missing,) if missing is not None else (),
        schema=schema,
    )
