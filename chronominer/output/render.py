"""The rendered views of a document, written from the store.

Formats: ``json`` (``{stem}_output.json``, always written once a chunk is
done or failed; :mod:`chronominer.output.output_json`), ``csv``
(``{stem}_output.csv`` beside it, on ``--formats csv``;
:mod:`chronominer.output.csv_view`) and ``line_ranges`` (the
``{stem}_line_ranges.txt`` input file of a text document, re-rendered from
its range units; :mod:`chronominer.output.line_ranges`). Users choose only
among :data:`FORMAT_CHOICES`.

:func:`render_item` and :func:`render_ranges` write the files through the
store's render driver: each file is replaced atomically under the
document's lease and recorded in ``outputs``; a file changed outside the
tool (a hand edit, or a downstream tool's metadata write-back into
``_output.json``) is kept unless forced, and a file that cannot be replaced
is marked stale while the units stay stored. :func:`render_database`
renders every document of a database that way (the ``render`` subcommand)
and :func:`preview_database` reports what it would write without writing.
"""

from __future__ import annotations

import contextlib
import hashlib
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from chronominer.common.sqlite import read
from chronominer.common.store import (
    DocumentRow,
    Lease,
    LeaseLostError,
    OutputRow,
    Renderer,
    RenderReport,
    RenderTarget,
    Store,
    file_sha256,
)
from chronominer.output.csv_view import CSV_FORMAT, render_csv
from chronominer.output.line_ranges import (
    LINE_RANGES_FORMAT,
    line_ranges_target,
    render_line_ranges,
)
from chronominer.output.output_json import render_output_json
from chronominer.output.store import (
    DATABASE_NAME,
    KIND_CHUNK,
    KIND_RANGE,
    open_store,
    output_path,
    read_only_connection,
    read_store,
)

__all__ = [
    "FORMAT_CHOICES",
    "JSON_FORMAT",
    "OUTPUT_FORMATS",
    "RENDERERS",
    "DocumentRender",
    "document_targets",
    "find_databases",
    "format_path",
    "output_formats",
    "preview_database",
    "render_database",
    "render_item",
    "render_ranges",
    "render_targets",
    "rendered_ranges",
    "rendered_units",
]

JSON_FORMAT: Final = "json"
OUTPUT_FORMATS: Final = (JSON_FORMAT, CSV_FORMAT)
"""The formats rendered from a document's chunks, in render order."""
FORMAT_CHOICES: Final = (CSV_FORMAT,)
"""The formats ``--formats`` accepts; ``_output.json`` is always written."""

RENDERERS: Final[Mapping[str, Renderer]] = {
    JSON_FORMAT: render_output_json,
    CSV_FORMAT: render_csv,
    LINE_RANGES_FORMAT: render_line_ranges,
}

_NOT_PENDING = (
    "SELECT count(*) FROM units WHERE doc_id = ? AND kind = ? AND status != 'pending'"
)


def output_formats(formats: Sequence[str]) -> tuple[str, ...]:
    """Return the chunk formats to render: json, and csv when in *formats*."""
    return tuple(
        name for name in OUTPUT_FORMATS if name == JSON_FORMAT or name in formats
    )


def format_path(output: Path, output_format: str) -> Path:
    """Return the file of *output_format* for the item's ``_output.json`` *output*."""
    if output_format == CSV_FORMAT:
        return output.with_suffix(".csv")
    return output


def _count(store: Store, doc_id: int, kind: str) -> int:
    (count,) = read(
        store.path,
        lambda connection: connection.execute(_NOT_PENDING, (doc_id, kind)).fetchone(),
    )
    return int(count)


def rendered_units(store: Store, doc_id: int) -> int:
    """Return the number of a document's chunk units that are done or failed."""
    return _count(store, doc_id, KIND_CHUNK)


def rendered_ranges(store: Store, doc_id: int) -> int:
    """Return the number of a document's range units that are not pending."""
    return _count(store, doc_id, KIND_RANGE)


def render_targets(output: Path, formats: Sequence[str]) -> list[RenderTarget]:
    """Return the render targets of an item's chunks for *formats*.

    *output* is the item's ``_output.json`` path; the CSV goes beside it.
    """
    return [
        RenderTarget(format_path(output, name), name, RENDERERS[name])
        for name in output_formats(formats)
    ]


def _doc_id(store: Store, document: int | str) -> int:
    if isinstance(document, int):
        return document
    row = store.document(document)
    if row is None:
        raise KeyError(f"No document named {document!r} in {store.path}")
    return row.doc_id


def _check_lease(doc_id: int, lease: Lease, document: int | str) -> None:
    if doc_id != lease.doc_id:
        raise ValueError(f"The lease names document {lease.doc_id}, not {document!r}.")


async def render_item(
    store: Store,
    document: int | str,
    formats: Sequence[str],
    output: Path,
    lease: Lease,
    *,
    force: bool = False,
) -> RenderReport:
    """Render document *document* (its id or name) into its chunk formats.

    *output* is the item's ``_output.json`` path and *lease* the document's
    lease of this run. A document without a done or failed chunk gets no
    file. A file that cannot be replaced is marked stale and reported in the
    warnings; a later call replaces it.
    """
    doc_id = _doc_id(store, document)
    _check_lease(doc_id, lease, document)
    if not rendered_units(store, doc_id):
        return RenderReport()
    return await store.render(lease, render_targets(output, formats), force=force)


async def render_ranges(
    store: Store,
    document: int | str,
    path: Path,
    lease: Lease,
    *,
    force: bool = False,
) -> RenderReport:
    """Render the line-ranges file *path* of document *document*.

    A document without a stored range gets no file; a hand-edited file is
    kept unless *force*.
    """
    doc_id = _doc_id(store, document)
    _check_lease(doc_id, lease, document)
    if not rendered_ranges(store, doc_id):
        return RenderReport()
    return await store.render(lease, [line_ranges_target(path)], force=force)


@dataclass
class DocumentRender:
    """What a render did, or would do, with one document's files.

    *skipped* says why the document was not rendered (its lease is held by
    another run, or it was lost); *kept* lists files changed outside the
    tool and left alone, *stale* files that could not be replaced.
    """

    database: Path
    name: str
    written: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    kept: list[Path] = field(default_factory=list)
    stale: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: str | None = None

    @property
    def failed(self) -> bool:
        """Whether a file could not be rendered."""
        return bool(self.stale) or self.skipped is not None


def find_databases(path: Path) -> list[Path]:
    """Return the tool's databases: *path* itself when it is a file, else below it."""
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(found for found in path.rglob(DATABASE_NAME) if found.is_file())


def _registered_path(root: Path, row: OutputRow) -> Path:
    path = Path(row.path)
    return path if path.is_absolute() else root / path


def document_targets(
    root: Path,
    name: str,
    registered: Sequence[OutputRow],
    formats: Sequence[str] | None = None,
    *,
    chunks: bool = True,
    ranges: bool = True,
) -> list[RenderTarget]:
    """Return the render targets of document *name* of output root *root*.

    With *chunks*, the ``_output.json`` and the CSV when *formats* asks for
    it or, without *formats*, when one is registered; registered paths are
    kept, else the files go to ``<root>/<name>_output.json``. With *ranges*,
    a registered line-ranges file; its path is known only once registered.
    """
    paths = {row.format: _registered_path(root, row) for row in registered}
    targets: list[RenderTarget] = []
    if chunks:
        chosen = output_formats(formats if formats else tuple(paths))
        base = paths.get(JSON_FORMAT)
        if base is None and CSV_FORMAT in paths:
            base = paths[CSV_FORMAT].with_suffix(".json")
        if base is None:
            base = output_path(root, name)
        targets.extend(
            RenderTarget(paths.get(fmt) or format_path(base, fmt), fmt, RENDERERS[fmt])
            for fmt in chosen
        )
    if ranges and LINE_RANGES_FORMAT in paths:
        targets.append(line_ranges_target(paths[LINE_RANGES_FORMAT]))
    return targets


@dataclass(frozen=True)
class _Renderable:
    document: DocumentRow
    chunks: bool
    ranges: bool


def _targets(
    root: Path,
    item: _Renderable,
    registered: Sequence[OutputRow],
    formats: Sequence[str] | None,
) -> list[RenderTarget]:
    return document_targets(
        root,
        item.document.name,
        registered,
        formats,
        chunks=item.chunks,
        ranges=item.ranges,
    )


def _preview_document(
    connection: sqlite3.Connection,
    database: Path,
    item: _Renderable,
    registered: Sequence[OutputRow],
    formats: Sequence[str] | None,
    force: bool,
) -> DocumentRender:
    result = DocumentRender(database, item.document.name)
    recorded = {
        _registered_path(database.parent, row): row.sha256 for row in registered
    }
    for target in _targets(database.parent, item, registered, formats):
        text = target.renderer(connection, item.document.doc_id)
        data = text if isinstance(text, bytes) else text.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        try:
            current = file_sha256(target.path)
        except OSError as exc:
            result.stale.append(target.path)
            result.warnings.append(f"Could not read {target.path}: {exc}")
            continue
        if current == digest:
            result.unchanged.append(target.path)
        elif current is not None and current != recorded.get(target.path) and not force:
            result.kept.append(target.path)
        else:
            result.written.append(target.path)
    return result


def _renderable_documents(root: Path) -> tuple[Store, list[_Renderable]] | None:
    reader = read_store(root)
    if reader is None:
        return None
    items = []
    for doc in reader.documents():
        chunks = rendered_units(reader, doc.doc_id) > 0
        ranges = rendered_ranges(reader, doc.doc_id) > 0
        if chunks or ranges:
            items.append(_Renderable(doc, chunks, ranges))
    return reader, items


def preview_database(
    database: Path,
    *,
    formats: Sequence[str] | None = None,
    force: bool = False,
) -> list[DocumentRender]:
    """Return what :func:`render_database` would do; write nothing.

    A target that cannot be read is listed as stale with a warning.

    Raises:
        StoreError: The database is versionless, newer or unreadable.
    """
    root = database.parent
    found = _renderable_documents(root)
    if found is None:
        return []
    reader, items = found
    outputs = {
        item.document.doc_id: reader.outputs(item.document.doc_id) for item in items
    }
    with read_only_connection(root) as connection:
        return [
            _preview_document(
                connection,
                database,
                item,
                outputs[item.document.doc_id],
                formats,
                force,
            )
            for item in items
        ]


async def _render_document(
    store: Store,
    item: _Renderable,
    formats: Sequence[str] | None,
    force: bool,
) -> DocumentRender:
    document = item.document
    result = DocumentRender(store.path, document.name)
    targets = _targets(store.path.parent, item, store.outputs(document.doc_id), formats)
    if not targets:
        return result
    lease = await store.acquire(document.doc_id)
    if lease is None:
        result.skipped = "another run holds its lease"
        await store.add_note(
            f"render skipped '{document.name}': another run holds its lease",
            doc_id=document.doc_id,
        )
        return result
    try:
        report = await store.render(lease, targets, force=force)
    except (LeaseLostError, OSError, sqlite3.Error) as exc:
        result.skipped = f"not rendered: {exc}"
        return result
    finally:
        with contextlib.suppress(sqlite3.Error):
            await store.release(lease)
    result.written = report.written
    result.unchanged = report.unchanged
    result.kept = report.kept
    result.stale = report.stale
    result.warnings = report.warnings
    return result


async def render_database(
    database: Path,
    *,
    formats: Sequence[str] | None = None,
    force: bool = False,
) -> list[DocumentRender]:
    """Render every document of *database* from the store, without model calls.

    Each document is rendered under its lease; a document another run holds
    is skipped with a note. A file changed outside the tool is left alone
    unless *force*; a file that cannot be replaced stays stale for a later
    render. Documents without a done or failed chunk and without a stored
    range are left out.

    Raises:
        StoreError: The database is versionless, newer or unreadable.
    """
    root = database.parent
    found = _renderable_documents(root)
    if found is None or not found[1]:
        return []
    items = found[1]
    results: list[DocumentRender] = []
    async with open_store(root) as store:
        await store.begin_run(
            "render",
            spec={
                "formats": {"value": list(formats or ()), "source": "flag"},
                "force": {"value": force, "source": "flag"},
            },
        )
        try:
            for item in items:
                results.append(await _render_document(store, item, formats, force))
        finally:
            await store.end_run(1 if any(result.failed for result in results) else 0)
    return results
