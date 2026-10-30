"""The units a classifier judges in ChronoMiner: chunks, entries and documents.

Every function here reads, none writes. A ``chunk`` unit is one extraction
chunk, cut as ``extract`` cuts it: a text item's line ranges come from the
store when it holds the document's chunks, else from ``--chunking auto`` in
memory or from the ``{stem}_line_ranges.txt`` beside the input; an image is
one chunk and a PDF one chunk per page, rendered by the extractor's page
sources and converted with :func:`~chronominer.common.images.decision_jpeg`.
An ``entry`` unit is one extracted entry, sent as the classifier's fields; a
``document`` unit is the classifier's selection of an item in one unit.

Locators and ordinals (the read contract of the ``classifications`` view):
a text chunk is ``lines 120-180``, a PDF page ``page 3`` and an image its
file name, each with the 1-based ``chunk_index`` as ordinal (chunk answers
join on document and locator, since positions move with the layout); an entry is
``chunk_index:entry_index`` (:data:`~chronominer.output.store.ENTRY_LOCATOR`)
with its chunk's ``chunk_index``; a document is ``document`` with ordinal 0.
"""

from __future__ import annotations

import base64
import binascii
import json
import sqlite3
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from PIL import UnidentifiedImageError

from chronominer.common.decision_runner import Unit
from chronominer.common.decisions import Image, Json, Part, Text
from chronominer.common.images import decision_jpeg
from chronominer.common.store import UnitRow
from chronominer.extract.chunks import TextChunk, text_chunks
from chronominer.extract.images import FolderPages, stream_pages
from chronominer.extract.llm import chunk_ranges
from chronominer.extract.pdf import PdfPages, PDFProcessor
from chronominer.output.store import ENTRY_LOCATOR, ENTRY_TABLE, KIND_CHUNK
from chronominer.ranges.generate import auto_ranges, count_tokens, truncate_tokens
from chronominer.ranges.lines import Range, clamp_ranges, ranges_path, read_ranges

__all__ = [
    "DOCUMENT_LOCATOR",
    "PageUnit",
    "SidecarMissingError",
    "chunk_locator",
    "entry_units",
    "page_count",
    "page_units",
    "text_chunk_unit",
    "text_document_unit",
    "text_item_chunks",
]

DOCUMENT_LOCATOR: Final = "document"
"""The locator of a document unit; its ordinal is 0."""


class SidecarMissingError(FileNotFoundError):
    """``--chunking line-ranges`` names a ``{stem}_line_ranges.txt`` that is missing."""


def chunk_locator(kind: str, index: int, name: str = "") -> str:
    """Return the locator of chunk *index* (0-based) of an item of *kind*.

    A text chunk's *name* is its line range ``start-end``, an image's its
    file name.
    """
    if kind == "text":
        return f"lines {name}"
    if kind == "pdf":
        return f"page {index + 1}"
    return name


def text_item_chunks(
    path: Path,
    lines: Sequence[str],
    *,
    stored: Sequence[UnitRow],
    mode: str,
    chunk_tokens: int,
    model: str,
) -> list[TextChunk]:
    """Return the chunks of the text item *path*, cut as extract cuts them.

    *stored* are the chunk units the store holds for the item's document;
    their line ranges win. Otherwise ``auto`` cuts *lines* by *chunk_tokens*
    in *model*'s encoding and ``line-ranges`` reads the sidecar.

    Raises:
        SidecarMissingError: ``line-ranges`` without a stored layout and
            without the ``{stem}_line_ranges.txt`` beside the input.
    """
    ranges: list[Range] = chunk_ranges(stored) if stored else []
    if not ranges and mode == "auto":
        ranges = auto_ranges(lines, chunk_tokens, model)
    elif not ranges:
        sidecar = ranges_path(path)
        if not sidecar.is_file():
            raise SidecarMissingError(
                f"{sidecar.name} is missing beside {path.name}; write it with "
                "'chronominer ranges generate' or 'chronominer ranges adjust' "
                "first, or classify with --chunking auto"
            )
        ranges = clamp_ranges(read_ranges(sidecar), len(lines))
    return text_chunks(path, lines, ranges)


def text_chunk_unit(chunk: TextChunk) -> Unit:
    """Return the unit of one text chunk."""
    return Unit(
        chunk_locator("text", chunk.index, chunk.source_index),
        chunk.index + 1,
        (Text(chunk.text),),
    )


def text_document_unit(
    chunks: Sequence[TextChunk], max_tokens: int, model: str
) -> Unit:
    """Return the document unit of a text: its leading chunks up to *max_tokens*.

    Chunks are taken whole, in order, while their tokens in *model*'s
    encoding stay within *max_tokens*; a first chunk longer than that is cut
    to its first *max_tokens* tokens.
    """
    parts: list[Part] = []
    total = 0
    for chunk in chunks:
        tokens = count_tokens(chunk.text, model)
        if not parts and tokens > max_tokens:
            parts.append(Text(truncate_tokens(chunk.text, max_tokens, model)))
            break
        if total + tokens > max_tokens:
            break
        parts.append(Text(chunk.text))
        total += tokens
    return Unit(DOCUMENT_LOCATOR, 0, tuple(parts))


def page_count(path: Path, kind: str) -> int:
    """Return the number of chunks of an image (1) or PDF item."""
    return PDFProcessor(path).page_count() if kind == "pdf" else 1


@dataclass(frozen=True)
class PageUnit:
    """One page of an item: its unit, or why it could not be made."""

    index: int
    unit: Unit | None
    image: bytes | None = None
    error: str | None = None


async def page_units(  # noqa: PLR0913 - the page source's settings
    path: Path,
    kind: str,
    indices: Sequence[int],
    *,
    img_cfg: Mapping[str, Any],
    model_type: str,
    section: str,
    max_px: int,
) -> AsyncIterator[PageUnit]:
    """Yield the chunk unit of each page *indices* selects, in order.

    The pages are rendered by the extractor's page sources with *img_cfg*
    and converted with ``decision_jpeg`` at *max_px*. A page that did not
    render, or whose image cannot be read, has no unit and names its error.
    """
    settings = dict(img_cfg)
    pages: PdfPages | FolderPages
    if kind == "pdf":
        pages = PdfPages(path, settings, model_type, section=section)
    else:
        pages = FolderPages(
            path.parent, settings, model_type, image_paths=[path], section=section
        )
    async for page in stream_pages(pages, indices):
        if page.render_error is not None:
            yield PageUnit(page.index, None, error=page.render_error)
            continue
        try:
            image = decision_jpeg(base64.b64decode(page.base64), max_px)
        except (UnidentifiedImageError, OSError, binascii.Error) as exc:
            yield PageUnit(page.index, None, error=str(exc) or type(exc).__name__)
            continue
        locator = chunk_locator(kind, page.index, path.name)
        unit = Unit(locator, page.index + 1, (Image(image),))
        yield PageUnit(page.index, unit, image)


_ENTRIES: Final = f"""
SELECT e.idx, e.entry_index, e.entry FROM {ENTRY_TABLE} AS e
JOIN units AS u ON u.doc_id = e.doc_id AND u.kind = e.kind AND u.idx = e.idx
WHERE e.doc_id = ? AND e.kind = ? AND u.status = 'done' AND u.stale = 0
AND u.run_id = e.run_id
ORDER BY e.idx, e.entry_index
"""
"""The entries the ``entries`` view shows for one document."""


def entry_units(
    connection: sqlite3.Connection, doc_id: int, fields: Sequence[str]
) -> list[list[Unit]]:
    """Return the entry units of document *doc_id*, one list per chunk.

    Each unit sends the classifier's *fields* of one entry as a JSON object
    in that order, a field the entry lacks as null.
    """
    chunks: dict[int, list[Unit]] = {}
    for idx, entry_index, text in connection.execute(_ENTRIES, (doc_id, KIND_CHUNK)):
        try:
            entry = json.loads(text)
        except (TypeError, ValueError):
            entry = {}
        if not isinstance(entry, dict):
            entry = {}
        payload = {name: entry.get(name) for name in fields}
        locator = ENTRY_LOCATOR.format(chunk_index=idx + 1, entry_index=entry_index)
        chunks.setdefault(int(idx), []).append(
            Unit(locator, int(idx) + 1, (Json(payload),))
        )
    return [chunks[idx] for idx in sorted(chunks)]
