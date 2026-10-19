"""The chunks of a document and the payload handler of the unit pipeline.

A text document is cut into line ranges by ``--chunking``: ``auto`` cuts the
text in memory by ``--chunk-tokens``; ``line-ranges`` reads the
``{stem}_line_ranges.txt`` beside the input, generating it from
``--chunk-tokens`` when it is missing; ``adjust`` first moves the boundaries
of that file with the model (:func:`~chronominer.ranges.adjust.adjust_ranges`)
in the run's database, under the run's lease on the document. Each range is
one :class:`TextChunk`. An image file is one chunk, a PDF one chunk per page
(:class:`~chronominer.extract.images.PageImage`).

:func:`extract_payload` extracts one chunk with the run's
:class:`~chronominer.extract.extraction.ChunkExtractor` and returns a
:data:`~chronominer.extract.units.PayloadResult`; a failure becomes a chunk
without answer text instead of raising.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronominer.common.selection import PageRange
from chronominer.common.store import Lease, Store
from chronominer.events import RunEvents
from chronominer.extract.units import PayloadResult
from chronominer.ranges.adjust import AdjustOptions, BoundaryCaller, adjust_ranges
from chronominer.ranges.generate import auto_ranges, chunk_texts, ranges_for_file
from chronominer.ranges.lines import Range, clamp_ranges, ranges_path, read_ranges
from chronominer.ranges.store import replace_ranges

__all__ = [
    "CHUNKING_METHODS",
    "CHUNKING_TEXT_VERSION",
    "PAGES_METHOD",
    "LineRangesMissingError",
    "RangeJob",
    "TextChunk",
    "build_custom_id",
    "chunk_slice",
    "custom_id",
    "extract_payload",
    "text_chunks",
    "text_ranges",
]

logger = logging.getLogger(__name__)

CHUNKING_TEXT_VERSION = 2
"""The text-chunking behavior the ``_output.json`` metadata records."""

CHUNKING_METHODS = {
    "auto": "auto",
    "line-ranges": "line_ranges",
    "adjust": "line_ranges.txt",
}
"""The ``chunking_method`` metadata of each ``--chunking`` mode."""

PAGES_METHOD = "pages"
"""The ``chunking_method`` metadata of image and PDF documents."""

_CUSTOM_ID_MAX_LENGTH = 64
_CUSTOM_ID_ILLEGAL = re.compile(r"[^A-Za-z0-9_-]")
_CUSTOM_ID_HASH_LENGTH = 8


class LineRangesMissingError(FileNotFoundError):
    """The line-ranges file of a text input is missing and cannot be made."""


def build_custom_id(stem: str, suffix: str) -> str:
    """Return a provider-safe custom id of a file stem and an ordering suffix.

    *suffix* (``-chunk-3``, ``-page-7``) is kept verbatim. The stem is
    reduced to ``[A-Za-z0-9_-]``; since that is lossy, a short hash of the
    original stem is appended whenever the stem changed, and again when the
    id is cut to 64 characters.
    """
    sanitized = _CUSTOM_ID_ILLEGAL.sub("_", stem)
    digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:_CUSTOM_ID_HASH_LENGTH]
    base = sanitized if sanitized == stem else f"{sanitized}_{digest}"
    candidate = f"{base}{suffix}"
    if len(candidate) <= _CUSTOM_ID_MAX_LENGTH:
        return candidate
    keep = max(_CUSTOM_ID_MAX_LENGTH - len(suffix) - _CUSTOM_ID_HASH_LENGTH - 1, 0)
    return f"{sanitized[:keep]}_{digest}{suffix}"


def custom_id(path: Path, kind: str, index: int) -> str:
    """Return the custom id of chunk *index* (0-based) of the input *path*.

    Text chunks are ``{stem}-chunk-{n}``, pages ``{stem}-page-{n}``, with
    the 1-based chunk number *n*.
    """
    unit = "chunk" if kind == "text" else "page"
    return build_custom_id(path.stem, f"-{unit}-{index + 1}")


def chunk_slice(pages: PageRange | None) -> dict[str, Any] | None:
    """Return the ``chunk_slice`` metadata of a ``--pages`` selection."""
    if pages is None:
        return None
    if pages.first_n is not None:
        return {"first_n": pages.first_n}
    if pages.last_n is not None:
        return {"last_n": pages.last_n}
    if len(pages.spans) == 1:
        start, end = pages.spans[0]
        return {"page_range": [start + 1, end + 1]}
    return {"pages": pages.describe()}


@dataclass(frozen=True)
class TextChunk:
    """One line range of a text document as the unit pipeline extracts it.

    *start* and *end* are 1-based and inclusive; *image_name* is the
    chunk's label, ``start-end``.
    """

    index: int
    start: int
    end: int
    text: str
    custom_id: str = ""
    render_error: str | None = None
    page_index: int | None = None
    base64: str = ""
    mime_type: str = "text/plain"

    @property
    def image_name(self) -> str:
        """The chunk's label: its line range."""
        return f"{self.start}-{self.end}"

    @property
    def source_index(self) -> str:
        """The chunk's line range, ``start-end``."""
        return self.image_name

    @property
    def chunk_range(self) -> list[int]:
        """The chunk's ``[start, end]``."""
        return [self.start, self.end]

    def provenance(self) -> dict[str, Any]:
        """Return the chunk unit's provenance: custom id and line range."""
        record: dict[str, Any] = {}
        if self.custom_id:
            record["custom_id"] = self.custom_id
        record["chunk_range"] = self.chunk_range
        return record


def text_chunks(
    path: Path, lines: Sequence[str], ranges: Sequence[Range]
) -> list[TextChunk]:
    """Return the chunks of the text *path* cut by *ranges*."""
    texts = chunk_texts(lines, ranges)
    return [
        TextChunk(index, start, end, text, custom_id(path, "text", index))
        for index, ((start, end), text) in enumerate(zip(ranges, texts, strict=True))
    ]


class _HeldLease:
    """The run's store with the run's lease on one document.

    The line-range functions lease the document they write; here they work
    the extraction document the run already holds, so acquiring it returns
    the run's lease and releasing it keeps the lease. Everything else goes
    to the store.
    """

    def __init__(self, store: Store, lease: Lease) -> None:
        self._store = store
        self._lease = lease

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    async def acquire(self, doc_id: int) -> Lease | None:
        """Return the held lease for its document, else acquire."""
        if doc_id == self._lease.doc_id:
            return self._lease
        return await self._store.acquire(doc_id)

    async def release(self, lease: Lease) -> None:
        """Keep the held lease; release any other."""
        if lease == self._lease:
            return
        await self._store.release(lease)


@dataclass(frozen=True)
class RangeJob:
    """What cutting one text document needs.

    *mode* is the ``--chunking`` mode, *model* the model whose tokenizer
    counts *chunk_tokens*; *caller* and *adjust* serve the ``adjust`` mode.
    """

    mode: str
    chunk_tokens: int
    model: str
    caller: BoundaryCaller | None = None
    adjust: AdjustOptions | None = None
    events: RunEvents | None = None


async def text_ranges(
    path: Path,
    lines: Sequence[str],
    job: RangeJob,
    *,
    store: Store,
    lease: Lease,
    document: str,
) -> list[Range]:
    """Return the line ranges of the text *path* for ``--chunking`` *job.mode*.

    The ``line-ranges`` and ``adjust`` modes write their ranges to the range
    units of *document* in *store* under *lease* and render the
    ``_line_ranges.txt`` beside the input.

    Raises:
        LineRangesMissingError: The ``adjust`` mode has no caller or options.
        AdjustDegradedError: Every boundary call failed.
    """
    if job.mode == "auto":
        return auto_ranges(lines, job.chunk_tokens, job.model)
    held: Any = _HeldLease(store, lease)
    ranges_file = ranges_path(path)
    if job.mode == "adjust":
        if job.caller is None or job.adjust is None:
            raise LineRangesMissingError(
                f"Cannot adjust the line ranges of {path.name}: no model caller."
            )
        report = await adjust_ranges(
            held,
            path,
            caller=job.caller,
            model=job.model,
            options=job.adjust,
            document=document,
            events=job.events,
        )
        if report.status == "empty":
            return []
        return clamp_ranges(report.ranges, len(lines))
    if not ranges_file.exists():
        generated = ranges_for_file(path, job.chunk_tokens, job.model)
        rendered = await replace_ranges(
            held, path, generated, model=job.model, name=document
        )
        for warning in rendered.warnings:
            logger.warning(warning)
        logger.info("Generated %s with %d range(s)", ranges_file.name, len(generated))
        if not ranges_file.exists():
            return clamp_ranges(generated, len(lines))
    return clamp_ranges(read_ranges(ranges_file), len(lines))


def _no_text(payload: Any) -> PayloadResult:
    return (payload, payload.image_name, None, None, payload.index)


async def extract_payload(payload: Any, extractor: Any) -> PayloadResult:
    """Extract one chunk payload with the chunk extractor.

    A text chunk goes as text, a page as an image. A page that never produced
    an image (``render_error``) is recorded as a failed chunk without a
    request; an answer with an error has no text.
    """
    label = payload.image_name
    if getattr(payload, "render_error", None) is not None:
        return _no_text(payload)
    try:
        text = getattr(payload, "text", None)
        if isinstance(text, str):
            result = await extractor.extract_text(text, label=label)
        else:
            result = await extractor.extract_image_from_base64(
                payload.base64, payload.mime_type, label=label
            )
    except Exception as exc:
        logger.exception("Error extracting %s: %s", label, exc)
        return _no_text(payload)
    logger.debug("LLM response for %s: %s", label, result)
    answer: str | None = None
    if isinstance(result, dict) and not result.get("error"):
        output = result.get("output_text")
        answer = output if isinstance(output, str) else None
    return (payload, label, answer, result, payload.index)
