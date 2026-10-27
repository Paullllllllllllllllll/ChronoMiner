r"""Token-based line ranges and the chunk texts they cut.

A chunk grows line by line until the next line would push it past the
token limit (``--chunk-tokens``); a single line longer than the limit
forms a chunk of its own. Tokens are counted with the tiktoken encoding of
the model, ``cl100k_base`` for models tiktoken does not know, and every line
counts with its newline, since chunks join their lines with ``"\n"``.

:func:`ranges_for_file` computes the ranges ``ranges generate`` writes;
:func:`auto_ranges` the in-memory ranges of ``extract --chunking auto``;
:func:`text_chunks` cuts a text file into the chunk texts of given ranges.
:func:`generate_files` runs ``ranges generate``: it records the ranges of
each text file in ``chronominer.sqlite`` beside it and renders its
``_line_ranges.txt``.
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken

from chronominer.common.selection import PageRange
from chronominer.common.store import Store
from chronominer.events import NullEvents, RunEvents
from chronominer.ranges.lines import Range, ranges_path, read_lines
from chronominer.ranges.store import (
    group_by_folder,
    open_folder_store,
    replace_ranges,
)

__all__ = [
    "GenerateReport",
    "auto_ranges",
    "chunk_texts",
    "generate_files",
    "ranges_for_file",
    "select_ranges",
    "text_chunks",
    "token_ranges",
]

logger = logging.getLogger(__name__)

_FALLBACK_ENCODING = "cl100k_base"


@functools.lru_cache(maxsize=8)
def _encoding(model: str) -> tiktoken.Encoding:
    """Return the tiktoken encoding of *model*, else ``cl100k_base``."""
    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except (KeyError, ValueError):
            pass
    return tiktoken.get_encoding(_FALLBACK_ENCODING)


@functools.lru_cache(maxsize=8)
def _special_tokens(encoding_name: str) -> re.Pattern[str] | None:
    """Return a pattern matching any special token of the encoding, or None."""
    specials = tiktoken.get_encoding(encoding_name).special_tokens_set
    if not specials:
        return None
    return re.compile("|".join(re.escape(token) for token in sorted(specials)))


def _encoder(lines: Sequence[str], model: str) -> Callable[[str], list[int]]:
    """Return the token encoder for *lines*.

    A literal special-token string in the text (a transcription may carry
    ``<|endoftext|>``) is encoded as plain text instead of raising.
    """
    encoding = _encoding(model)
    pattern = _special_tokens(encoding.name)
    if pattern is None or pattern.search("\n".join(lines)) is None:
        return encoding.encode_ordinary
    return functools.partial(encoding.encode, disallowed_special=())


def token_ranges(lines: Sequence[str], chunk_tokens: int, model: str) -> list[Range]:
    """Return the line ranges that keep each chunk within *chunk_tokens*."""
    encode = _encoder(lines, model)
    ranges: list[Range] = []
    current = 0
    start = 1
    for number, line in enumerate(lines, 1):
        tokens = len(encode(line + "\n"))
        if current + tokens > chunk_tokens and current > 0:
            ranges.append((start, number - 1))
            start = number
            current = tokens
        else:
            current += tokens
    if lines:
        ranges.append((start, len(lines)))
    return ranges


def ranges_for_file(path: Path, chunk_tokens: int, model: str) -> list[Range]:
    """Return the token-based ranges of the text file *path*.

    Lines are counted without their surrounding whitespace.
    """
    lines = [line.strip() for line in read_lines(path)]
    return token_ranges(lines, chunk_tokens, model)


def auto_ranges(lines: Sequence[str], chunk_tokens: int, model: str) -> list[Range]:
    """Return the in-memory ranges of ``--chunking auto`` over *lines*."""
    return token_ranges(lines, chunk_tokens, model)


def chunk_texts(lines: Sequence[str], ranges: Sequence[Range]) -> list[str]:
    r"""Return the text of each range: its lines joined with ``"\n"``."""
    return ["\n".join(lines[start - 1 : end]) for start, end in ranges]


def text_chunks(path: Path, ranges: Sequence[Range]) -> list[str]:
    """Return the chunk texts of the text file *path* for *ranges*."""
    return chunk_texts(read_lines(path), ranges)


def select_ranges(ranges: Sequence[Range], pages: PageRange | None) -> list[Range]:
    """Return the ranges *pages* selects, in order; all without a selection."""
    if pages is None:
        return list(ranges)
    return pages.select(list(ranges))


@dataclass(frozen=True)
class GenerateReport:
    """The outcome of ``ranges generate`` for one text file.

    *total* counts the generated ranges and *written* those the selection
    kept; *error* says why the file failed.
    """

    text_file: Path
    ranges_file: Path
    total: int = 0
    written: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the file's ranges were written."""
        return self.error is None


async def generate_files(
    text_files: Sequence[Path],
    *,
    chunk_tokens: int,
    model: str,
    pages: PageRange | None = None,
    events: RunEvents | None = None,
    run_spec: Mapping[str, Any] | None = None,
) -> list[GenerateReport]:
    """Generate and render the line ranges of each text file.

    The ranges of a file are recorded in ``chronominer.sqlite`` in its
    folder, replacing earlier range decisions, and its ``_line_ranges.txt``
    is rendered from them, replacing an existing file. *pages* keeps a
    selection of the generated ranges. A file that fails is reported and
    the others continue.
    """
    sink = events or NullEvents()
    reports: list[GenerateReport] = []
    for folder, files in group_by_folder(text_files).items():
        async with open_folder_store(folder) as store:
            await store.begin_run("ranges generate", spec=run_spec)
            failed = False
            for text_file in files:
                report = await _generate_one(
                    store, text_file, chunk_tokens, model, pages, sink
                )
                failed = failed or not report.ok
                reports.append(report)
            await store.end_run(1 if failed else 0)
    return reports


async def _generate_one(
    store: Store,
    text_file: Path,
    chunk_tokens: int,
    model: str,
    pages: PageRange | None,
    events: RunEvents,
) -> GenerateReport:
    target = ranges_path(text_file)
    try:
        generated = ranges_for_file(text_file, chunk_tokens, model)
        selected = select_ranges(generated, pages)
        rendered = await replace_ranges(store, text_file, selected, model=model)
    except Exception as exc:  # reported per file; the others go on
        logger.exception("Line-range generation failed for %s", text_file)
        events.warning(f"Failed to generate line ranges for {text_file.name}: {exc}")
        return GenerateReport(text_file, target, error=str(exc) or type(exc).__name__)
    for warning in rendered.warnings:
        events.warning(warning)
    if rendered.stale:
        return GenerateReport(
            text_file, target, len(generated), 0, error=rendered.warnings[-1]
        )
    logger.info("Line ranges written to %s", target)
    return GenerateReport(text_file, target, len(generated), len(selected))
