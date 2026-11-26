"""The ``_line_ranges.txt`` format beside an input text, and text reading.

A line-ranges file holds one 1-based, inclusive ``(start, end)`` pair per
line. It is an input: ``ranges generate`` writes it, a user may edit it, and
``extract --chunking line-ranges`` reads it. ``ranges adjust`` renders it
again from the store.

A line that begins with the word ``context`` names the context ranges of
the range on the line above it: further ranges of the same text, written as
``(start, end)`` pairs separated by commas, that are sent with that chunk as
read-only context. Several context lines of one range add up. Context
ranges change neither the lines a chunk covers nor the number of chunks.
Only ``--chunking line-ranges`` reads them: a file with a context line is
not adjusted (:func:`context_lines_refusal`), and the store never renders
one.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

from charset_normalizer import detect

__all__ = [
    "CONTEXT_KEYWORD",
    "LINE_RANGES_SUFFIX",
    "Context",
    "Range",
    "RangeLine",
    "clamp_line_ranges",
    "clamp_ranges",
    "context_lines_refusal",
    "format_context",
    "format_ranges",
    "parse_line_ranges",
    "parse_ranges",
    "ranges_path",
    "read_line_ranges",
    "read_lines",
    "read_ranges",
    "stored_context",
]

logger = logging.getLogger(__name__)

Range = tuple[int, int]
"""A 1-based, inclusive ``(start, end)`` line range."""

Context = tuple[Range, ...]
"""The context ranges of one chunk, sorted and without repeats."""

RangeLine = tuple[Range, Context]
"""One range of a line-ranges file with its context ranges."""

LINE_RANGES_SUFFIX = "_line_ranges.txt"

CONTEXT_KEYWORD: Final = "context"
"""The first word of a line that lists the context ranges of the range above."""

_DETECT_BYTES = 100_000
_PAIR: Final = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)"
_CONTEXT_LINE: Final = re.compile(
    rf"{CONTEXT_KEYWORD}\s*{_PAIR}(?:\s*,?\s*{_PAIR})*\s*,?", re.IGNORECASE
)
_CONTEXT_PAIR: Final = re.compile(_PAIR)


def ranges_path(text_file: Path) -> Path:
    """Return the ``{stem}_line_ranges.txt`` file beside *text_file*."""
    return text_file.with_name(f"{text_file.stem}{LINE_RANGES_SUFFIX}")


def _is_context_line(text: str) -> bool:
    """Whether *text* is ``context`` and pairs, at least one a valid range."""
    if _CONTEXT_LINE.fullmatch(text) is None:
        return False
    pairs = (match.groups() for match in _CONTEXT_PAIR.finditer(text))
    return any(1 <= int(start) <= int(end) for start, end in pairs)


def context_lines_refusal(path: Path) -> str | None:
    """Return why the line-ranges file *path* cannot be adjusted, or None.

    A file with a context line cannot be adjusted; a note in prose that
    starts with the word is no context line. A file that is missing or
    cannot be read or decoded is left to the reader that opens it.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None
    if not any(_is_context_line(line.strip()) for line in text.splitlines()):
        return None
    return (
        f"{path} holds context lines, and a line-ranges file with context lines "
        "cannot be adjusted. Remove the context lines, or extract with the file "
        "as it is under --chunking line-ranges."
    )


def stored_context(value: Any) -> Context:
    """Return the context ranges a stored ``[[start, end], ...]`` value holds.

    Pairs that are not two integers are left out; the rest come sorted and
    without repeats. Anything but a list gives none.
    """
    if not isinstance(value, list):
        return ()
    pairs: set[Range] = set()
    for item in value:
        if (
            isinstance(item, list)
            and len(item) == 2
            and all(isinstance(n, int) and not isinstance(n, bool) for n in item)
        ):
            pairs.add((int(item[0]), int(item[1])))
    return tuple(sorted(pairs))


def format_context(context: Iterable[Range]) -> str:
    """Return the context line of *context*, without indent or newline."""
    pairs = ", ".join(f"({start}, {end})" for start, end in context)
    return f"{CONTEXT_KEYWORD} {pairs}"


def _parse_line(line: str, source: str) -> Range | None:
    text = line.strip()
    if not text:
        return None
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    elif not text[0].isdigit():
        logger.warning("Invalid line range format in %s: %s", source, line.strip())
        return None
    parts = text.split(",")
    if len(parts) != 2:
        logger.warning("Expected two numbers in a line range of %s: %s", source, text)
        return None
    try:
        start, end = int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        logger.warning("Invalid integers in a line range of %s: %s", source, text)
        return None
    if start < 1 or end < start:
        logger.warning(
            "Skipping out-of-order or non-positive line range (%d, %d) in %s",
            start,
            end,
            source,
        )
        return None
    return start, end


def _parse_context(text: str, source: str) -> Context:
    """Return the valid context ranges of the context line *text*."""
    ranges: list[Range] = []
    for match in _CONTEXT_PAIR.finditer(text):
        start, end = int(match.group(1)), int(match.group(2))
        if start < 1 or end < start:
            logger.warning(
                "Skipping out-of-order or non-positive context range (%d, %d) in %s",
                start,
                end,
                source,
            )
            continue
        ranges.append((start, end))
    return tuple(ranges)


def parse_line_ranges(text: str, *, source: str = "line ranges") -> list[RangeLine]:
    """Return the ranges of a line-ranges text with their context ranges.

    A range line is ``(start, end)`` or ``start, end``; a context line,
    the word ``context`` followed by ``(start, end)`` pairs of which at
    least one is a valid range, belongs to the range line above it. Lines
    that do not parse (a note in prose among them), ranges with a start
    below 1 or an end before the
    start, and context lines without a valid range above them are logged
    and skipped; a note leaves the range above it valid.
    """
    entries: list[tuple[Range, list[Range]]] = []
    previous_valid = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _is_context_line(stripped):
            if not previous_valid:
                logger.warning(
                    "Skipping a context line without a range above it in %s: %s",
                    source,
                    stripped,
                )
                continue
            entries[-1][1].extend(_parse_context(stripped, source))
            continue
        parsed = _parse_line(line, source)
        if parsed is not None:
            entries.append((parsed, []))
            previous_valid = True
        elif stripped.startswith("(") or stripped[0].isdigit():
            previous_valid = False
    return [(bounds, tuple(sorted(set(context)))) for bounds, context in entries]


def parse_ranges(text: str, *, source: str = "line ranges") -> list[Range]:
    """Return the ranges of a line-ranges text; malformed lines are skipped.

    A line is ``(start, end)`` or ``start, end``. Lines that do not parse,
    and ranges with a start below 1 or an end before the start, are logged
    and skipped. Context lines are read and left out.
    """
    return [bounds for bounds, _ in parse_line_ranges(text, source=source)]


def read_line_ranges(path: Path) -> list[RangeLine]:
    """Return the ranges of the line-ranges file *path* with their context ranges.

    A leading byte-order mark is ignored. ``OSError`` propagates.
    """
    text = path.read_text(encoding="utf-8-sig")
    return parse_line_ranges(text, source=str(path))


def read_ranges(path: Path) -> list[Range]:
    """Return the ranges of the line-ranges file *path*.

    A leading byte-order mark is ignored. ``OSError`` propagates.
    """
    return [bounds for bounds, _ in read_line_ranges(path)]


def format_ranges(
    ranges: Iterable[Range], contexts: Sequence[Context] | None = None
) -> str:
    """Return *ranges* in the file format, one ``(start, end)`` per line.

    *contexts* holds the context ranges of each range, in the same order; a
    range with any is followed by its context line.
    """
    lines: list[str] = []
    for position, (start, end) in enumerate(ranges):
        lines.append(f"({start}, {end})\n")
        context = contexts[position] if contexts is not None else ()
        if context:
            lines.append(f"  {format_context(context)}\n")
    return "".join(lines)


def _clamped(bounds: Range, total_lines: int, noun: str) -> Range | None:
    start, end = bounds
    low, high = max(1, start), min(total_lines, end)
    if low > high:
        logger.warning(
            "Dropping out-of-bounds %s (%d, %d): the text has %d line(s)",
            noun,
            start,
            end,
            total_lines,
        )
        return None
    return low, high


def clamp_line_ranges(
    entries: Iterable[RangeLine], total_lines: int
) -> list[RangeLine]:
    """Clamp ranges and their context ranges to lines ``1`` to *total_lines*.

    A range that lies wholly outside the text is dropped with its context
    ranges, and a context range outside it alone; each with a warning.
    """
    clamped: list[RangeLine] = []
    for bounds, context in entries:
        kept = _clamped(bounds, total_lines, "line range")
        if kept is None:
            continue
        inside = (_clamped(pair, total_lines, "context range") for pair in context)
        clamped.append((kept, tuple(sorted({pair for pair in inside if pair}))))
    return clamped


def clamp_ranges(ranges: Iterable[Range], total_lines: int) -> list[Range]:
    """Clamp *ranges* to lines ``1`` to *total_lines*.

    A range that lies wholly outside the text is dropped with a warning.
    """
    lines = clamp_line_ranges(((bounds, ()) for bounds in ranges), total_lines)
    return [bounds for bounds, _ in lines]


def _detected_encoding(path: Path) -> str:
    with path.open("rb") as handle:
        sample = handle.read(_DETECT_BYTES)
    encoding = detect(sample).get("encoding")
    return str(encoding) if encoding else "utf-8"


def read_lines(path: Path) -> list[str]:
    """Return the lines of the text file *path* without their line ends.

    The file is read as UTF-8; when that fails, in the encoding that
    ``charset_normalizer`` detects.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        encoding = _detected_encoding(path)
        logger.warning("%s is not UTF-8; reading it as %s", path.name, encoding)
        text = path.read_text(encoding=encoding)
    if not text:
        return []
    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()
    return lines
