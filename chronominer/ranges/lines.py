"""The ``_line_ranges.txt`` format beside an input text, and text reading.

A line-ranges file holds one 1-based, inclusive ``(start, end)`` pair per
line. It is an input: ``ranges generate`` writes it, a user may edit it, and
``extract --chunking line-ranges`` reads it. ``ranges adjust`` renders it
again from the store.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from charset_normalizer import detect

__all__ = [
    "LINE_RANGES_SUFFIX",
    "Range",
    "clamp_ranges",
    "format_ranges",
    "parse_ranges",
    "ranges_path",
    "read_lines",
    "read_ranges",
]

logger = logging.getLogger(__name__)

Range = tuple[int, int]
"""A 1-based, inclusive ``(start, end)`` line range."""

LINE_RANGES_SUFFIX = "_line_ranges.txt"

_DETECT_BYTES = 100_000


def ranges_path(text_file: Path) -> Path:
    """Return the ``{stem}_line_ranges.txt`` file beside *text_file*."""
    return text_file.with_name(f"{text_file.stem}{LINE_RANGES_SUFFIX}")


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


def parse_ranges(text: str, *, source: str = "line ranges") -> list[Range]:
    """Return the ranges of a line-ranges text; malformed lines are skipped.

    A line is ``(start, end)`` or ``start, end``. Lines that do not parse,
    and ranges with a start below 1 or an end before the start, are logged
    and skipped.
    """
    ranges: list[Range] = []
    for line in text.splitlines():
        parsed = _parse_line(line, source)
        if parsed is not None:
            ranges.append(parsed)
    return ranges


def read_ranges(path: Path) -> list[Range]:
    """Return the ranges of the line-ranges file *path*.

    A leading byte-order mark is ignored. ``OSError`` propagates.
    """
    text = path.read_text(encoding="utf-8-sig")
    return parse_ranges(text, source=str(path))


def format_ranges(ranges: Iterable[Range]) -> str:
    """Return *ranges* in the file format, one ``(start, end)`` per line."""
    return "".join(f"({start}, {end})\n" for start, end in ranges)


def clamp_ranges(ranges: Iterable[Range], total_lines: int) -> list[Range]:
    """Clamp *ranges* to lines ``1`` to *total_lines*.

    A range that lies wholly outside the text is dropped with a warning.
    """
    clamped: list[Range] = []
    for start, end in ranges:
        low, high = max(1, start), min(total_lines, end)
        if low > high:
            logger.warning(
                "Dropping out-of-bounds line range (%d, %d): the text has %d line(s)",
                start,
                end,
                total_lines,
            )
            continue
        clamped.append((low, high))
    return clamped


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
