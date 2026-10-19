"""The ``{stem}_line_ranges.txt`` view of a text document's range units.

The file is an input as well (an editable chunk plan, read by
``--chunking line-ranges`` and by Mabillon). Each line holds one range as
``(start, end)``, 1-based and inclusive, in the order of the range units;
units without content (deleted or merged ranges) are left out. A file with
ranges ends with one newline; a document without ranges renders empty.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from chronominer.common.store import RenderTarget
from chronominer.output.store import KIND_RANGE, LINE_RANGES_SUFFIX

__all__ = [
    "LINE_RANGES_FORMAT",
    "format_range",
    "line_ranges_path",
    "line_ranges_target",
    "parse_range",
    "range_content",
    "render_line_ranges",
]

LINE_RANGES_FORMAT = "line_ranges"
"""The format name of the line-ranges file in ``outputs``."""

_RANGES = (
    "SELECT content FROM units WHERE doc_id = ? AND kind = ? "
    "AND content IS NOT NULL ORDER BY idx"
)


def range_content(start: int, end: int) -> str:
    """Return the unit content that stores the range *start* to *end*."""
    return json.dumps([int(start), int(end)])


def parse_range(content: str | None) -> tuple[int, int] | None:
    """Return the range a range unit's content stores, or None."""
    if content is None:
        return None
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        return int(value[0]), int(value[1])
    return None


def format_range(start: int, end: int) -> str:
    """Return one line of the file, without its newline."""
    return f"({start}, {end})"


def render_line_ranges(connection: sqlite3.Connection, doc_id: int) -> str:
    """Return the line-ranges file of document *doc_id*."""
    lines = []
    for (content,) in connection.execute(_RANGES, (doc_id, KIND_RANGE)):
        parsed = parse_range(content)
        if parsed is not None:
            lines.append(format_range(*parsed))
    return "".join(f"{line}\n" for line in lines)


def line_ranges_path(text_file: Path) -> Path:
    """Return the line-ranges file beside the text file *text_file*."""
    return text_file.with_name(f"{text_file.stem}{LINE_RANGES_SUFFIX}")


def line_ranges_target(path: Path) -> RenderTarget:
    """Return the render target of the line-ranges file at *path*."""
    return RenderTarget(path, LINE_RANGES_FORMAT, render_line_ranges)
