"""Find a model's semantic marker in the source lines.

A marker is a short verbatim substring that starts the line where a unit
begins. Marker and lines are compared after normalization: whitespace runs
collapse to one space, diacritics and case are dropped. The prompt asks for
a marker of :data:`chronominer.spec.DEFAULT_MARKER_LENGTH` characters unless
the run sets another range; a marker shorter than the range's minimum is
rejected, so the caller asks for a longer one. The search stays near the
range: the given window first, then the window widened once by the context
radius on each side; a match farther away would move the boundary far from
the range it belongs to. Among
several matches, a line the marker begins (leading markup, punctuation and
numbering aside) wins over a line that only contains it, so a heading is
not lost to a nearer mention of the same words.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence

from chronominer.spec import DEFAULT_MARKER_LENGTH

__all__ = [
    "ALLOW_SUBSTRING_MATCH",
    "CASE_SENSITIVE",
    "NORMALIZE_DIACRITICS",
    "NORMALIZE_WHITESPACE",
    "STRIP_PUNCTUATION",
    "match_marker",
    "matching_lines",
    "normalize",
]

logger = logging.getLogger(__name__)

NORMALIZE_WHITESPACE = True
CASE_SENSITIVE = False
NORMALIZE_DIACRITICS = True
STRIP_PUNCTUATION = False
ALLOW_SUBSTRING_MATCH = True

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_MARKUP_RE = re.compile(r"^[\W_]+")
_NUMBERING_RE = re.compile(r"^[\W\d_]+")


def _begins(line: str, needle: str) -> bool:
    """Whether *line* opens with *needle*, leading markup and numbering aside.

    Markup is stripped from both; numbering ("1864.", "12)") only from the
    line, so a marker that is itself a number or date keeps its digits.
    """
    target = _MARKUP_RE.sub("", normalize(needle))
    text = normalize(line)
    return bool(target) and (
        _MARKUP_RE.sub("", text).startswith(target)
        or _NUMBERING_RE.sub("", text).startswith(target)
    )


def normalize(text: str) -> str:
    """Return *text* in the form markers and lines are compared in."""
    normalized = text.strip()
    if NORMALIZE_WHITESPACE:
        normalized = _WHITESPACE_RE.sub(" ", normalized)
    if NORMALIZE_DIACRITICS:
        normalized = "".join(
            char
            for char in unicodedata.normalize("NFD", normalized)
            if unicodedata.category(char) != "Mn"
        )
    if not CASE_SENSITIVE:
        normalized = normalized.lower()
    if STRIP_PUNCTUATION:
        normalized = _PUNCTUATION_RE.sub("", normalized)
    return normalized


def matching_lines(
    lines: Sequence[str], needle: str, span: tuple[int, int]
) -> list[int]:
    """Return the 1-based numbers of the lines in *span* that match *needle*.

    A line matches when the normalized needle occurs in it, or, without
    substring matching, when both are equal.
    """
    start, end = span
    end = min(len(lines), max(start, end))
    target = normalize(needle)
    found: list[int] = []
    for number in range(start, end + 1):
        line = normalize(lines[number - 1])
        if (target in line) if ALLOW_SUBSTRING_MATCH else (line == target):
            found.append(number)
    return found


def match_marker(
    marker: str | None,
    lines: Sequence[str],
    window: tuple[int, int],
    *,
    nearest_to: int,
    widen_by: int,
    min_length: int = DEFAULT_MARKER_LENGTH[0],
) -> int | None:
    """Return the line where *marker* starts a unit, or None.

    A marker shorter than *min_length* characters is rejected. The search
    covers *window*, then, without a match, the window widened by *widen_by*
    lines on each side. Several matches resolve to the lines the marker
    begins, if any, and among those to the line nearest *nearest_to*, ties to
    the earlier line.
    """
    needle = (marker or "").strip()
    if not needle:
        return None
    if len(needle) < min_length:
        logger.debug(
            "Rejecting marker %r: shorter than %d characters", needle, min_length
        )
        return None
    total = len(lines)
    bounded = (max(1, window[0]), min(total, window[1]))
    found = matching_lines(lines, needle, bounded)
    if not found:
        widened = (max(1, bounded[0] - widen_by), min(total, bounded[1] + widen_by))
        if widened != bounded:
            found = matching_lines(lines, needle, widened)
    if not found:
        return None
    if len(found) > 1:
        logger.debug(
            "Marker %r matches lines %s; preferring a line it begins, then the "
            "one nearest line %d",
            needle,
            found,
            nearest_to,
        )
        found = [
            number for number in found if _begins(lines[number - 1], needle)
        ] or found
    return min(found, key=lambda number: (abs(number - nearest_to), number))
