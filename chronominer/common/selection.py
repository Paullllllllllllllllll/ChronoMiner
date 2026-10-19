"""Page ranges and item selection.

Page ranges select pages, images or chunks of one document (1-based,
inclusive):

- ``5`` or ``first:5``: the first 5 units
- ``last:5``: the last 5 units
- ``3-7``, ``3-``, ``-7``: closed and open ranges
- ``1,3,5-8``: a comma-separated list of numbers and ranges

Item selection picks discovered items by number, range or name search.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

_FIRST_RE = re.compile(r"^first\s*:\s*(\d+)$", re.IGNORECASE)
_LAST_RE = re.compile(r"^last\s*:\s*(\d+)$", re.IGNORECASE)
_OPEN_END = 2**31


@dataclass(frozen=True)
class PageRange:
    """A parsed page-range specification.

    Holds either a first/last count or spans of 0-based, inclusive
    ``(start, end)`` indices.
    """

    first_n: int | None = None
    last_n: int | None = None
    spans: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    raw: str = ""

    def resolve(self, total: int) -> list[int]:
        """Return the sorted 0-based indices selected out of *total* units."""
        if total <= 0:
            return []
        if self.first_n is not None:
            return list(range(min(self.first_n, total)))
        if self.last_n is not None:
            n = min(self.last_n, total)
            return list(range(total - n, total))
        indices: set[int] = set()
        for start, end in self.spans:
            low, high = max(0, start), min(end, total - 1)
            if low <= high:
                indices.update(range(low, high + 1))
        return sorted(indices)

    def select[T](self, units: Sequence[T]) -> list[T]:
        """Return the selected units in document order."""
        return [units[i] for i in self.resolve(len(units))]

    def describe(self) -> str:
        """Return a short human-readable description."""
        if self.first_n is not None:
            return f"first {self.first_n} page(s)"
        if self.last_n is not None:
            return f"last {self.last_n} page(s)"
        if not self.spans:
            return "all pages"
        parts: list[str] = []
        for start, end in self.spans:
            if start == end:
                parts.append(str(start + 1))
            elif end >= _OPEN_END - 1:
                parts.append(f"{start + 1}-")
            else:
                parts.append(f"{start + 1}-{end + 1}")
        return "pages " + ",".join(parts)


def parse_page_range(spec: str) -> PageRange:
    """Parse a page-range specification; raise ``ValueError`` when malformed."""
    spec = spec.strip()
    if not spec:
        raise ValueError("Page range specification cannot be empty.")
    if spec.isdigit():
        return PageRange(first_n=_positive_count(spec), raw=spec)
    match = _FIRST_RE.match(spec)
    if match:
        return PageRange(first_n=_positive_count(match.group(1)), raw=spec)
    match = _LAST_RE.match(spec)
    if match:
        return PageRange(last_n=_positive_count(match.group(1)), raw=spec)

    spans = [_parse_segment(seg.strip()) for seg in spec.split(",") if seg.strip()]
    if not spans:
        raise ValueError(f"No valid page segments found in '{spec}'.")
    return PageRange(spans=tuple(_merge_spans(spans)), raw=spec)


def _positive_count(text: str) -> int:
    n = int(text)
    if n <= 0:
        raise ValueError(f"Page count must be positive, got {n}.")
    return n


def _page_number(text: str, label: str) -> int:
    if not text.isdigit():
        raise ValueError(f"Invalid {label} '{text}'.")
    page = int(text)
    if page <= 0:
        raise ValueError(f"Page numbers are 1-indexed; got {page}.")
    return page


def _parse_segment(seg: str) -> tuple[int, int]:
    """Parse ``3``, ``3-7``, ``3-`` or ``-7`` into a 0-based span."""
    if "-" not in seg:
        idx = _page_number(seg, "page number") - 1
        return (idx, idx)
    left, right = (part.strip() for part in seg.split("-", 1))
    if not left and not right:
        raise ValueError("Invalid range '-'; both sides are empty.")
    if not left:
        return (0, _page_number(right, "range end") - 1)
    if not right:
        return (_page_number(left, "range start") - 1, _OPEN_END)
    start = _page_number(left, "range start")
    end = _page_number(right, "range end")
    if start > end:
        raise ValueError(
            f"Range start ({start}) must not exceed end ({end}) in '{seg}'."
        )
    return (start - 1, end - 1)


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort spans and merge overlapping or adjacent ones."""
    ordered = sorted(spans)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


@dataclass(frozen=True)
class ItemSelection:
    """The outcome of an item-selection expression.

    ``indices`` are 0-based and sorted. ``unmatched`` lists the numeric parts
    that were out of range or malformed and therefore ignored.
    """

    indices: tuple[int, ...]
    unmatched: tuple[str, ...] = ()
    by_name: bool = False

    def pick[T](self, items: Sequence[T]) -> list[T]:
        """Return the selected items in discovery order."""
        return [items[i] for i in self.indices]


def is_numeric_selection(expr: str) -> bool:
    """Whether *expr* is a number/range list rather than a name search.

    Spaces are ignored and ``;`` counts as a separator like ``,``.
    """
    compact = _compact(expr)
    return all(c.isdigit() or c in ",-" for c in compact) and any(
        c.isdigit() for c in compact
    )


def match_names(term: str, names: Sequence[str]) -> set[int]:
    """Return the 0-based indices of *names* containing *term*, ignoring case."""
    needle = term.strip().lower()
    return {idx for idx, name in enumerate(names) if needle in name.lower()}


def select_items(names: Sequence[str], expr: str) -> ItemSelection:
    """Select items by number (``2``), range (``2-3``), list or name search.

    *names* are the item names searched by a non-numeric expression, in
    discovery order. Numbers are 1-based. A range must lie within
    ``1..len(names)`` as a whole; an out-of-range or malformed part is
    reported in ``unmatched`` and the remaining parts still apply.
    """
    expr = expr.strip()
    if not is_numeric_selection(expr):
        return ItemSelection(
            indices=tuple(sorted(match_names(expr, names))), by_name=True
        )

    selected: set[int] = set()
    unmatched: list[str] = []
    count = len(names)
    for part in _compact(expr).split(","):
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            try:
                start, end = int(start_text), int(end_text)
            except ValueError:
                unmatched.append(part)
                continue
            if 1 <= start <= end <= count:
                selected.update(range(start - 1, end))
            else:
                unmatched.append(part)
        elif 1 <= int(part) <= count:
            selected.add(int(part) - 1)
        else:
            unmatched.append(part)
    return ItemSelection(indices=tuple(sorted(selected)), unmatched=tuple(unmatched))


def _compact(expr: str) -> str:
    return expr.strip().replace(" ", "").replace(";", ",")
