"""Boundary decisions: the rules that move, keep or delete one line range.

For each range the model looks at a window of lines around the range start
and answers in :data:`SEMANTIC_BOUNDARY_SCHEMA`. :func:`decide_range` routes
the answers:

- an answer below :data:`CERTAINTY_THRESHOLD` moves to the next, wider
  window, at most :data:`MAX_LOW_CERTAINTY_RETRIES` times;
- ``boundary_already_on_target`` keeps the range;
- ``needs_more_context`` moves to the next window;
- ``contains_no_semantic_boundary`` scans the range interior; the range is
  deleted only when every scan finds no content at or above
  :data:`DELETE_CERTAINTY_THRESHOLD`, and a scan that finds content with a
  marker inside the range moves the start there;
- a marker moves the start to the line it matches; a marker past the range
  end moves to the next window, and one that matches nothing is asked for
  again in the same window, at most :data:`MAX_MARKER_MISMATCH_RETRIES`
  times, with the failed markers listed.

Windows grow geometrically from :data:`DEFAULT_CONTEXT_WINDOW` lines on
each side of the start, at most :data:`MAX_WINDOWS` of them. When the
windows run out, or certainty stays low, the interior scan decides between
keeping and deleting the range. A range whose last window still asked for
more context, or named a start past the range end, lies inside an entry
that began before it: when the scan, too, asks for more context with
certainty in a window that reaches the range end, it is merged into the
range before it instead of keeping a cut inside the entry.

:func:`final_ranges` turns the decisions of a file into its ranges: deleted
ranges drop out and merged ranges join their predecessor once every range is
decided, overlaps are trimmed or merged (:func:`remove_overlaps`) and the
first range is anchored to the file start.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from chronominer.ranges.lines import Range
from chronominer.ranges.matching import match_marker
from chronominer.spec import DEFAULT_MARKER_LENGTH

__all__ = [
    "ANCHOR_FIRST_RANGE",
    "CERTAINTY_THRESHOLD",
    "DEFAULT_CONTEXT_WINDOW",
    "DELETE_CERTAINTY_THRESHOLD",
    "DELETE_RANGES_WITHOUT_CONTENT",
    "MAX_GAP_BETWEEN_RANGES",
    "MAX_LOW_CERTAINTY_RETRIES",
    "MAX_MARKER_MISMATCH_RETRIES",
    "MAX_MERGED_RANGES",
    "MAX_VERIFY_WINDOW_LINES",
    "MAX_WINDOWS",
    "MIN_RANGE_LINES",
    "SEMANTIC_BOUNDARY_SCHEMA",
    "Ask",
    "BoundaryDecision",
    "DecidedRange",
    "RangeResult",
    "anchor_first_range",
    "boundary_schema",
    "context_windows",
    "decide_range",
    "enforce_max_gap",
    "final_layout",
    "final_ranges",
    "remove_overlaps",
    "unsure_payload",
    "verify_no_content",
]

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_WINDOW = 6
"""Lines shown on each side of the range start in the first window."""
CERTAINTY_THRESHOLD = 70
MAX_LOW_CERTAINTY_RETRIES = 3
MAX_WINDOWS = 7
"""Boundary windows per range; the last reaches 64 times the first radius
on each side, 384 lines at the default."""
MAX_MARKER_MISMATCH_RETRIES = 2
DELETE_RANGES_WITHOUT_CONTENT = True
DELETE_CERTAINTY_THRESHOLD = 85
"""Deleting a range drops its lines from extraction, so it needs more
certainty than moving a boundary."""
MAX_VERIFY_WINDOW_LINES = 1000
"""Lines per interior scan; longer ranges are scanned in consecutive parts."""
MIN_RANGE_LINES = 3
"""A range this short after trimming merges into its neighbor; 0 disables."""
MAX_MERGED_RANGES = 3
"""Merged ranges one range may absorb; a further one keeps its bounds, so a
run of unresolved ranges cannot fold a whole text into one chunk."""
ANCHOR_FIRST_RANGE = True
MAX_GAP_BETWEEN_RANGES: int | None = 0
"""Unassigned lines allowed between ranges; a wider gap extends the range
before it, short of deleted spans, and the lines after a deleted span join
the range after it (:func:`enforce_max_gap`). At 0 no line between two
ranges is lost outside a deleted span. None leaves gaps alone."""

SEMANTIC_BOUNDARY_SCHEMA: Mapping[str, Any] = {
    "name": "SemanticBoundaryResponse",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "contains_no_semantic_boundary": {
                "type": "boolean",
                "description": (
                    "Set to true when the provided text contains NO content of the"
                    " required semantic type at all. Use this when you are confident"
                    " no relevant content exists anywhere in the visible context."
                ),
            },
            "needs_more_context": {
                "type": "boolean",
                "description": (
                    "Set to true when you believe the semantic boundary exists"
                    " somewhere around the visible text but you need to see more"
                    " surrounding content to accurately identify it."
                ),
            },
            "boundary_already_on_target": {
                "type": "boolean",
                "description": (
                    "Set to true when the current range start is already positioned"
                    " at a semantic boundary and no adjustment is needed. Leave"
                    " semantic_marker empty when this is true."
                ),
            },
            "certainty": {
                "type": "integer",
                "description": (
                    "Your confidence level in this response as an integer from 0-100."
                    " Use 0-40 for low confidence, 41-70 for moderate confidence,"
                    " 71-100 for high confidence. This applies to whatever decision"
                    " you make (boundary found, no content, or needs more context)."
                ),
            },
            "semantic_marker": {
                "type": "string",
                "description": (
                    "A precise 5-15 character verbatim substring that marks the"
                    " semantic boundary. Leave empty if contains_no_semantic_boundary"
                    " or needs_more_context is true."
                ),
            },
        },
        "required": [
            "contains_no_semantic_boundary",
            "needs_more_context",
            "boundary_already_on_target",
            "certainty",
            "semantic_marker",
        ],
        "additionalProperties": False,
    },
}
"""The answer schema at the default marker length; see :func:`boundary_schema`."""


def boundary_schema(
    marker_length: tuple[int, int] = DEFAULT_MARKER_LENGTH,
) -> dict[str, Any]:
    """Return the answer schema that asks for markers of *marker_length* characters."""
    schema = copy.deepcopy(dict(SEMANTIC_BOUNDARY_SCHEMA))
    low, high = marker_length
    schema["schema"]["properties"]["semantic_marker"]["description"] = (
        f"A precise {low}-{high} character verbatim substring that marks the"
        " semantic boundary. Leave empty if contains_no_semantic_boundary"
        " or needs_more_context is true."
    )
    return schema


Ask = Callable[[Range, int, Sequence[str]], Awaitable[Mapping[str, Any]]]
"""``(window, window_index, failed_markers) -> answer`` for one range.

The answer is a payload of the boundary schema; an unusable answer is
:func:`unsure_payload`. Interior scans have negative window indices.
"""


def unsure_payload() -> dict[str, Any]:
    """Return the neutral answer used when no usable answer arrived.

    Certainty 0 with every flag false counts as a low-certainty answer and
    never deletes a range.
    """
    return {
        "contains_no_semantic_boundary": False,
        "needs_more_context": False,
        "boundary_already_on_target": False,
        "certainty": 0,
        "semantic_marker": "",
    }


def _certainty(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class BoundaryDecision:
    """One answer of the model in the boundary schema."""

    contains_no_semantic_boundary: bool
    needs_more_context: bool
    certainty: int
    boundary_already_on_target: bool = False
    semantic_marker: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> BoundaryDecision:
        """Read an answer; missing fields count as false, empty markers as None."""
        marker = payload.get("semantic_marker")
        return cls(
            contains_no_semantic_boundary=bool(
                payload.get("contains_no_semantic_boundary", False)
            ),
            needs_more_context=bool(payload.get("needs_more_context", False)),
            certainty=_certainty(payload.get("certainty", 0)),
            boundary_already_on_target=bool(
                payload.get("boundary_already_on_target", False)
            ),
            semantic_marker=str(marker) if marker else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the decision as stored."""
        return {
            "contains_no_semantic_boundary": self.contains_no_semantic_boundary,
            "needs_more_context": self.needs_more_context,
            "boundary_already_on_target": self.boundary_already_on_target,
            "certainty": self.certainty,
            "semantic_marker": self.semantic_marker,
        }


@dataclass(frozen=True)
class RangeResult:
    """The decision on one range and the record of every answer behind it.

    *index* is the range's 1-based position in the file; *calls* counts
    the model answers. *merged* marks a range inside an entry that began
    before it, which joins the range before it; it is stored only when true.
    """

    index: int
    original: Range
    adjusted: Range
    should_delete: bool
    decision: BoundaryDecision
    attempts: tuple[Mapping[str, Any], ...] = ()
    calls: int = 0
    merged: bool = False

    @property
    def outcome(self) -> str:
        """Return ``deleted``, ``merged``, ``moved``, ``on_target`` or ``kept``."""
        if self.should_delete:
            return "deleted"
        if self.merged:
            return "merged"
        if self.adjusted != self.original:
            return "moved"
        if self.decision.boundary_already_on_target:
            return "on_target"
        return "kept"

    def to_body(self) -> dict[str, Any]:
        """Return the stored form of the result."""
        body: dict[str, Any] = {
            "original_range": list(self.original),
            "adjusted_range": list(self.adjusted),
            "should_delete": self.should_delete,
            "decision": self.decision.to_dict(),
            "attempts": [dict(attempt) for attempt in self.attempts],
            "total_llm_calls": self.calls,
        }
        if self.merged:
            body["merged"] = True
        return body

    @classmethod
    def from_body(cls, index: int, body: Mapping[str, Any]) -> RangeResult:
        """Read a stored result back."""
        original = body.get("original_range") or (0, 0)
        adjusted = body.get("adjusted_range") or original
        return cls(
            index=index,
            original=(int(original[0]), int(original[1])),
            adjusted=(int(adjusted[0]), int(adjusted[1])),
            should_delete=bool(body.get("should_delete", False)),
            decision=BoundaryDecision.from_payload(body.get("decision") or {}),
            attempts=tuple(body.get("attempts") or ()),
            calls=int(body.get("total_llm_calls") or 0),
            merged=bool(body.get("merged", False)),
        )


def context_windows(start: int, total_lines: int, radius: int) -> Iterator[Range]:
    """Yield up to :data:`MAX_WINDOWS` windows around line *start*.

    The radius doubles each time; growth stops once a window spans the
    whole text.
    """
    previous: Range | None = None
    for step in range(MAX_WINDOWS):
        reach = radius * (2**step)
        window = (max(1, start - reach), min(total_lines, start + reach))
        if window[1] < window[0] or window == previous:
            return
        previous = window
        yield window
        if window == (1, total_lines):
            return


def _attempt(
    window: Range, index: int, kind: str, decision: BoundaryDecision, **extra: Any
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "window": list(window),
        "window_index": index,
        "decision_type": kind,
        "certainty": decision.certainty,
        "semantic_marker": decision.semantic_marker,
        "marker_matched": False,
    }
    record.update(extra)
    return record


def _scan_windows(start: int, end: int) -> list[Range]:
    windows: list[Range] = []
    first = start
    while first <= end:
        last = min(end, first + MAX_VERIFY_WINDOW_LINES - 1)
        windows.append((first, last))
        first = last + 1
    return windows


async def verify_no_content(
    ask: Ask,
    lines: Sequence[str],
    original: Range,
    index: int,
    *,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    boundary_type: str = "",
    min_marker_length: int = DEFAULT_MARKER_LENGTH[0],
) -> tuple[bool, Range | None, list[dict[str, Any]]]:
    """Scan the interior of *original* for content of the semantic type.

    Returns whether to delete the range, the range moved to a marker the
    scan found inside it (or None), and the scan's attempt records. Any
    content signal keeps the range; so does a no-content answer below
    :data:`DELETE_CERTAINTY_THRESHOLD`. Markers shorter than
    *min_marker_length* match nothing.
    """
    start, end = original
    attempts: list[dict[str, Any]] = []
    windows = _scan_windows(start, end)
    logger.info(
        "[Range %d] Verifying no content: %d interior window(s) within %d-%d",
        index,
        len(windows),
        start,
        end,
    )
    for position, window in enumerate(windows):
        window_index = -(position + 1)
        decision = BoundaryDecision.from_payload(await ask(window, window_index, ()))
        found = not decision.contains_no_semantic_boundary
        attempts.append(
            {
                "window": list(window),
                "window_index": window_index,
                "decision_type": f"verify_interior_{position + 1}",
                "certainty": decision.certainty,
                "semantic_marker": decision.semantic_marker,
                "found_content": found,
            }
        )
        if found:
            moved: Range | None = None
            if decision.semantic_marker:
                line = match_marker(
                    decision.semantic_marker,
                    lines,
                    window,
                    nearest_to=start,
                    widen_by=context_window,
                    min_length=min_marker_length,
                )
                if line is not None and start <= line <= end:
                    moved = (line, end)
            logger.info(
                "[Range %d] The interior scan found content (marker %r); %s",
                index,
                decision.semantic_marker or "<none>",
                f"moving the start to {moved}" if moved else "keeping the range",
            )
            return False, moved, attempts
        if decision.certainty < DELETE_CERTAINTY_THRESHOLD:
            logger.info(
                "[Range %d] No-content answer below the delete threshold "
                "(%d < %d); keeping the range",
                index,
                decision.certainty,
                DELETE_CERTAINTY_THRESHOLD,
            )
            return False, None, attempts
    lowest = min((attempt["certainty"] for attempt in attempts), default=0)
    logger.warning(
        "[Range %d] DELETION accepted: dropping lines %d-%d (no '%s' content; "
        "model certainty %d >= delete threshold %d across %d verification "
        "window(s))",
        index,
        start,
        end,
        boundary_type,
        lowest,
        DELETE_CERTAINTY_THRESHOLD,
        len(attempts),
    )
    return True, None, attempts


_START_BEFORE_RANGE = frozenset({"needs_more_context", "marker_beyond_end"})
"""Last-window answers that place the range inside an earlier entry."""


@dataclass
class _Search:
    """The running state of :func:`decide_range`."""

    original: Range
    adjusted: Range
    attempts: list[dict[str, Any]] = field(default_factory=list)
    calls: int = 0
    low_certainty: int = 0
    failed_markers: list[str] = field(default_factory=list)
    decision: BoundaryDecision | None = None
    scan: BoundaryDecision | None = None
    """The interior scan's last answer, if a scan ran."""
    scan_end: int = 0
    """The last line of the window that answer judged."""
    stop: bool = False
    exhausted_low: bool = False
    delete: bool = False
    min_marker_length: int = DEFAULT_MARKER_LENGTH[0]


async def decide_range(
    ask: Ask,
    lines: Sequence[str],
    original: Range,
    index: int,
    *,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    boundary_type: str = "",
    min_marker_length: int = DEFAULT_MARKER_LENGTH[0],
) -> RangeResult:
    """Decide where range *index* should start, or whether to delete it.

    Markers shorter than *min_marker_length* match nothing and are asked
    for again.
    """
    state = _Search(
        original=original, adjusted=original, min_marker_length=min_marker_length
    )
    radius = max(1, context_window)
    for window_index, window in enumerate(
        context_windows(original[0], len(lines), radius)
    ):
        await _search_window(
            ask, lines, state, (window, window_index), radius, index, boundary_type
        )
        if state.stop:
            break
    if state.delete:
        return _result(state, index, delete=True)
    inside = (
        not state.stop
        and bool(state.attempts)
        and state.attempts[-1]["decision_type"] in _START_BEFORE_RANGE
    )
    if (not state.stop or state.exhausted_low) and DELETE_RANGES_WITHOUT_CONTENT:
        logger.info(
            "[Range %d] Boundary windows exhausted; scanning the interior", index
        )
        await _verify(ask, lines, state, index, radius, boundary_type)
        if state.delete:
            return _result(state, index, delete=True)
    scan = state.scan
    agrees = scan is None or (
        scan.needs_more_context
        and scan.certainty >= CERTAINTY_THRESHOLD
        and state.scan_end >= original[1]
    )
    merge = inside and agrees and state.adjusted == state.original
    if merge:
        logger.warning(
            "[Range %d] No entry starts in reach of line %d; merging the range "
            "into the one before it",
            index,
            original[0],
        )
    return _result(state, index, delete=False, merge=merge)


def _result(
    state: _Search, index: int, *, delete: bool, merge: bool = False
) -> RangeResult:
    decision = state.decision or BoundaryDecision(
        contains_no_semantic_boundary=True, needs_more_context=False, certainty=0
    )
    return RangeResult(
        index=index,
        original=state.original,
        adjusted=state.original if delete else state.adjusted,
        should_delete=delete,
        decision=decision,
        attempts=tuple(state.attempts),
        calls=state.calls,
        merged=merge,
    )


async def _verify(
    ask: Ask,
    lines: Sequence[str],
    state: _Search,
    index: int,
    radius: int,
    boundary_type: str,
) -> None:
    async def scan(
        window: Range, window_index: int, failed: Sequence[str]
    ) -> Mapping[str, Any]:
        payload = await ask(window, window_index, failed)
        state.scan = BoundaryDecision.from_payload(payload)
        state.scan_end = window[1]
        return payload

    delete, moved, attempts = await verify_no_content(
        scan,
        lines,
        state.original,
        index,
        context_window=radius,
        boundary_type=boundary_type,
        min_marker_length=state.min_marker_length,
    )
    state.calls += len(attempts)
    state.attempts.extend(attempts)
    state.delete = delete
    if moved is not None:
        state.adjusted = moved


async def _search_window(
    ask: Ask,
    lines: Sequence[str],
    state: _Search,
    position: tuple[Range, int],
    radius: int,
    index: int,
    boundary_type: str,
) -> None:
    """Ask about one window until it yields a decision or the next window."""
    window, window_index = position
    mismatches = 0
    while True:
        payload = await ask(window, window_index, tuple(state.failed_markers))
        decision = BoundaryDecision.from_payload(payload)
        state.decision = decision
        state.calls += 1
        if decision.certainty < CERTAINTY_THRESHOLD:
            _low_certainty(state, window, window_index, decision, index)
            return
        if decision.boundary_already_on_target:
            state.attempts.append(
                _attempt(window, window_index, "already_on_target", decision)
            )
            state.stop = True
            return
        if decision.needs_more_context:
            state.attempts.append(
                _attempt(window, window_index, "needs_more_context", decision)
            )
            return
        if decision.contains_no_semantic_boundary:
            state.attempts.append(
                _attempt(window, window_index, "no_semantic_boundary", decision)
            )
            if DELETE_RANGES_WITHOUT_CONTENT:
                await _verify(ask, lines, state, index, radius, boundary_type)
            state.stop = True
            return
        line = match_marker(
            decision.semantic_marker,
            lines,
            window,
            nearest_to=state.original[0],
            widen_by=radius,
            min_length=state.min_marker_length,
        )
        if line is not None and line > state.original[1]:
            state.attempts.append(
                _attempt(
                    window,
                    window_index,
                    "marker_beyond_end",
                    decision,
                    matched_line=line,
                )
            )
            return
        if line is not None:
            state.adjusted = (line, state.original[1])
            record = _attempt(window, window_index, "marker_found", decision)
            record["marker_matched"] = True
            state.attempts.append(record)
            logger.info(
                "[Range %d] Moved to %s by marker %r (certainty %d)",
                index,
                state.adjusted,
                decision.semantic_marker,
                decision.certainty,
            )
            state.stop = True
            return
        state.attempts.append(
            _attempt(window, window_index, "marker_mismatch", decision)
        )
        marker = (decision.semantic_marker or "").strip()
        if marker and marker not in state.failed_markers:
            state.failed_markers.append(marker)
        mismatches += 1
        if mismatches > MAX_MARKER_MISMATCH_RETRIES:
            logger.warning(
                "[Range %d, window %d] Marker not matched after %d retries; "
                "trying the next window",
                index,
                window_index,
                MAX_MARKER_MISMATCH_RETRIES,
            )
            return


def _low_certainty(
    state: _Search,
    window: Range,
    window_index: int,
    decision: BoundaryDecision,
    index: int,
) -> None:
    state.attempts.append(_attempt(window, window_index, "low_certainty", decision))
    state.low_certainty += 1
    if state.low_certainty > MAX_LOW_CERTAINTY_RETRIES:
        logger.warning(
            "[Range %d] Certainty stays low after %d retries",
            index,
            MAX_LOW_CERTAINTY_RETRIES,
        )
        state.exhausted_low = True
        state.stop = True


@dataclass(frozen=True)
class DecidedRange:
    """One range of a file: its starting bounds and its result, if decided."""

    original: Range
    result: RangeResult | None = None


def final_layout(entries: Sequence[DecidedRange]) -> list[tuple[int, Range]]:
    """Return the ranges of a file and the entry each one is kept under.

    An undecided range keeps its bounds. Deleted ranges drop out, and a
    merged range extends the range before it, only once every range is
    decided; until then they keep their bounds, so a later run finds every
    range at its position. A merged range first in the file, after a
    deleted one, or beyond :data:`MAX_MERGED_RANGES` keeps its bounds. A
    failed range counts as undecided. Overlaps are then removed, gaps closed
    (:data:`MAX_GAP_BETWEEN_RANGES`) and the first range anchored to the
    start of the first starting range.

    Each range comes with the position of an entry; the positions ascend
    with the ranges, and the entries of deleted or merged ranges get none.
    """
    originals = [entry.original for entry in entries]
    complete = all(entry.result is not None for entry in entries)
    kept: list[tuple[int, Range]] = []
    deleted: list[int] = []
    merged: list[int] = []
    absorbed = 0
    for position, entry in enumerate(entries):
        result = entry.result
        if complete and result is not None and result.should_delete:
            deleted.append(position)
            continue
        if (
            complete
            and result is not None
            and result.merged
            and kept
            and position - 1 not in deleted
        ):
            if absorbed < MAX_MERGED_RANGES:
                owner, (start, end) = kept[-1]
                kept[-1] = (owner, (start, max(end, result.adjusted[1])))
                merged.append(position)
                absorbed += 1
                continue
            logger.warning(
                "Range %d stays on its own: the range before it already absorbed "
                "%d merged ranges",
                position + 1,
                MAX_MERGED_RANGES,
            )
        if result is None or result.should_delete:
            kept.append((position, entry.original))
        else:
            kept.append((position, result.adjusted))
        absorbed = 0
    owned = _remove_overlaps(kept, MIN_RANGE_LINES)
    owners = sorted(position for position, _ in owned)
    ranges = [bounds for _, bounds in owned]
    if MAX_GAP_BETWEEN_RANGES is not None:
        spans = [originals[position] for position in deleted]
        ranges = enforce_max_gap(ranges, MAX_GAP_BETWEEN_RANGES, spans)
    if ANCHOR_FIRST_RANGE:
        ranges = anchor_first_range(ranges, originals, deleted)
    if deleted:
        logger.info(
            "Deleted %d range(s) without semantic content: %s",
            len(deleted),
            [position + 1 for position in deleted],
        )
    if merged:
        logger.info(
            "Merged %d range(s) into the range before them: %s",
            len(merged),
            [position + 1 for position in merged],
        )
    return list(zip(owners, ranges, strict=True))


def final_ranges(entries: Sequence[DecidedRange]) -> list[Range]:
    """Return the ranges of a file from its starting ranges and decisions.

    See :func:`final_layout`.
    """
    return [bounds for _, bounds in final_layout(entries)]


def anchor_first_range(
    ranges: Sequence[Range], originals: Sequence[Range], deleted: Sequence[int]
) -> list[Range]:
    """Pull the first range back to the start of the first starting range.

    No range before it would hold the lines above a moved first start. A
    deleted first range (index 0 in *deleted*) leaves the ranges as they are.
    """
    if not ranges or not originals or 0 in set(deleted):
        return list(ranges)
    first = originals[0][0]
    start, end = ranges[0]
    if start > first:
        logger.info("Anchored the first range (%d, %d) to line %d", start, end, first)
        return [(first, end), *ranges[1:]]
    return list(ranges)


def remove_overlaps(ranges: Sequence[Range], min_range_lines: int = 0) -> list[Range]:
    """Return *ranges* without overlaps, keeping the decided starts.

    Ranges are taken in order of start. A range overlapping its predecessor
    trims the predecessor's end. Two ranges with the same start, or a range
    whose trimming would leave its predecessor with at most
    *min_range_lines* lines, are one unit and merge. A range enclosed by its
    predecessor takes over the predecessor's end. Gaps stay.
    """
    owned = _remove_overlaps(list(enumerate(ranges)), min_range_lines)
    return [bounds for _, bounds in owned]


def _remove_overlaps(
    ranges: Sequence[tuple[int, Range]], min_range_lines: int
) -> list[tuple[int, Range]]:
    """Remove overlaps from ``(owner, range)`` pairs; merged pairs drop out."""
    ordered = sorted(ranges, key=lambda item: (item[1][0], item[0]))
    merged: list[list[int]] = []
    for owner, (start, end) in ordered:
        end = max(end, start)
        if merged:
            previous = merged[-1]
            if previous[2] > end:
                end = previous[2]
            stub = (
                min_range_lines > 0
                and previous[2] >= start
                and start - previous[1] <= min_range_lines
            )
            if start <= previous[1] or stub:
                previous[2] = max(previous[2], end)
                continue
            if previous[2] >= start:
                previous[2] = min(previous[2], max(previous[1], start - 1))
            if previous[2] >= start:
                start = previous[2] + 1
        merged.append([owner, start, max(end, start)])
    return [(owner, (start, end)) for owner, start, end in merged]


def enforce_max_gap(
    ranges: Sequence[Range], max_gap: int, deleted_spans: Sequence[Range] = ()
) -> list[Range]:
    """Extend each range so at most *max_gap* lines precede the next one.

    The extension never reaches into a deleted span, which would undo the
    deletion. Lines between a deleted span and the next range go to that
    range instead: its start moves back to at most *max_gap* lines after
    the span.
    """
    enforced: list[Range] = []
    for start, end in ranges:
        if enforced:
            previous_start, previous_end = enforced[-1]
            if start - previous_end - 1 > max_gap:
                new_end = start - max_gap - 1
                spans = [
                    (span_start, span_end)
                    for span_start, span_end in deleted_spans
                    if span_end > previous_end and span_start < start
                ]
                for span_start, _ in spans:
                    if span_start <= new_end:
                        new_end = min(new_end, span_start - 1)
                if new_end > previous_end:
                    enforced[-1] = (previous_start, new_end)
                if spans:
                    after = max(span_end for _, span_end in spans)
                    start = min(start, max(after, new_end) + max_gap + 1)
        enforced.append((start, end))
    return enforced
