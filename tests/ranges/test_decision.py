"""Boundary decisions on one range and the ranges of a file."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from chronominer.ranges import decision as decision_module
from chronominer.ranges.decision import (
    MAX_LOW_CERTAINTY_RETRIES,
    BoundaryDecision,
    DecidedRange,
    RangeResult,
    anchor_first_range,
    context_windows,
    decide_range,
    enforce_max_gap,
    final_layout,
    final_ranges,
    remove_overlaps,
    verify_no_content,
)
from tests.ranges.fakes import boundary

pytestmark = pytest.mark.unit

Window = tuple[int, int]


class Script:
    """An ``Ask`` that answers boundary and interior-scan windows apart."""

    def __init__(
        self,
        answer: Callable[[Window, int], Mapping[str, Any]],
        scan: Mapping[str, Any] | None = None,
    ) -> None:
        self.answer = answer
        self.scan = scan if scan is not None else boundary(certainty=90)
        self.calls: list[tuple[Window, int, tuple[str, ...]]] = []

    async def __call__(
        self, window: Window, index: int, failed: Sequence[str]
    ) -> Mapping[str, Any]:
        self.calls.append((window, index, tuple(failed)))
        if index < 0:
            return self.scan
        return self.answer(window, index)

    @property
    def boundary_calls(self) -> int:
        return sum(1 for _, index, _ in self.calls if index >= 0)


def always(payload: Mapping[str, Any]) -> Callable[[Window, int], Mapping[str, Any]]:
    return lambda _window, _index: payload


def test_windows_grow_geometrically_and_stop_at_the_whole_text() -> None:
    assert list(context_windows(5, 40, 20)) == [(1, 25), (1, 40)]
    windows = list(context_windows(50, 1000, 3))
    assert windows[:3] == [(47, 53), (44, 56), (38, 62)]
    assert len(windows) == 7


def test_decision_round_trips() -> None:
    decision = BoundaryDecision(False, False, 85, boundary_already_on_target=True)
    assert BoundaryDecision.from_payload(decision.to_dict()) == decision
    partial = BoundaryDecision.from_payload({"certainty": "80", "semantic_marker": ""})
    assert partial.boundary_already_on_target is False
    assert partial.semantic_marker is None
    assert partial.certainty == 80
    assert BoundaryDecision(False, True, 55).to_dict() == {
        "contains_no_semantic_boundary": False,
        "needs_more_context": True,
        "boundary_already_on_target": False,
        "certainty": 55,
        "semantic_marker": None,
    }


def test_result_body_round_trips_through_json() -> None:
    result = RangeResult(
        index=3,
        original=(10, 50),
        adjusted=(12, 50),
        should_delete=False,
        decision=BoundaryDecision(False, False, 92, semantic_marker="Chapter III"),
        attempts=({"window": [4, 16], "decision_type": "marker_found"},),
        calls=1,
    )
    body = json.loads(json.dumps(result.to_body()))
    assert body["original_range"] == [10, 50]
    assert body["adjusted_range"] == [12, 50]
    assert body["decision"]["semantic_marker"] == "Chapter III"
    assert body["total_llm_calls"] == 1
    assert RangeResult.from_body(3, body) == result
    assert result.outcome == "moved"


ENTRIES = [f"Entry number {i} begins here" for i in range(1, 41)]


async def test_a_marker_past_the_range_end_moves_to_the_next_window() -> None:
    script = Script(always(boundary(marker="Entry number 11 begins here")))

    result = await decide_range(script, ENTRIES, (5, 10), 1, context_window=20)

    assert script.boundary_calls == 2
    assert result.adjusted == (5, 10)
    assert result.should_delete is False
    beyond = [a for a in result.attempts if a["decision_type"] == "marker_beyond_end"]
    assert len(beyond) == 2
    assert all(attempt["matched_line"] == 11 for attempt in beyond)


async def test_a_marker_inside_the_range_moves_the_start() -> None:
    script = Script(always(boundary(marker="Entry number 3 begins here")))

    result = await decide_range(script, ENTRIES, (5, 10), 1, context_window=20)

    assert len(script.calls) == 1
    assert result.adjusted == (3, 10)
    assert result.attempts[-1]["decision_type"] == "marker_found"
    assert result.attempts[-1]["marker_matched"] is True


async def test_an_unmatched_marker_is_asked_again_with_the_failed_markers() -> None:
    answers = iter(
        [
            boundary(marker="Nothing like this"),
            boundary(marker="Entry number 4 begins here"),
        ]
    )
    script = Script(lambda _window, _index: next(answers))

    result = await decide_range(script, ENTRIES, (5, 10), 1, context_window=3)

    assert result.adjusted == (4, 10)
    assert script.calls[0][2] == ()
    assert script.calls[1][2] == ("Nothing like this",)
    assert script.calls[0][0] == script.calls[1][0]


async def test_a_boundary_move_needs_only_the_move_threshold() -> None:
    lines = [
        "filler alpha",
        "filler beta",
        "filler gamma",
        "1864. Host and Guest, a book about dinners.",
        "continuation of the entry",
        "more continuation",
        "filler delta",
        "filler epsilon",
        "filler zeta",
        "filler eta",
    ]
    script = Script(always(boundary(marker="Host and Guest", certainty=75)))

    result = await decide_range(script, lines, (5, 10), 1, context_window=3)

    assert result.should_delete is False
    assert result.adjusted == (4, 10)
    assert all(a["decision_type"] != "low_certainty" for a in result.attempts)


async def test_low_certainty_exhausts_and_falls_back_to_the_interior_scan() -> None:
    script = Script(always(boundary(on_target=True, certainty=40)))

    result = await decide_range(script, ENTRIES, (20, 30), 1, context_window=1)

    assert script.boundary_calls == MAX_LOW_CERTAINTY_RETRIES + 1
    assert script.calls[-1][1] == -1
    assert result.adjusted == (20, 30)
    assert result.should_delete is False


async def test_on_target_keeps_the_range_in_one_call() -> None:
    script = Script(always(boundary(on_target=True)))

    result = await decide_range(script, ENTRIES, (5, 10), 1)

    assert len(script.calls) == 1
    assert result.outcome == "on_target"


async def test_no_content_confirmed_by_the_scan_deletes_the_range() -> None:
    script = Script(always(boundary(no_content=True)), scan=boundary(no_content=True))

    result = await decide_range(script, ENTRIES, (5, 10), 1)

    assert result.should_delete is True
    assert result.outcome == "deleted"
    assert result.calls == 2
    assert result.attempts[-1]["decision_type"] == "verify_interior_1"


async def test_the_scan_moves_the_start_to_content_it_finds() -> None:
    script = Script(
        always(boundary(no_content=True)),
        scan=boundary(marker="Entry number 8 begins here"),
    )

    result = await decide_range(script, ENTRIES, (5, 10), 1)

    assert result.should_delete is False
    assert result.adjusted == (8, 10)


LEDGER = [f"Line {i}" for i in range(1, 21)]


async def test_a_no_content_answer_below_the_delete_threshold_keeps() -> None:
    script = Script(always({}), scan=boundary(no_content=True, certainty=80))

    delete, moved, attempts = await verify_no_content(script, LEDGER, (5, 12), 1)

    assert (delete, moved, len(attempts)) == (False, None, 1)


async def test_an_accepted_deletion_is_logged_with_its_span(
    caplog: pytest.LogCaptureFixture,
) -> None:
    script = Script(always({}), scan=boundary(no_content=True, certainty=90))

    with caplog.at_level(logging.WARNING, logger=decision_module.__name__):
        delete, _, _ = await verify_no_content(script, LEDGER, (5, 12), 1)

    assert delete is True
    assert any(
        "DELETION accepted" in record.getMessage()
        and "5-12" in record.getMessage()
        and "90" in record.getMessage()
        for record in caplog.records
    )


async def test_long_ranges_are_scanned_in_consecutive_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(decision_module, "MAX_VERIFY_WINDOW_LINES", 4)
    script = Script(always({}), scan=boundary(no_content=True, certainty=95))

    delete, _, attempts = await verify_no_content(script, LEDGER, (1, 10), 1)

    assert delete is True
    assert [a["window"] for a in attempts] == [[1, 4], [5, 8], [9, 10]]


def test_same_start_ranges_merge_into_one() -> None:
    ranges = [(1, 111), (1, 215), (291, 322), (291, 439), (444, 531), (458, 580)]
    assert remove_overlaps(ranges) == [(1, 215), (291, 439), (444, 457), (458, 580)]
    assert remove_overlaps([(5, 10), (5, 20), (5, 30), (40, 50)]) == [
        (5, 30),
        (40, 50),
    ]


def test_plain_overlap_trims_the_previous_end() -> None:
    assert remove_overlaps([(1, 120), (100, 200)]) == [(1, 99), (100, 200)]


def test_a_title_stub_merges_into_the_next_range() -> None:
    ranges = [(2400, 2570), (2571, 2700), (2573, 2800)]
    assert remove_overlaps(ranges, min_range_lines=3) == [(2400, 2570), (2571, 2800)]
    assert remove_overlaps(ranges) == [(2400, 2570), (2571, 2572), (2573, 2800)]


def test_an_enclosed_range_keeps_the_predecessor_tail() -> None:
    assert remove_overlaps([(2108, 2511), (2344, 2472)]) == [
        (2108, 2343),
        (2344, 2511),
    ]


def test_the_first_range_is_anchored_to_the_file_start() -> None:
    ranges = [(156, 300), (301, 400)]
    originals = [(1, 200), (201, 400)]
    assert anchor_first_range(ranges, originals, []) == [(1, 300), (301, 400)]
    assert anchor_first_range(ranges, originals, [0]) == ranges


def test_gap_enforcement_stops_before_deleted_spans() -> None:
    ranges = [(1, 10), (30, 40)]
    assert enforce_max_gap(ranges, 2, []) == [(1, 27), (30, 40)]
    assert enforce_max_gap(ranges, 2, [(15, 20)]) == [(1, 14), (23, 40)]


def _decided(
    original: tuple[int, int],
    adjusted: tuple[int, int] | None = None,
    *,
    delete: bool = False,
) -> DecidedRange:
    result = RangeResult(
        index=1,
        original=original,
        adjusted=adjusted or original,
        should_delete=delete,
        decision=BoundaryDecision(delete, False, 90),
    )
    return DecidedRange(original, result)


def test_deletions_wait_until_every_range_is_decided() -> None:
    entries = [
        _decided((1, 10)),
        _decided((11, 20), delete=True),
        DecidedRange((21, 30)),
    ]
    assert final_ranges(entries) == [(1, 10), (11, 20), (21, 30)]
    entries[2] = _decided((21, 30))
    assert final_ranges(entries) == [(1, 10), (21, 30)]


def test_layout_names_the_entry_of_each_range() -> None:
    entries = [
        _decided((1, 4)),
        _decided((5, 20), (1, 20)),
        _decided((21, 38), (25, 38)),
    ]
    assert final_layout(entries) == [(0, (1, 24)), (2, (25, 38))]
    assert final_ranges([DecidedRange((3, 9))]) == [(3, 9)]


def test_a_forward_move_extends_the_preceding_range() -> None:
    entries = [_decided((1, 10)), _decided((11, 20), (14, 20)), _decided((21, 30))]
    assert final_ranges(entries) == [(1, 13), (14, 20), (21, 30)]


@pytest.mark.parametrize("gap", [1, 2])
def test_no_line_is_lost_before_a_moved_start(gap: int) -> None:
    entries = [_decided((1, 10)), _decided((11, 20), (11 + gap, 20))]
    assert final_ranges(entries) == [(1, 10 + gap), (11 + gap, 20)]


def test_lines_after_a_deleted_span_join_the_next_range() -> None:
    entries = [
        _decided((1, 10)),
        _decided((11, 20), delete=True),
        _decided((21, 30), (23, 30)),
    ]
    assert final_ranges(entries) == [(1, 10), (21, 30)]
