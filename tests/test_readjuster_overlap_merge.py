"""Ranges that resolve to the same start are merged, not trimmed to stubs.

A continuation chunk of a long entry snaps back onto the open entry's
title, so two consecutive ranges end up with the same start. The overlap
resolver used to trim the first to a one-line title stub and shift the
second forward by one line, leaving a headless body. The pair is one entry
and is now merged.
"""

from __future__ import annotations

import pytest

from modules.line_ranges.readjuster import LineRangeReadjuster

pytestmark = pytest.mark.unit


def test_same_start_ranges_merge_into_one() -> None:
    ranges = [(1, 111), (1, 215), (291, 322), (291, 439), (444, 531), (458, 580)]
    assert LineRangeReadjuster._remove_overlaps(ranges) == [
        (1, 215),
        (291, 439),
        (444, 457),
        (458, 580),
    ]


def test_three_way_collision_merges_all() -> None:
    assert LineRangeReadjuster._remove_overlaps(
        [(5, 10), (5, 20), (5, 30), (40, 50)]
    ) == [
        (5, 30),
        (40, 50),
    ]


def test_plain_overlap_still_trims_previous_end() -> None:
    assert LineRangeReadjuster._remove_overlaps([(1, 120), (100, 200)]) == [
        (1, 99),
        (100, 200),
    ]


def test_title_stub_merges_into_next_range() -> None:
    # (2571, 2700) is a title plus blank line; the next range starts two lines
    # later on the body. With min_range_lines the pair becomes one range.
    assert LineRangeReadjuster._remove_overlaps(
        [(2400, 2570), (2571, 2700), (2573, 2800)], min_range_lines=3
    ) == [(2400, 2570), (2571, 2800)]
    # Without the guard the legacy stub behaviour is unchanged.
    assert LineRangeReadjuster._remove_overlaps(
        [(2400, 2570), (2571, 2700), (2573, 2800)]
    ) == [(2400, 2570), (2571, 2572), (2573, 2800)]


def test_enclosed_range_keeps_predecessor_tail() -> None:
    # The last range snapped back to 2108 and encloses (2344, 2472); the
    # lines 2473..2511 must survive on the enclosed range.
    assert LineRangeReadjuster._remove_overlaps([(2108, 2511), (2344, 2472)]) == [
        (2108, 2343),
        (2344, 2511),
    ]


def test_first_range_anchored_to_file_start() -> None:
    ranges = [(156, 300), (301, 400)]
    assert LineRangeReadjuster._anchor_first_range(
        ranges, [(1, 200), (201, 400)], []
    ) == [
        (1, 300),
        (301, 400),
    ]
    # A deleted first mechanical range leaves the ranges untouched.
    assert (
        LineRangeReadjuster._anchor_first_range(ranges, [(1, 200), (201, 400)], [0])
        == ranges
    )
