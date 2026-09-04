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
