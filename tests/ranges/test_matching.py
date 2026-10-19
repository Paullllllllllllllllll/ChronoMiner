"""Marker matching: normalization, the length floor and the bounded search."""

from __future__ import annotations

import pytest

from chronominer.ranges.matching import (
    MIN_MARKER_LENGTH,
    match_marker,
    matching_lines,
    normalize,
)

pytestmark = pytest.mark.unit

LINES = [
    "Preface of the editor",
    "  Gâteau  de   Savoie, a cake",
    "Take six eggs",
    "GATEAU DE SAVOIE again",
    "Bake slowly",
]


def test_normalize_collapses_space_case_and_diacritics() -> None:
    assert normalize("  Gâteau  de   Savoie ") == "gateau de savoie"


def test_substring_matches_after_normalization() -> None:
    assert matching_lines(LINES, "Gateau de Savoie", (1, 5)) == [2, 4]
    assert matching_lines(LINES, "Gateau de Savoie", (3, 4)) == [4]


def test_short_and_empty_markers_are_rejected() -> None:
    short = "x" * (MIN_MARKER_LENGTH - 1)
    assert match_marker(short, [short], (1, 1), nearest_to=1, widen_by=3) is None
    assert match_marker("   ", LINES, (1, 5), nearest_to=1, widen_by=3) is None
    assert match_marker(None, LINES, (1, 5), nearest_to=1, widen_by=3) is None


def test_ambiguous_matches_take_the_nearest_line() -> None:
    marker = "Gateau de Savoie"
    assert match_marker(marker, LINES, (1, 5), nearest_to=5, widen_by=0) == 4
    assert match_marker(marker, LINES, (1, 5), nearest_to=1, widen_by=0) == 2
    assert match_marker(marker, LINES, (1, 5), nearest_to=3, widen_by=0) == 2


def test_the_search_widens_once_by_the_context_radius() -> None:
    lines = [f"filler line {i}" for i in range(1, 21)]
    lines[14] = "Chapter the third"
    marker = "Chapter the third"
    assert match_marker(marker, lines, (1, 10), nearest_to=5, widen_by=3) is None
    assert match_marker(marker, lines, (1, 10), nearest_to=5, widen_by=5) == 15
