"""Marker matching: normalization, the length floor and the bounded search."""

from __future__ import annotations

import pytest

from chronominer.ranges.matching import match_marker, matching_lines, normalize
from chronominer.spec import DEFAULT_MARKER_LENGTH, MarkerLength, parse_marker_length

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
    short = "x" * (DEFAULT_MARKER_LENGTH.minimum - 1)
    assert match_marker(short, [short], (1, 1), nearest_to=1, widen_by=3) is None
    assert match_marker("   ", LINES, (1, 5), nearest_to=1, widen_by=3) is None
    assert match_marker(None, LINES, (1, 5), nearest_to=1, widen_by=3) is None


def test_the_minimum_marker_length_is_the_callers() -> None:
    lines = ["**PILAF.**", "Lavate il riso"]
    assert match_marker("PILAF.", lines, (1, 2), nearest_to=2, widen_by=0) == 1
    assert (
        match_marker("PILAF.", lines, (1, 2), nearest_to=2, widen_by=0, min_length=8)
        is None
    )


def test_marker_lengths_parse_and_print_as_min_max() -> None:
    assert parse_marker_length(" 8-20 ") == MarkerLength(8, 20)
    assert str(DEFAULT_MARKER_LENGTH) == "5-15"
    for bad in ("5", "0-5", "15-5", "a-b", "-5-15"):
        with pytest.raises(ValueError):
            parse_marker_length(bad)


def test_ambiguous_matches_take_the_nearest_line() -> None:
    marker = "Gateau de Savoie"
    assert match_marker(marker, LINES, (1, 5), nearest_to=5, widen_by=0) == 4
    assert match_marker(marker, LINES, (1, 5), nearest_to=1, widen_by=0) == 2
    assert match_marker(marker, LINES, (1, 5), nearest_to=3, widen_by=0) == 2


def test_a_line_the_marker_begins_wins_over_a_nearer_mention() -> None:
    lines = [
        "**Pomi di terra alla salsa bianca.**",
        "",
        "Lavate dei pomi di terra e fateli cuocere in una",
        "sale, coprite bene la casseruola e scuotete di quando",
        "in quando i pomi di terra affinchè possino cuocere",
    ]
    marker = "Pomi di terra"
    assert match_marker(marker, lines, (1, 5), nearest_to=5, widen_by=0) == 1
    assert match_marker(marker, lines[1:], (1, 4), nearest_to=4, widen_by=0) == 4


def test_a_dated_heading_counts_as_a_line_the_marker_begins() -> None:
    lines = [
        "1864. Host and Guest, a book about dinners.",
        "filler alpha",
        "filler beta",
        "Host and Guest returns in a later paragraph",
    ]
    marker = "Host and Guest"
    assert match_marker(marker, lines, (1, 4), nearest_to=1, widen_by=0) == 1


def test_a_date_marker_keeps_its_digits() -> None:
    lines = [
        "21.10.1896. Notice of the sale of a house",
        "filler alpha",
        "the sale announced on 21.10.1896 was postponed",
    ]
    assert match_marker("21.10.1896", lines, (1, 3), nearest_to=3, widen_by=0) == 1


def test_the_search_widens_once_by_the_context_radius() -> None:
    lines = [f"filler line {i}" for i in range(1, 21)]
    lines[14] = "Chapter the third"
    marker = "Chapter the third"
    assert match_marker(marker, lines, (1, 10), nearest_to=5, widen_by=3) is None
    assert match_marker(marker, lines, (1, 10), nearest_to=5, widen_by=5) == 15
