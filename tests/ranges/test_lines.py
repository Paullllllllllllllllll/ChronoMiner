"""The ``_line_ranges.txt`` format and the tolerant text reader."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from chronominer.ranges.lines import (
    clamp_ranges,
    format_ranges,
    parse_ranges,
    ranges_path,
    read_lines,
    read_ranges,
)

pytestmark = pytest.mark.unit


def test_ranges_path_sits_beside_the_text() -> None:
    assert ranges_path(Path("docs") / "ledger.txt") == Path(
        "docs/ledger_line_ranges.txt"
    )
    assert ranges_path(Path("notes.md")).name == "notes_line_ranges.txt"


def test_parse_accepts_both_forms_and_skips_bad_lines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    text = "(1, 10)\n11, 20\n\nnot a range\n(21, 22, 23)\n(x, 5)\n(0, 4)\n(9, 3)\n"
    with caplog.at_level(logging.WARNING):
        assert parse_ranges(text) == [(1, 10), (11, 20)]
    assert len(caplog.records) == 5


def test_read_ignores_a_byte_order_mark(tmp_path: Path) -> None:
    path = tmp_path / "doc_line_ranges.txt"
    path.write_bytes("﻿(1, 2)\n(3, 4)\n".encode())
    assert read_ranges(path) == [(1, 2), (3, 4)]


def test_format_round_trips() -> None:
    ranges = [(1, 10), (11, 20)]
    assert format_ranges(ranges) == "(1, 10)\n(11, 20)\n"
    assert parse_ranges(format_ranges(ranges)) == ranges
    assert format_ranges([]) == ""


def test_clamp_ranges_clamps_and_drops() -> None:
    assert clamp_ranges([(0, 5), (3, 9999), (8, 4)], 10) == [(1, 5), (3, 10)]
    assert clamp_ranges([(1, 4), (5, 10)], 10) == [(1, 4), (5, 10)]
    assert clamp_ranges([(20, 30)], 10) == []


def test_read_lines_strips_line_ends_only(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_bytes(b"  indented\r\nsecond\n\nlast")
    assert read_lines(path) == ["  indented", "second", "", "last"]
    path.write_text("a\nb\n", encoding="utf-8")
    assert read_lines(path) == ["a", "b"]
    path.write_text("", encoding="utf-8")
    assert read_lines(path) == []


def test_read_lines_falls_back_to_the_detected_encoding(tmp_path: Path) -> None:
    path = tmp_path / "latin.txt"
    path.write_bytes(("Küchen Meisterey\n" * 20).encode("cp1252"))
    lines = read_lines(path)
    assert len(lines) == 20
    assert lines[0].startswith("K")
