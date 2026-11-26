"""The ``_line_ranges.txt`` format and the tolerant text reader."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from chronominer.ranges.lines import (
    clamp_line_ranges,
    clamp_ranges,
    context_lines_refusal,
    format_ranges,
    parse_line_ranges,
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


def test_context_lines_belong_to_the_range_above(
    caplog: pytest.LogCaptureFixture,
) -> None:
    text = (
        "context (1, 2)\n"
        "(1, 10)\n"
        "  context (3, 4), (1, 1)\n"
        "CONTEXT(3, 4) (8, 8)\n"
        "(11, 20)\n"
        "(x, 5)\n"
        "  context (2, 2)\n"
        "(21, 30)\n"
        "  context (5, 1), (6, 7)\n"
        "  context 6-7\n"
    )
    with caplog.at_level(logging.WARNING):
        lines = parse_line_ranges(text)
    assert lines == [
        ((1, 10), ((1, 1), (3, 4), (8, 8))),
        ((11, 20), ()),
        ((21, 30), ((6, 7),)),
    ]
    assert len(caplog.records) == 5
    assert parse_ranges(text) == [(1, 10), (11, 20), (21, 30)]


def test_a_note_in_prose_is_skipped_as_before_and_is_no_context_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    text = "(1, 10)\nContext for the editor: check (3, 4)\n  context (2, 2)\n(11, 20)\n"
    with caplog.at_level(logging.WARNING):
        assert parse_line_ranges(text) == [((1, 10), ((2, 2),)), ((11, 20), ())]
    assert [record.getMessage() for record in caplog.records] == [
        "Invalid line range format in line ranges: Context for the editor: check (3, 4)"
    ]
    path = tmp_path / "doc_line_ranges.txt"
    path.write_text("(1, 10)\nContext for the editor\n(11, 20)\n", encoding="utf-8")
    assert context_lines_refusal(path) is None
    path.write_text("(1, 10)\n  context (0, 0), (2, 3)\n", encoding="utf-8")
    assert context_lines_refusal(path) is not None


def test_context_lines_round_trip_and_clamp() -> None:
    lines = [((1, 10), ((2, 3),)), ((11, 20), ()), ((21, 40), ((1, 1), (35, 50)))]
    text = format_ranges([b for b, _ in lines], [c for _, c in lines])
    assert text == (
        "(1, 10)\n  context (2, 3)\n(11, 20)\n(21, 40)\n  context (1, 1), (35, 50)\n"
    )
    assert parse_line_ranges(text) == lines
    assert clamp_line_ranges([*lines, ((45, 50), ((1, 1),))], 30) == [
        ((1, 10), ((2, 3),)),
        ((11, 20), ()),
        ((21, 30), ((1, 1),)),
    ]


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
