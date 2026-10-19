"""Token-based line ranges, chunk texts and ``ranges generate``."""

from __future__ import annotations

from pathlib import Path

import pytest

from chronominer.common.selection import parse_page_range
from chronominer.output.store import DATABASE_NAME
from chronominer.ranges.generate import (
    auto_ranges,
    chunk_texts,
    generate_files,
    ranges_for_file,
    select_ranges,
    text_chunks,
    token_ranges,
)
from chronominer.ranges.lines import ranges_path, read_ranges
from tests.characterization.inputs import make_text_file

pytestmark = pytest.mark.unit

LINES = ["aa", "bbb", "c", "dd"]


def _assert_tiles(ranges: list[tuple[int, int]], last: int) -> None:
    assert ranges[0][0] == 1
    assert ranges[-1][1] == last
    for (_, end), (start, _) in zip(ranges, ranges[1:], strict=False):
        assert start == end + 1


def test_ranges_tile_the_lines_within_the_limit() -> None:
    ranges = token_ranges(LINES, 4, "x")
    _assert_tiles(ranges, len(LINES))
    assert len(ranges) >= 2
    assert token_ranges(LINES, 10_000, "x") == [(1, 4)]


def test_empty_input_has_no_phantom_range() -> None:
    assert token_ranges([], 100, "x") == []
    assert token_ranges(["single line"], 100, "x") == [(1, 1)]


def test_a_line_over_the_limit_forms_its_own_range() -> None:
    lines = ["short", "word " * 200, "short"]
    assert token_ranges(lines, 20, "gpt-4o") == [(1, 1), (2, 2), (3, 3)]


def test_unknown_models_and_special_tokens_count() -> None:
    lines = ["plain text", "a literal <|endoftext|> marker", "more"]
    assert token_ranges(lines, 10_000, "no-such-model") == [(1, 3)]
    _assert_tiles(token_ranges(lines, 6, "gpt-4o"), 3)


def test_auto_ranges_match_token_ranges() -> None:
    assert auto_ranges(LINES, 4, "x") == token_ranges(LINES, 4, "x")


def test_chunk_texts_join_lines_with_newlines(tmp_path: Path) -> None:
    lines = ["Zucker", "und", "Mehl", "Salz"]
    assert chunk_texts(lines, [(1, 2), (3, 4)]) == ["Zucker\nund", "Mehl\nSalz"]
    path = tmp_path / "doc.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert text_chunks(path, [(2, 3)]) == ["und\nMehl"]


def test_ranges_for_file_reads_non_utf8(tmp_path: Path) -> None:
    path = tmp_path / "latin1.txt"
    path.write_bytes(("Küchen Meisterey\n" * 20).encode("cp1252"))
    ranges = ranges_for_file(path, 10, "gpt-4o")
    _assert_tiles(ranges, 20)


def test_select_ranges_follows_the_page_selection() -> None:
    ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
    assert select_ranges(ranges, None) == ranges
    assert select_ranges(ranges, parse_page_range("first:2")) == ranges[:2]
    assert select_ranges(ranges, parse_page_range("last:2")) == ranges[-2:]
    assert select_ranges(ranges, parse_page_range("first:10")) == ranges
    assert select_ranges(ranges, parse_page_range("2-3")) == ranges[1:3]
    assert select_ranges(ranges, parse_page_range("7-9")) == []


async def test_generate_files_records_and_renders_the_ranges(tmp_path: Path) -> None:
    ledger = make_text_file(tmp_path / "docs" / "ledger.txt", 12)
    notes = make_text_file(tmp_path / "docs" / "sub" / "notes.md", 2)
    ranges_path(ledger).write_text("(1, 3)\n(4, 9)\n", encoding="utf-8")

    reports = await generate_files([ledger, notes], chunk_tokens=60, model="gpt-4o")

    assert [report.ok for report in reports] == [True, True]
    expected = ranges_for_file(ledger, 60, "gpt-4o")
    assert read_ranges(ranges_path(ledger)) == expected
    assert reports[0].total == reports[0].written == len(expected)
    _assert_tiles(read_ranges(ranges_path(notes)), 11)
    assert (ledger.parent / DATABASE_NAME).is_file()
    assert (notes.parent / DATABASE_NAME).is_file()


async def test_generate_files_writes_the_selected_ranges(tmp_path: Path) -> None:
    ledger = make_text_file(tmp_path / "ledger.txt", 12)
    every = ranges_for_file(ledger, 60, "gpt-4o")

    (report,) = await generate_files(
        [ledger], chunk_tokens=60, model="gpt-4o", pages=parse_page_range("last:2")
    )

    assert (report.total, report.written) == (len(every), 2)
    assert read_ranges(ranges_path(ledger)) == every[-2:]


async def test_generate_files_reports_a_failed_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    good = make_text_file(tmp_path / "good.txt", 2)

    reports = await generate_files([missing, good], chunk_tokens=60, model="gpt-4o")

    assert [report.ok for report in reports] == [False, True]
    assert reports[0].error
