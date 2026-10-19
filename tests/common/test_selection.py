"""Page-range grammar: parsing, resolution against a page count and description."""

from __future__ import annotations

import pytest

from chronominer.common.selection import PageRange, parse_page_range


class TestParsePageRange:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("text", "first_n"),
        [("5", 5), ("3", 3), ("first:10", 10), ("first : 3", 3)],
    )
    def test_first_n(self, text: str, first_n: int) -> None:
        pr = parse_page_range(text)
        assert pr.first_n == first_n
        assert pr.last_n is None
        assert pr.spans == ()

    @pytest.mark.unit
    @pytest.mark.parametrize(("text", "last_n"), [("last:7", 7), ("LAST:2", 2)])
    def test_last_n(self, text: str, last_n: int) -> None:
        pr = parse_page_range(text)
        assert pr.last_n == last_n
        assert pr.first_n is None

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("text", "spans"),
        [
            ("3-7", ((2, 6),)),
            ("3-", ((2, 2**31),)),
            ("-7", ((0, 6),)),
            ("1,3,5-8", ((0, 0), (2, 2), (4, 7))),
            ("1-5,3-8", ((0, 7),)),
            ("1-3,4-6", ((0, 5),)),
        ],
    )
    def test_spans_are_zero_indexed_and_merged(
        self, text: str, spans: tuple[tuple[int, int], ...]
    ) -> None:
        pr = parse_page_range(text)
        assert pr.first_n is None
        assert pr.last_n is None
        assert pr.spans == spans

    @pytest.mark.unit
    def test_raw_preserved(self) -> None:
        pr = parse_page_range("first:5")
        assert pr.raw == "first:5"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("", "cannot be empty"),
            ("0", "positive"),
            ("first:0", "positive"),
            ("last:0", "positive"),
            ("7-3", "must not exceed"),
            ("abc", "Invalid"),
            ("-", "both sides are empty"),
        ],
    )
    def test_invalid_text_raises(self, text: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_page_range(text)


class TestPageRangeResolve:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("text", "total", "expected"),
        [
            ("5", 20, [0, 1, 2, 3, 4]),
            ("50", 10, list(range(10))),
            ("last:3", 10, [7, 8, 9]),
            ("last:50", 5, [0, 1, 2, 3, 4]),
            ("3-7", 20, [2, 3, 4, 5, 6]),
            ("3-100", 10, [2, 3, 4, 5, 6, 7, 8, 9]),
            ("8-", 10, [7, 8, 9]),
            ("-3", 10, [0, 1, 2]),
            ("1,3,5-8", 20, [0, 2, 4, 5, 6, 7]),
            ("3-3", 10, [2]),
            ("5", 0, []),
            ("5", -1, []),
        ],
    )
    def test_resolve(self, text: str, total: int, expected: list[int]) -> None:
        assert parse_page_range(text).resolve(total) == expected


class TestPageRangeDescribe:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("first:5", "first 5 page(s)"),
            ("last:3", "last 3 page(s)"),
            ("3-7", "pages 3-7"),
            ("1,3,5-8", "pages 1,3,5-8"),
            ("3-3", "pages 3"),
            ("3-", "pages 3-"),
        ],
    )
    def test_describe(self, text: str, expected: str) -> None:
        assert parse_page_range(text).describe() == expected

    @pytest.mark.unit
    def test_default_describe_all_pages(self) -> None:
        assert PageRange().describe() == "all pages"
