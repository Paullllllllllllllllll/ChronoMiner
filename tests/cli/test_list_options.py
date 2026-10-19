"""Repeated list options accumulate, deduplicated, in spec.parse.

The CLI parser's accumulation is covered by the rows of
``tests/core/test_argv_to_spec.py``; the option table's own parser, with
both the ``--flag value`` and the ``--flag=value`` spelling, is covered here.
"""

from __future__ import annotations

from typing import Any

import pytest

from chronominer.common.options import Kind, Option, OptionTable
from chronominer.spec import parse

BASE = ["--input", "in", "--schema", "price_list_entries"]

TABLE = OptionTable(
    [
        Option("model", "--model", "model"),
        Option("pages", "--pages", "pages", type=int),
        Option("files", "--files", "files", kind=Kind.LIST),
        Option(
            "formats", "--formats", "formats", kind=Kind.LIST, choices=("txt", "md")
        ),
        Option("force", "--force", "force", kind=Kind.FLAG, default=False),
        Option("resume", "--resume", "resume", kind=Kind.SWITCH, default=True),
    ]
)


@pytest.mark.parametrize(
    "argv",
    [
        ["--files", "a", "--files", "b,a", "--formats", "md", "--formats", "txt"],
        ["--files=a", "--files=b,a", "--formats=md", "--formats=txt"],
        ["--files", "a", "--files=b,a", "--formats=md", "--formats", "txt"],
    ],
)
def test_the_table_parser_accumulates_lists_in_both_spellings(
    argv: list[str],
) -> None:
    assert TABLE.parse(argv) == {"files": ("a", "b"), "formats": ("txt", "md")}


def test_the_equals_spelling_parses_like_the_spaced_one() -> None:
    spaced = ["--model", "m", "--pages", "3", "--files", "a,b", "--formats", "md"]
    joined = ["--model=m", "--pages=3", "--files=a,b", "--formats=md"]
    flags = ["--force", "--no-resume"]
    assert TABLE.parse(joined + flags) == TABLE.parse(spaced + flags)


@pytest.mark.parametrize(
    "values",
    [
        {"model": "-x"},
        {"model": "--foo"},
        {"pages": -3},
        {"files": ("-a", "b")},
        {"model": "m", "files": ("a",), "force": True, "resume": False},
    ],
)
def test_values_that_begin_with_a_dash_round_trip(values: dict[str, Any]) -> None:
    argv = TABLE.to_argv(values)
    assert TABLE.parse(argv) == values


def test_repeated_files_accumulate_through_spec_parse() -> None:
    argv = [*BASE, "--type", "pdf", "--files", "a", "--files", "b"]
    assert parse(argv).input.files == ("a", "b")
    argv += ["--files", "b,c"]
    assert parse(argv).input.files == ("a", "b", "c")


def test_an_invalid_repeated_value_is_still_a_usage_error() -> None:
    with pytest.raises(SystemExit) as info:
        parse([*BASE, "--formats", "csv", "--formats", "md"])
    assert info.value.code == 2
