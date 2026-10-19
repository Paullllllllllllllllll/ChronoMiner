"""The README agrees with the option tables, the settings example and the views."""

from __future__ import annotations

import asyncio
import re
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from chronominer.common.options import Kind, Option, OptionTable
from chronominer.output.store import VIEWS, database_path, open_store
from chronominer.settings import EXAMPLE_SETTINGS_PATH
from chronominer.spec import (
    BATCH_CANCEL_OPTIONS,
    BATCH_CHECK_OPTIONS,
    DROP_OPTIONS,
    OPTIONS,
    RANGES_ADJUST_OPTIONS,
    RANGES_GENERATE_OPTIONS,
    RENDER_OPTIONS,
    REPAIR_OPTIONS,
)
from tests.conftest import PROJECT_ROOT

README = PROJECT_ROOT / "README.md"
COMMANDS = "## Command Reference"
SETTINGS = "## Settings File"
STORE = "## Store"
FLAG_HEADER = ["Flag", "Description", "Default", "In `defaults:`"]
VIEW_HEADER = ["View", "One row per", "Columns"]
SETTABLE_TEXT = {True: "yes", False: "no"}

TABLES: dict[str, OptionTable] = {
    "extract": OPTIONS,
    "ranges generate": RANGES_GENERATE_OPTIONS,
    "ranges adjust": RANGES_ADJUST_OPTIONS,
    "batch check": BATCH_CHECK_OPTIONS,
    "batch cancel": BATCH_CANCEL_OPTIONS,
    "repair": REPAIR_OPTIONS,
    "render": RENDER_OPTIONS,
    "drop": DROP_OPTIONS,
    "ui": OPTIONS.subset(("settings",)),
}
SHARED_GROUPS = frozenset({"model", "images"})
"""Groups whose extract options these commands take; their sections may omit them."""
SHARING_COMMANDS = frozenset({"ranges adjust", "repair"})


def _section(text: str, heading: str) -> list[str]:
    lines = text.splitlines()
    start = lines.index(heading) + 1
    end = next(
        (i for i in range(start, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    return lines[start:end]


def _subsections(lines: list[str]) -> dict[str, list[str]]:
    """Split a section at its ``### `` headings; deeper headings stay inside."""
    parts: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in lines:
        if line.startswith("### "):
            current = parts.setdefault(line.removeprefix("### ").strip(), [])
        elif current is not None:
            current.append(line)
    return parts


def _cells(line: str) -> list[str]:
    inner = line.strip()[1:-1]
    return [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", inner)]


def _tables(lines: list[str]) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in [*lines, ""]:
        if line.startswith("|"):
            current.append(_cells(line))
        elif current:
            tables.append([current[0], *current[2:]])
            current = []
    return tables


def default_text(option: Option) -> str:
    """Return the default that ``--help`` prints for *option*, or ''."""
    if option.default_help is not None:
        return option.default_help
    if option.default is None or option.kind is Kind.FLAG:
        return ""
    if option.kind is Kind.SWITCH:
        return "on" if option.default else "off"
    if option.kind is Kind.LIST:
        return ",".join(str(item) for item in option.default)
    return str(option.default)


def description(option: Option) -> str:
    """Return the ``--help`` line of *option* without its default."""
    text = option.help_text()
    default = default_text(option)
    return text.replace(f"; default: {default}", "") if default else text


def _argument(option: Option) -> str:
    if option.kind in (Kind.SWITCH, Kind.FLAG):
        return ""
    if option.metavar is not None:
        return option.metavar
    return "{" + ",".join(str(choice) for choice in option.choices or ()) + "}"


def _flag_spellings(option: Option) -> list[str]:
    argument = _argument(option)
    if option.kind is Kind.SWITCH:
        return [option.flag, option.negative_flag]
    return [f"{option.flag} {argument}" if argument else option.flag]


def _flags(option: Option) -> list[str]:
    if option.kind is Kind.SWITCH:
        return [option.flag, option.negative_flag]
    return [option.flag]


def _check_row(
    command: str,
    row: list[str],
    table: OptionTable,
    seen: Counter[str],
    problems: list[str],
) -> None:
    spans = re.findall(r"`([^`]*)`", row[0])
    flags = [span.split(" ")[0] for span in spans]
    seen.update(flags)
    by_flag = {option.flag: option for option in table}
    if not flags or flags[0] not in by_flag:
        problems.append(f"{command}: unknown flag row: {row[0]}")
        return
    option = by_flag[flags[0]]
    where = f"{command}: {option.flag}"
    spellings = _flag_spellings(option)
    if spans != spellings:
        problems.append(f"{where}: flag cell {spans} != {spellings}")
    if row[1] != description(option):
        problems.append(f"{where}: description {row[1]!r}")
    default = default_text(option)
    if row[2] != default:
        problems.append(f"{where}: default {row[2]!r} != {default!r}")
    if row[3] != SETTABLE_TEXT[option.settable]:
        problems.append(f"{where}: defaults column {row[3]!r}")


def _optional(command: str, option: Option) -> bool:
    """Whether the section of *command* may leave *option* to extract's."""
    return (
        command in SHARING_COMMANDS
        and option.group in SHARED_GROUPS
        and option.name in OPTIONS
        and option == OPTIONS[option.name]
    )


def _command_drift(command: str, lines: list[str], table: OptionTable) -> list[str]:
    problems: list[str] = []
    seen: Counter[str] = Counter()
    for header, *rows in _tables(lines):
        if header == FLAG_HEADER:
            for row in rows:
                _check_row(command, row, table, seen, problems)
    expected = {flag: option for option in table for flag in _flags(option)}
    for flag, option in expected.items():
        allowed = {0, 1} if _optional(command, option) else {1}
        if seen[flag] not in allowed:
            problems.append(f"{command}: {flag}: {seen[flag]} README rows")
    problems.extend(
        f"{command}: {flag}: not an option" for flag in seen if flag not in expected
    )
    return problems


def table_drift(text: str) -> list[str]:
    """Return every disagreement between the Command Reference and the tables."""
    sections = _subsections(_section(text, COMMANDS))
    problems = [f"{name}: no section" for name in TABLES if name not in sections]
    for name, table in TABLES.items():
        problems.extend(_command_drift(name, sections.get(name, []), table))
    return problems


def readme_views(text: str) -> dict[str, list[str]]:
    """Return the columns the README lists for each view."""
    views: dict[str, list[str]] = {}
    for header, *rows in _tables(_section(text, STORE)):
        if header != VIEW_HEADER:
            continue
        for row in rows:
            name = row[0].strip("`")
            views[name] = re.findall(r"`([^`]*)`", row[2])
    return views


def store_views(root: Path) -> dict[str, list[str]]:
    """Return the columns of each documented view of a new store in *root*.

    No entries are written, so ``entries`` has its fixed columns only.
    """

    async def create() -> None:
        async with open_store(root) as store:
            await store.begin_run("extract")
            await store.end_run(0)

    asyncio.run(create())
    connection = sqlite3.connect(database_path(root))
    try:
        return {
            view: [
                str(row[1]) for row in connection.execute(f"PRAGMA table_info({view})")
            ]
            for view in VIEWS
        }
    finally:
        connection.close()


def settings_block(text: str) -> str:
    """Return the first YAML block of the Settings File section."""
    lines = _section(text, SETTINGS)
    start = lines.index("```yaml") + 1
    end = lines.index("```", start)
    return "\n".join(lines[start:end]) + "\n"


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_tables_match_the_option_tables() -> None:
    assert table_drift(_readme()) == []


def test_default_text_is_what_help_prints() -> None:
    for table in TABLES.values():
        for option in table:
            default = default_text(option)
            if default:
                assert f"default: {default}" in option.help_text(), option.flag
            else:
                assert "default: " not in option.help_text(), option.flag


def test_readme_settings_block_is_the_example_file() -> None:
    example = EXAMPLE_SETTINGS_PATH.read_text(encoding="utf-8")

    assert settings_block(_readme()) == example


def test_readme_view_columns_match_the_store(tmp_path: Path) -> None:
    assert readme_views(_readme()) == store_views(tmp_path)


DRIFT = [
    pytest.param(
        "| `--layout {flat,mirror}` | output folder layout: flat (every output in "
        "the output folder) or mirror (the input folder tree) | flat | yes |",
        "| `--layout {flat,mirror}` | output folder layout: flat (every output in "
        "the output folder) or mirror (the input folder tree) | mirror | yes |",
        "extract: --layout: default",
        id="changed-default",
    ),
    pytest.param(
        "| `--abandon SUBMISSION_ID` | give up an unresolved or submitted "
        "submission: its chunks return to pending and its uploaded files are "
        "deleted; a submitted batch is not cancelled | | no |\n",
        "",
        "batch check: --abandon: 0 README rows",
        id="missing-row",
    ),
    pytest.param(
        "| `--force` | replace files changed outside the tool | | no |",
        "| `--force` | replace files changed outside the tool | | yes |",
        "render: --force: defaults column",
        id="flipped-defaults-column",
    ),
    pytest.param(
        "| `--batch`, `--no-batch` | submit the chunks",
        "| `--batch` | submit the chunks",
        "extract: --no-batch: 0 README rows",
        id="missing-no-form",
    ),
    pytest.param(
        "| `--all` | cancel every unfinished batch of every provider | | no |",
        "| `--all` | cancel every unfinished batch of every provider | | no |\n"
        "| `--bogus` | x | | no |",
        "batch cancel: unknown flag row",
        id="unknown-flag",
    ),
    pytest.param(
        "| `--all` | cancel every unfinished batch of every provider | | no |",
        "| `--all` | cancel all unfinished batches of every provider | | no |",
        "batch cancel: --all: description",
        id="changed-description",
    ),
    pytest.param(
        "| `--grayscale`, `--no-grayscale` |",
        "| `--grayscale`, `--no-grayscale`, `--jpeg-quality N` |",
        "extract: --jpeg-quality: 2 README rows",
        id="duplicate-flag",
    ),
]


@pytest.mark.parametrize(("old", "new", "message"), DRIFT)
def test_table_drift_is_reported(old: str, new: str, message: str) -> None:
    text = _readme()
    assert text.count(old) == 1
    problems = table_drift(text.replace(old, new))
    assert any(message in problem for problem in problems), problems


def test_a_dropped_view_column_is_reported(tmp_path: Path) -> None:
    text = _readme().replace("`output_text`, ", "", 1)

    assert readme_views(text) != store_views(tmp_path)
