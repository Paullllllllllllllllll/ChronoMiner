"""Plan-time checks and runs of ``ranges generate`` and ``ranges adjust``.

A scripted chat model (``tests.llm.helpers``) answers every boundary
request, so no request leaves the machine.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

import chronominer.run as core
from chronominer.common.contract import ExitCode
from chronominer.common.sqlite import read_rows
from chronominer.output.store import DATABASE_NAME
from chronominer.spec import (
    RangesAdjustSpec,
    RangesGenerateSpec,
    resolve_ranges_adjust,
    resolve_ranges_generate,
)
from tests.conftest import DEFAULT_TEST_SCHEMA, make_settings
from tests.llm import helpers

ON_TARGET: dict[str, Any] = {
    "contains_no_semantic_boundary": False,
    "needs_more_context": False,
    "boundary_already_on_target": True,
    "certainty": 95,
    "semantic_marker": "",
}


def _text(path: Path, lines: int = 40) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        f"Entry {n}: John Carter, mason, 2 shillings" for n in range(lines)
    )
    path.write_text(body + "\n", encoding="utf-8")
    return path


def _ranges(text: Path, *bounds: tuple[int, int]) -> Path:
    path = text.with_name(f"{text.stem}_line_ranges.txt")
    path.write_text(
        "".join(f"({start}, {end})\n" for start, end in bounds), encoding="utf-8"
    )
    return path


def _generate(input_path: Path, **flags: Any) -> RangesGenerateSpec:
    return resolve_ranges_generate({"input": input_path, **flags}).spec


def _adjust(input_path: Path, **flags: Any) -> RangesAdjustSpec:
    flags.setdefault("schema", DEFAULT_TEST_SCHEMA)
    return resolve_ranges_adjust({"input": input_path, **flags}).spec


def _files(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.rglob("*"))


@pytest.fixture
def openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variable of the default provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_generate_plans_every_text_file_without_a_key(tmp_path: Path) -> None:
    _text(tmp_path / "in" / "a.txt")
    _text(tmp_path / "in" / "sub" / "b.md")
    (tmp_path / "in" / "scan.pdf").write_bytes(b"%PDF")

    flat = core.plan_ranges(_generate(tmp_path / "in"), make_settings())
    deep = core.plan_ranges(_generate(tmp_path / "in", recursive=True), make_settings())

    assert [path.name for path in flat.files] == ["a.txt"]
    assert sorted(path.name for path in deep.files) == ["a.txt", "b.md"]
    assert (flat.command, flat.adjust) == ("ranges generate", False)


def _missing_input(tmp_path: Path) -> Path:
    return tmp_path / "missing.txt"


def _empty_folder(tmp_path: Path) -> Path:
    (tmp_path / "in").mkdir()
    return tmp_path / "in"


def _pdf_file(tmp_path: Path) -> Path:
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF")
    return path


@pytest.mark.parametrize(
    ("setup", "error", "message"),
    [
        (_missing_input, core.InputError, "Input path does not exist"),
        (_empty_folder, core.NoItemsError, "No text files found"),
        (_pdf_file, core.InputError, "does not fit scan.pdf"),
    ],
)
def test_input_errors_of_both_commands_exit_2(
    tmp_path: Path,
    openai_key: None,
    setup: Any,
    error: type[core.PlanError],
    message: str,
) -> None:
    input_path = setup(tmp_path)

    for spec in (_generate(input_path), _adjust(input_path)):
        with pytest.raises(error, match=message) as raised:
            core.plan_ranges(spec, make_settings())
        assert raised.value.exit_code == ExitCode.USAGE


def test_adjust_needs_the_key_of_its_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = _text(tmp_path / "in" / "a.txt")
    _ranges(text, (1, 20), (21, 40))
    spec = _adjust(text, model="claude-sonnet-5-5")

    with pytest.raises(core.ConfigurationError, match="ANTHROPIC_API_KEY") as raised:
        core.plan_ranges(spec, make_settings())

    assert raised.value.items_total == 1
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert core.plan_ranges(spec, make_settings()).files == (text.resolve(),)


@pytest.mark.usefixtures("openai_key")
def test_adjust_refuses_an_unknown_schema(tmp_path: Path) -> None:
    text = _text(tmp_path / "a.txt")
    _ranges(text, (1, 40))

    with pytest.raises(core.ConfigurationError, match="unknown schema 'NoSuch'"):
        core.plan_ranges(_adjust(text, schema="NoSuch"), make_settings())


@pytest.mark.usefixtures("openai_key")
def test_adjust_resolves_the_schema_by_its_name_field(tmp_path: Path) -> None:
    text = _text(tmp_path / "a.txt")
    _ranges(text, (1, 40))

    by_name = core.plan_ranges(_adjust(text), make_settings())
    by_file = core.plan_ranges(_adjust(text, schema="wage_records"), make_settings())

    assert by_name.schema is not None and by_file.schema is not None
    assert by_name.schema.name == by_file.schema.name == DEFAULT_TEST_SCHEMA


@pytest.mark.usefixtures("openai_key")
def test_adjust_refuses_a_missing_context_file(tmp_path: Path) -> None:
    text = _text(tmp_path / "a.txt")
    _ranges(text, (1, 40))
    spec = _adjust(text, context=tmp_path / "nope.txt")

    with pytest.raises(core.ConfigurationError, match="nope.txt"):
        core.plan_ranges(spec, make_settings())


@pytest.mark.usefixtures("openai_key")
def test_adjust_takes_each_file_s_sidecar_context(tmp_path: Path) -> None:
    a = _text(tmp_path / "in" / "a.txt")
    b = _text(tmp_path / "in" / "b.txt")
    for text in (a, b):
        _ranges(text, (1, 40))
    (tmp_path / "in" / "a_adjust_context.txt").write_text(
        "Payroll of a mason's lodge.", encoding="utf-8"
    )

    ranges_plan = core.plan_ranges(_adjust(tmp_path / "in"), make_settings())

    contexts = {path.name: text for path, text in ranges_plan.contexts.items()}
    assert contexts["a.txt"] == "Payroll of a mason's lodge."
    assert contexts["b.txt"] != contexts["a.txt"]


@pytest.mark.usefixtures("openai_key")
def test_adjust_leaves_out_files_without_line_ranges(tmp_path: Path) -> None:
    ready = _text(tmp_path / "in" / "a.txt")
    _ranges(ready, (1, 40))
    _text(tmp_path / "in" / "b.txt")

    ranges_plan = core.plan_ranges(_adjust(tmp_path / "in"), make_settings())

    assert [path.name for path in ranges_plan.files] == ["a.txt"]
    (warning,) = ranges_plan.warnings
    assert "b_line_ranges.txt" in warning and "ranges generate" in warning


@pytest.mark.usefixtures("openai_key")
def test_adjust_without_any_line_ranges_file_exits_2(tmp_path: Path) -> None:
    _text(tmp_path / "in" / "a.txt")

    with pytest.raises(core.NoItemsError, match="ranges generate") as raised:
        core.plan_ranges(_adjust(tmp_path / "in"), make_settings())

    assert raised.value.items_total == 1


@pytest.mark.usefixtures("openai_key")
def test_planning_writes_nothing(tmp_path: Path) -> None:
    text = _text(tmp_path / "in" / "a.txt")
    _ranges(text, (1, 40))
    before = _files(tmp_path)

    core.plan_ranges(_generate(tmp_path / "in"), make_settings())
    core.plan_ranges(_adjust(tmp_path / "in"), make_settings())

    assert _files(tmp_path) == before


def test_a_dry_run_plan_is_not_run(tmp_path: Path) -> None:
    text = _text(tmp_path / "a.txt")
    ranges_plan = core.plan_ranges(_generate(text, dry_run=True), make_settings())

    with pytest.raises(ValueError, match="dry-run"):
        asyncio.run(core.run_ranges(ranges_plan))


def test_generate_writes_the_ranges_file_and_the_database(tmp_path: Path) -> None:
    text = _text(tmp_path / "in" / "a.txt", lines=200)
    ranges_plan = core.plan_ranges(_generate(text, chunk_tokens=500), make_settings())

    result = asyncio.run(core.run_ranges(ranges_plan))

    ranges_file = text.with_name("a_line_ranges.txt")
    assert [(item.name, item.success) for item in result.items] == [("a.txt", True)]
    assert result.items[0].outputs == (str(ranges_file.resolve()),)
    assert len(ranges_file.read_text(encoding="utf-8").splitlines()) > 1
    database = text.parent / DATABASE_NAME
    assert result.databases == (str(database.resolve()),)
    (run_row,) = read_rows(database, "runs")
    assert (run_row["command"], run_row["exit_code"]) == ("ranges generate", 0)


@pytest.mark.usefixtures("openai_key")
def test_adjust_decides_every_range_through_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = helpers.install(monkeypatch, [ON_TARGET])
    text = _text(tmp_path / "in" / "a.txt")
    ranges_file = _ranges(text, (1, 20), (21, 40))
    ranges_plan = core.plan_ranges(
        _adjust(text), make_settings(rate_limits=((100_000, 1),))
    )

    result = asyncio.run(core.run_ranges(ranges_plan))

    assert [(item.name, item.success) for item in result.items] == [("a.txt", True)]
    assert len(script.requests) == 2
    assert result.usage["adjust"].total_tokens == 240
    units = read_rows(text.parent / DATABASE_NAME, "units", order_by=["idx"])
    assert [(row["kind"], row["status"]) for row in units] == [("range", "done")] * 2
    assert ranges_file.read_text(encoding="utf-8").splitlines() == [
        "(1, 20)",
        "(21, 40)",
    ]
