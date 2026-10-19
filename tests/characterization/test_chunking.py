"""Characterization of the chunking modes of the extract command.

``auto`` with a small chunk size, ``line_ranges`` over an existing
``{stem}_line_ranges.txt`` and ``adjust-line-ranges``, which adjusts the
ranges file through boundary requests before extracting. Each case runs
twice in fresh roots; both snapshots must match before the run is compared
with ``golden/chunking/<case>/``. Besides the standard snapshot, a case may
record ``outputs/``: the content after the run of input-side files the case
wrote before it, such as a ranges file the run rewrites.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.testing import WriteGuard
from tests.characterization.adapter import RunResult, run_tool
from tests.characterization.conftest import CaseRoot
from tests.characterization.fakes import (
    ADJUST,
    EXTRACT,
    FakeLLM,
    Reply,
    adjust_unit,
    record_unit,
    transient_error,
)
from tests.characterization.inputs import (
    heading_lines,
    make_text_file,
    ranges_path,
    read_ranges,
    write_ranges,
)
from tests.characterization.stages import Stage, single_case_runner, stage

pytestmark = pytest.mark.integration

CASE_PREFIX = "chunking"
RECORDS = 12
ALIGNED = ((1, 23), (25, 47), (49, 71))
"""Ranges of the 12-record ledger that start on records 0, 4 and 8."""
MISALIGNED = ((1, 20), (21, 44), (45, 71))
"""Ranges whose second and third starts fall inside records 3 and 7."""

Build = Callable[[Path], tuple[RunResult, list[Path]]]


@pytest.fixture
def chunking_case(
    fake_llm: FakeLLM, case_root: CaseRoot, write_guard: WriteGuard
) -> Callable[[str, Build], RunResult]:
    """Return ``run(case, build)``: run twice, compare, check the golden files.

    *build* returns the run and the pre-existing files whose content after
    the run is recorded under ``outputs/``.
    """
    runner = single_case_runner(
        CASE_PREFIX, case_root, write_guard, before_each=fake_llm.units.clear
    )

    def run(case: str, build: Build) -> RunResult:
        def staged(root: Path) -> Stage:
            result, outputs = build(root)
            return stage(result, outputs=outputs)

        return runner(case, staged).result

    return run


def _ledger(root: Path) -> Path:
    return make_text_file(root / "docs" / "ledger.txt", RECORDS)


def _calls(result: RunResult, role: str) -> list[dict[str, Any]]:
    return [call for call in result.calls if call["role"] == role]


def _chunk_ranges(result: RunResult, ledger: Path) -> list[list[int]]:
    data = json.loads(result.output_json(ledger).read_text(encoding="utf-8"))
    return [record["chunk_range"] for record in data["records"]]


def test_auto_with_a_small_chunk_size(
    fake_llm: FakeLLM, chunking_case: Callable[[str, Build], RunResult]
) -> None:
    def build(root: Path) -> tuple[RunResult, list[Path]]:
        ledger = _ledger(root)
        result = run_tool(
            "extract",
            llm=fake_llm,
            root=root,
            input=ledger,
            output="out",
            chunking="auto",
            chunk_size=60,
        )
        return result, []

    result = chunking_case("auto_small_chunk_size", build)

    assert result.exit_code == 0, result.stdout + result.stderr
    ledger = result.root / "docs" / "ledger.txt"
    ranges = _chunk_ranges(result, ledger)
    assert len(ranges) > 4
    assert ranges[0][0] == 1
    assert ranges[-1][1] == len(ledger.read_text(encoding="utf-8").splitlines())


def test_line_ranges_from_an_existing_file(
    fake_llm: FakeLLM, chunking_case: Callable[[str, Build], RunResult]
) -> None:
    def build(root: Path) -> tuple[RunResult, list[Path]]:
        ledger = _ledger(root)
        ranges = write_ranges(ledger, ALIGNED)
        result = run_tool(
            "extract",
            llm=fake_llm,
            root=root,
            input=ledger,
            output="out",
            chunking="line_ranges",
        )
        return result, [ranges]

    result = chunking_case("line_ranges", build)

    assert result.exit_code == 0, result.stdout + result.stderr
    ledger = result.root / "docs" / "ledger.txt"
    assert _chunk_ranges(result, ledger) == [list(pair) for pair in ALIGNED]
    assert [call["unit"] for call in result.calls] == [
        record_unit(0),
        record_unit(4),
        record_unit(8),
    ]


def test_adjust_line_ranges_then_extract(
    fake_llm: FakeLLM, chunking_case: Callable[[str, Build], RunResult]
) -> None:
    def build(root: Path) -> tuple[RunResult, list[Path]]:
        ledger = _ledger(root)
        ranges = write_ranges(ledger, MISALIGNED)
        fake_llm.reply(adjust_unit(4), 1, Reply(transient_error()))
        result = run_tool(
            "extract",
            llm=fake_llm,
            root=root,
            input=ledger,
            output="out",
            chunking="adjust-line-ranges",
        )
        return result, [ranges]

    result = chunking_case("adjust_line_ranges", build)

    assert result.exit_code == 0, result.stdout + result.stderr
    ledger = result.root / "docs" / "ledger.txt"
    adjusted = read_ranges(ranges_path(ledger))
    starts = {start for start, _ in adjusted}
    assert starts - {1} <= set(heading_lines(ledger))
    assert [list(pair) for pair in adjusted] == _chunk_ranges(result, ledger)
    adjust_calls = _calls(result, ADJUST)
    assert adjust_calls
    assert any(call["attempt"] == 2 for call in adjust_calls)
    assert _calls(result, EXTRACT)
