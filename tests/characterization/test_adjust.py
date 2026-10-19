"""Characterization of line-range adjustment (``ranges adjust``).

Every case writes a ledger of twelve records (headings on lines 1, 7, 13 ...
67 of 71) and a hand-made ``_line_ranges.txt``, then adjusts it through
``run_tool`` with boundary answers from ``FakeLLM``. By default the fake keeps
a range start that sits on a record heading and otherwise moves it to the
next heading; cases script other answers per ``adjust_unit``. The prompt is
``resources/prompts/semantic_boundary_prompt.txt`` and the adjust context
comes from ``tests/characterization/context``. The decisions are ``range``
units of the ``chronominer.sqlite`` beside the ledger, each with its decision
in the unit's provenance.

The case runs twice in fresh roots; both snapshots must match before each
stage is compared with ``golden/ranges/adjust/<case>/<stage>/``. Besides the
standard snapshot (which holds the database dump when the stage created it),
each stage records under ``outputs/`` the current rewritten
``_line_ranges.txt``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.testing import WriteGuard
from tests.characterization.adapter import DATABASE_NAME, RunResult, run_tool
from tests.characterization.conftest import CaseRoot
from tests.characterization.fakes import (
    FakeLLM,
    Reply,
    adjust_unit,
    permanent_error,
    transient_error,
)
from tests.characterization.inputs import (
    heading_lines,
    make_text_file,
    ranges_path,
    read_ranges,
    write_ranges,
)
from tests.characterization.stages import RunCase, Stage, case_runner, stage
from tests.characterization.test_batch import store_rows

pytestmark = pytest.mark.integration

CASE_PREFIX = "ranges/adjust"
RECORDS = 12

ON_HEADINGS = [(1, 18), (19, 36), (37, 54), (55, 71)]
"""Every start on a record heading; the default answers keep them all."""

MID_RECORD = [(1, 20), (21, 38), (39, 56), (57, 71)]
"""Starts 21, 39 and 57 lie inside records; the next headings are 25, 43, 61."""

SPLIT_FIRST = [(1, 4), (5, 20), (21, 38), (39, 56), (57, 71)]
"""The second range starts inside the first record."""


@pytest.fixture
def run_case(case_root: CaseRoot, write_guard: WriteGuard) -> RunCase:
    """Return ``run(case, build)``: run twice, compare, check every stage."""
    return case_runner(CASE_PREFIX, case_root, write_guard)


def ledger(root: Path) -> Path:
    """Write the case ledger under ``<root>/docs``."""
    return make_text_file(root / "docs" / "ledger.txt", RECORDS)


def database(text_file: Path) -> Path:
    """Return the store beside *text_file*."""
    return text_file.with_name(DATABASE_NAME)


def decisions(text_file: Path) -> list[dict[str, Any]]:
    """Return the stored decision of every decided range, in range order."""
    rows = store_rows(
        database(text_file),
        "SELECT provenance FROM units WHERE kind = 'range' ORDER BY idx",
    )
    stored = [json.loads(provenance or "{}") for (provenance,) in rows]
    return [entry["decision"] for entry in stored if "decision" in entry]


def boundary(
    *,
    marker: str = "",
    on_target: bool = False,
    no_content: bool = False,
    more_context: bool = False,
    certainty: int = 95,
) -> dict[str, Any]:
    """Return a boundary answer in the shape of the semantic-boundary schema."""
    return {
        "contains_no_semantic_boundary": no_content,
        "needs_more_context": more_context,
        "boundary_already_on_target": on_target,
        "certainty": certainty,
        "semantic_marker": marker,
    }


def adjust(fake_llm: FakeLLM, root: Path, text_file: Path, **keywords: Any) -> Stage:
    """Run ``adjust_line_ranges`` on *text_file*; record its working files."""
    result = run_tool(
        "adjust_line_ranges", llm=fake_llm, root=root, input=text_file, **keywords
    )
    return stage(result, outputs=[ranges_path(text_file)])


def units(result: RunResult) -> list[str | None]:
    """Return the unit of every boundary call of a run, in sorted order."""
    return [call["unit"] for call in result.calls]


def _ok(result: RunResult) -> None:
    assert result.exit_code == 0, result.console


def _counts(result: RunResult) -> tuple[int, int]:
    """Return the complete and failed files of a run's summary."""
    assert result.summary is not None, result.console
    return result.summary["items_complete"], result.summary["items_failed"]


def test_boundaries_on_headings_are_kept(fake_llm: FakeLLM, run_case: RunCase) -> None:
    def build(root: Path) -> dict[str, Stage]:
        text = ledger(root)
        write_ranges(text, ON_HEADINGS)
        return {"adjust": adjust(fake_llm, root, text)}

    stages = run_case("kept", build)

    result = stages["adjust"].result
    _ok(result)
    text = result.root / "docs" / "ledger.txt"
    assert read_ranges(ranges_path(text)) == ON_HEADINGS
    assert database(text) in result.created
    assert units(result) == [adjust_unit(index) for index in (0, 3, 6, 9)]
    assert {call["role"] for call in result.calls} == {"adjust"}
    bodies = decisions(text)
    assert len(bodies) == len(ON_HEADINGS)
    assert all(body["decision"]["boundary_already_on_target"] for body in bodies)
    assert _counts(result) == (1, 0)


def test_boundaries_move_to_the_next_heading_in_the_window(
    fake_llm: FakeLLM, run_case: RunCase
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        text = ledger(root)
        write_ranges(text, MID_RECORD)
        return {"adjust": adjust(fake_llm, root, text)}

    stages = run_case("moved", build)

    result = stages["adjust"].result
    _ok(result)
    text = result.root / "docs" / "ledger.txt"
    adjusted = read_ranges(ranges_path(text))
    assert adjusted == [(1, 24), (25, 42), (43, 60), (61, 71)]
    assert {start for start, _ in adjusted} <= set(heading_lines(text))
    assert units(result) == [adjust_unit(index) for index in (0, 4, 7, 10)]


def test_a_range_moved_onto_its_predecessor_is_merged(
    fake_llm: FakeLLM, run_case: RunCase
) -> None:
    fake_llm.script(adjust_unit(1), payload=boundary(marker="Record 001."))

    def build(root: Path) -> dict[str, Stage]:
        text = ledger(root)
        write_ranges(text, SPLIT_FIRST)
        return {"adjust": adjust(fake_llm, root, text)}

    stages = run_case("merged", build)

    result = stages["adjust"].result
    _ok(result)
    text = result.root / "docs" / "ledger.txt"
    assert read_ranges(ranges_path(text)) == [(1, 24), (25, 42), (43, 60), (61, 71)]
    bodies = decisions(text)
    assert bodies[1]["adjusted_range"] == [1, 20]


def test_a_range_without_content_is_deleted(
    fake_llm: FakeLLM, run_case: RunCase
) -> None:
    fake_llm.script(adjust_unit(6), payload=boundary(no_content=True))

    def build(root: Path) -> dict[str, Stage]:
        text = ledger(root)
        write_ranges(text, ON_HEADINGS)
        return {"adjust": adjust(fake_llm, root, text)}

    stages = run_case("no_content", build)

    result = stages["adjust"].result
    _ok(result)
    text = result.root / "docs" / "ledger.txt"
    assert read_ranges(ranges_path(text)) == [(1, 18), (19, 36), (55, 71)]
    assert units(result).count(adjust_unit(6)) == 2
    bodies = decisions(text)
    assert [body["should_delete"] for body in bodies] == [False, False, True, False]


def test_a_transient_model_error_is_retried(
    fake_llm: FakeLLM, run_case: RunCase
) -> None:
    fake_llm.reply(adjust_unit(4), 1, Reply(transient_error()))

    def build(root: Path) -> dict[str, Stage]:
        text = ledger(root)
        write_ranges(text, MID_RECORD)
        return {"adjust": adjust(fake_llm, root, text)}

    stages = run_case("retry", build)

    result = stages["adjust"].result
    _ok(result)
    text = result.root / "docs" / "ledger.txt"
    assert read_ranges(ranges_path(text)) == [(1, 24), (25, 42), (43, 60), (61, 71)]
    retried = [call for call in result.calls if call["unit"] == adjust_unit(4)]
    assert [call["attempt"] for call in retried] == [1, 2]


def test_a_permanent_model_error_fails_the_file(
    fake_llm: FakeLLM, run_case: RunCase
) -> None:
    fake_llm.script(adjust_unit(4), always=Reply(permanent_error()))

    def build(root: Path) -> dict[str, Stage]:
        text = ledger(root)
        write_ranges(text, MID_RECORD)
        return {"adjust": adjust(fake_llm, root, text)}

    stages = run_case("error", build)

    result = stages["adjust"].result
    assert result.exit_code == 1
    text = result.root / "docs" / "ledger.txt"
    assert read_ranges(ranges_path(text)) == MID_RECORD
    assert _counts(result) == (0, 1)


def test_resume_continues_a_partial_adjustment_and_force_redoes_it(
    fake_llm: FakeLLM, run_case: RunCase
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        text = ledger(root)
        write_ranges(text, MID_RECORD)
        stages = {"partial": adjust(fake_llm, root, text, first_n_chunks=2)}
        stages["resume"] = adjust(fake_llm, root, text, resume=True)
        stages["complete"] = adjust(fake_llm, root, text, resume=True)
        stages["force"] = adjust(fake_llm, root, text, resume=True, force=True)
        return stages

    stages = run_case("resume", build)

    partial = stages["partial"].result
    resumed = stages["resume"].result
    complete = stages["complete"].result
    forced = stages["force"].result
    for result in (partial, resumed, complete, forced):
        _ok(result)
    text = partial.root / "docs" / "ledger.txt"
    assert units(partial) == [adjust_unit(0), adjust_unit(4)]
    assert units(resumed) == [adjust_unit(7), adjust_unit(10)]
    assert complete.calls == []
    assert units(forced) == [adjust_unit(index) for index in (0, 4, 7, 10)]
    assert read_ranges(ranges_path(text)) == [(1, 24), (25, 42), (43, 60), (61, 71)]
    assert len(decisions(text)) == len(MID_RECORD)
