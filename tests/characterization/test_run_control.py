"""Characterization of run control: resume, force, dry run, JSON line, exit codes.

Each case builds its inputs, runs one or more stages through ``run_tool`` in
the same root and snapshots every stage. The case runs twice in fresh roots;
both snapshots must match before each stage is compared with
``golden/run_control/<case>/<stage>/``. Besides the standard snapshot, a
stage may record ``outputs/`` (the current content of named output files,
for files an earlier stage created) and ``report.txt`` (selected console
lines, for messages the JSON line does not carry).
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.testing import WriteGuard
from tests.characterization.adapter import RunResult, run_tool
from tests.characterization.conftest import CaseRoot
from tests.characterization.fakes import (
    FakeLLM,
    Reply,
    page_unit,
    permanent_error,
    record_unit,
)
from tests.characterization.golden import normalizer_for
from tests.characterization.inputs import (
    make_image_folder,
    make_scanned_pdf,
    make_text_file,
)
from tests.characterization.stages import Build, RunCase, Stage, case_runner
from tests.characterization.stages import stage as snapshot_stage

pytestmark = pytest.mark.integration

CASE_PREFIX = "run_control"
RECORDS = 12
PAGES = 3
REVISED: dict[str, Any] = {
    "contains_no_content_of_requested_type": False,
    "entries": [
        {
            "record_id": "Record revised",
            "item": "revised item",
            "quantity": None,
            "price": None,
            "date": None,
            "source_page": None,
            "notes": "second run",
        }
    ],
}

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def stage(
    result: RunResult,
    *,
    outputs: Iterable[Path] = (),
    report: Sequence[str] = (),
) -> Stage:
    """Snapshot *result*; add the content of *outputs* and the *report* lines.

    *report* lists substrings; every console line (stdout and stderr) that
    contains one of them is recorded in ``report.txt``.
    """
    current = snapshot_stage(result, outputs=outputs)
    if report:
        current.files["report.txt"] = _report(result, report)
    return current


def _report(result: RunResult, needles: Sequence[str]) -> str:
    norm = normalizer_for(result)
    lines = [
        norm.text(_ANSI_RE.sub("", line).rstrip())
        for line in (result.stdout + "\n" + result.stderr).splitlines()
        if any(needle in line for needle in needles)
    ]
    return "".join(f"{line}\n" for line in lines)


@pytest.fixture
def run_case(
    fake_llm: FakeLLM, case_root: CaseRoot, write_guard: WriteGuard
) -> RunCase:
    """Return ``run(case, build)``: run twice, compare, check every stage."""
    return case_runner(
        CASE_PREFIX, case_root, write_guard, before_each=fake_llm.units.clear
    )


@dataclasses.dataclass(frozen=True)
class Source:
    """An input kind: how to build it, its units, and the unit that fails."""

    make: Callable[[Path], Path]
    units: tuple[str, ...]
    failing: str


SOURCES: dict[str, Source] = {
    "text": Source(
        make=lambda root: make_text_file(root / "docs" / "ledger.txt", RECORDS),
        units=tuple(record_unit(i) for i in (0, 3, 6, 9)),
        failing=record_unit(3),
    ),
    "image_folder": Source(
        make=lambda root: make_image_folder(root / "scans", PAGES),
        units=tuple(page_unit(i) for i in range(PAGES)),
        failing=page_unit(1),
    ),
    "pdf": Source(
        make=lambda root: make_scanned_pdf(root / "docs" / "book.pdf", PAGES),
        units=tuple(page_unit(i) for i in range(PAGES)),
        failing=page_unit(1),
    ),
}


def _extract(llm: FakeLLM, root: Path, *, input: Path, **flags: Any) -> RunResult:
    return run_tool("extract", llm=llm, root=root, input=input, output="out", **flags)


def _outputs(root: Path) -> list[Path]:
    """Return the ``_output.json`` files under the case's output folder."""
    return sorted((root / "out").glob("*_output.json"))


def _units(result: RunResult) -> list[str | None]:
    return [call["unit"] for call in result.calls]


def _ok(result: RunResult) -> None:
    assert result.exit_code == 0, result.stdout + result.stderr


@pytest.mark.parametrize("source", list(SOURCES))
@pytest.mark.parametrize(
    ("mode", "flags"),
    [("resume", {"resume": True}), ("force", {"resume": True, "force": True})],
)
def test_second_run_on_a_completed_input(
    source: str,
    mode: str,
    flags: dict[str, Any],
    fake_llm: FakeLLM,
    run_case: Callable[[str, Build], dict[str, Stage]],
) -> None:
    spec = SOURCES[source]

    def build(root: Path) -> dict[str, Stage]:
        item = spec.make(root)
        first = _extract(fake_llm, root, input=item)
        _ok(first)
        for unit in spec.units:
            fake_llm.script(unit, payload=REVISED)
        second = _extract(fake_llm, root, input=item, **flags)
        return {
            "first": stage(first),
            "second": stage(second, outputs=_outputs(root)),
        }

    second = run_case(f"{mode}_complete_{source}", build)["second"].result

    _ok(second)
    assert second.summary is not None
    documents = len(_outputs(second.root))
    if mode == "resume":
        assert second.calls == []
        assert second.created == []
        assert second.summary["items_skipped"] == documents
    else:
        assert sorted(_units(second)) == sorted(spec.units)
        assert second.summary["items_complete"] == documents
        for output in _outputs(second.root):
            assert "revised item" in output.read_text(encoding="utf-8")


def test_pages_beyond_the_last_chunk_skip_the_item_with_a_warning(
    fake_llm: FakeLLM, case_root: CaseRoot
) -> None:
    root = case_root()
    item = SOURCES["pdf"].make(root)

    for state in ("fresh", "complete"):
        result = _extract(fake_llm, root, input=item, page_range="9-")

        _ok(result)
        assert result.summary is not None
        counts = ("items_total", "items_complete", "items_skipped")
        assert [result.summary[key] for key in counts] == [0, 0, 1], state
        assert result.calls == [], state
        assert "Page range 'pages 9-'" in result.stderr, state
        if state == "fresh":
            assert _outputs(root) == []
            _ok(_extract(fake_llm, root, input=item))


def test_second_run_without_flags_resumes(
    fake_llm: FakeLLM, run_case: Callable[[str, Build], dict[str, Stage]]
) -> None:
    spec = SOURCES["text"]

    def build(root: Path) -> dict[str, Stage]:
        item = spec.make(root)
        first = _extract(fake_llm, root, input=item)
        _ok(first)
        second = _extract(fake_llm, root, input=item)
        return {
            "first": stage(first),
            "second": stage(second, outputs=_outputs(root)),
        }

    second = run_case("rerun_without_flags_text", build)["second"].result

    _ok(second)
    assert second.calls == []
    assert second.created == []


@pytest.mark.parametrize("source", list(SOURCES))
def test_resume_of_a_partial_input(
    source: str,
    fake_llm: FakeLLM,
    run_case: Callable[[str, Build], dict[str, Stage]],
) -> None:
    spec = SOURCES[source]

    def build(root: Path) -> dict[str, Stage]:
        item = spec.make(root)
        fake_llm.script(spec.failing, always=Reply(permanent_error()))
        first = _extract(fake_llm, root, input=item)
        assert first.exit_code == 1, first.stdout + first.stderr
        fake_llm.units.clear()
        second = _extract(fake_llm, root, input=item, resume=True)
        return {
            "first": stage(first),
            "second": stage(second, outputs=_outputs(root)),
        }

    second = run_case(f"resume_partial_{source}", build)["second"].result

    _ok(second)
    assert _units(second) == [spec.failing]
    for output in _outputs(second.root):
        metadata = json.loads(output.read_text(encoding="utf-8"))[
            "_chronominer_metadata"
        ]
        assert "failed_chunks" not in metadata


def test_dry_run_on_a_partly_completed_folder(
    fake_llm: FakeLLM, run_case: Callable[[str, Build], dict[str, Stage]]
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        docs = root / "docs"
        alpha = make_text_file(docs / "alpha.txt", 3)
        make_text_file(docs / "beta.txt", 3)
        done = _extract(fake_llm, root, input=alpha)
        _ok(done)
        planned = _extract(fake_llm, root, input=docs, resume=True, dry_run=True)
        return {
            "completed_one": stage(done),
            "dry_run": stage(planned, report=("[DRY RUN]",)),
        }

    planned = run_case("dry_run_partly_completed", build)["dry_run"].result

    _ok(planned)
    assert planned.created == []
    assert planned.calls == []
    assert planned.summary is not None
    assert planned.summary["dry_run"] is True
    assert planned.summary["skipped"] == ["alpha.txt"]
    assert [entry["name"] for entry in planned.summary["to_process"]] == ["beta.txt"]


@pytest.mark.parametrize("json_flag", [True, False])
def test_json_summary_line(
    json_flag: bool,
    fake_llm: FakeLLM,
    run_case: Callable[[str, Build], dict[str, Stage]],
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        ledger = make_text_file(root / "docs" / "ledger.txt", 3)
        result = _extract(fake_llm, root, input=ledger, json=json_flag, quiet=True)
        return {"run": stage(result)}

    case = "json_line" if json_flag else "no_json_line"
    result = run_case(case, build)["run"].result

    _ok(result)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    json_lines = [line for line in lines if line.startswith("{")]
    if json_flag:
        assert json_lines == [lines[-1]]
        assert json.loads(lines[-1]) == result.summary
    else:
        assert json_lines == []
        assert result.summary is None


def _exit_success(llm: FakeLLM, root: Path) -> RunResult:
    ledger = make_text_file(root / "docs" / "ledger.txt", RECORDS)
    return _extract(llm, root, input=ledger)


def _exit_failed_chunk(llm: FakeLLM, root: Path) -> RunResult:
    llm.script(record_unit(3), always=Reply(permanent_error()))
    ledger = make_text_file(root / "docs" / "ledger.txt", RECORDS)
    return _extract(llm, root, input=ledger)


def _exit_missing_input(llm: FakeLLM, root: Path) -> RunResult:
    return _extract(llm, root, input=root / "docs" / "missing.txt")


def _exit_invalid_page_range(llm: FakeLLM, root: Path) -> RunResult:
    pdf = make_scanned_pdf(root / "docs" / "book.pdf", PAGES)
    return _extract(llm, root, input=pdf, page_range="3-1")


def _exit_invalid_choice(llm: FakeLLM, root: Path) -> RunResult:
    ledger = make_text_file(root / "docs" / "ledger.txt", RECORDS)
    return _extract(llm, root, input=ledger, chunking="bogus")


@pytest.mark.parametrize(
    ("case", "run", "exit_code"),
    [
        ("exit_0_success", _exit_success, 0),
        ("exit_1_failed_chunk", _exit_failed_chunk, 1),
        ("exit_2_missing_input", _exit_missing_input, 2),
        ("exit_2_invalid_page_range", _exit_invalid_page_range, 2),
        ("exit_2_invalid_choice", _exit_invalid_choice, 2),
    ],
)
def test_exit_code(
    case: str,
    run: Callable[[FakeLLM, Path], RunResult],
    exit_code: int,
    fake_llm: FakeLLM,
    run_case: Callable[[str, Build], dict[str, Stage]],
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        return {"run": stage(run(fake_llm, root), report=("[ERROR]", "Error:"))}

    result = run_case(case, build)["run"].result

    assert result.exit_code == exit_code, result.console
    assert result.summary is not None
    if exit_code == 2:
        assert result.created == []
        assert result.calls == []
        assert result.summary["items_total"] == 0
        assert "Invalid arguments" in result.stderr
    else:
        assert result.summary["items_complete"] == 1 - exit_code
        assert result.summary["items_failed"] == exit_code
