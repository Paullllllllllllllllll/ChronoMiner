"""Characterization of the extract command over every input source.

Text files and recursive text folders, image folders, PDFs and mixed folders,
the ``--type`` override, the flat and mirror output layouts, a page
range on a PDF and the first and last chunk slices. Each case runs twice in
fresh roots; both snapshots must match before the run is compared with
``golden/sources/<case>/``.
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
from tests.characterization.fakes import FakeLLM, page_unit, record_unit
from tests.characterization.inputs import (
    make_image,
    make_image_folder,
    make_mixed_folder,
    make_scanned_pdf,
    make_text_file,
)
from tests.characterization.stages import single_case_runner, stage

pytestmark = pytest.mark.integration

CASE_PREFIX = "sources"
RECORDS = 12
PAGES = 3
RANGE_PAGES = 5

Build = Callable[[Path], RunResult]


@pytest.fixture
def source_case(
    fake_llm: FakeLLM, case_root: CaseRoot, write_guard: WriteGuard
) -> Callable[[str, Build], RunResult]:
    """Return ``run(case, build)``: run twice, compare, check the golden files."""
    runner = single_case_runner(
        CASE_PREFIX, case_root, write_guard, before_each=fake_llm.units.clear
    )
    return lambda case, build: runner(case, lambda root: stage(build(root))).result


def _extract(llm: FakeLLM, root: Path, *, input: Path, **flags: Any) -> RunResult:
    return run_tool("extract", llm=llm, root=root, input=input, output="out", **flags)


def _records(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def _units(result: RunResult) -> list[str | None]:
    return [call["unit"] for call in result.calls]


def _ok(result: RunResult) -> None:
    assert result.exit_code == 0, result.stdout + result.stderr


def test_text_file(
    fake_llm: FakeLLM, source_case: Callable[[str, Build], RunResult]
) -> None:
    def build(root: Path) -> RunResult:
        ledger = make_text_file(root / "docs" / "ledger.txt", RECORDS)
        return _extract(fake_llm, root, input=ledger)

    result = source_case("text_file", build)

    _ok(result)
    ledger = result.root / "docs" / "ledger.txt"
    assert result.output_json(ledger) in result.created
    assert _units(result) == [record_unit(i) for i in (0, 3, 6, 9)]


def test_text_folder_recursive(
    fake_llm: FakeLLM, source_case: Callable[[str, Build], RunResult]
) -> None:
    def build(root: Path) -> RunResult:
        docs = root / "docs"
        make_text_file(docs / "alpha.txt", 3)
        make_text_file(docs / "nested" / "beta.md", 3)
        make_text_file(docs / "alpha_line_ranges.txt", 1)
        return _extract(fake_llm, root, input=docs)

    result = source_case("text_folder_recursive", build)

    _ok(result)
    assert result.summary is not None
    assert result.summary["items_total"] == 2
    out = result.root / "out"
    assert out / "alpha_output.json" in result.created
    assert out / "beta_output.json" in result.created


def test_image_folder(
    fake_llm: FakeLLM, source_case: Callable[[str, Build], RunResult]
) -> None:
    def build(root: Path) -> RunResult:
        folder = make_image_folder(root / "scans", PAGES)
        return _extract(fake_llm, root, input=folder)

    result = source_case("image_folder", build)

    _ok(result)
    assert sorted(_units(result)) == [page_unit(i) for i in range(PAGES)]
    assert result.summary is not None
    assert result.summary["items_total"] == PAGES


def test_pdf(fake_llm: FakeLLM, source_case: Callable[[str, Build], RunResult]) -> None:
    def build(root: Path) -> RunResult:
        pdf = make_scanned_pdf(root / "docs" / "book.pdf", PAGES)
        return _extract(fake_llm, root, input=pdf)

    result = source_case("pdf", build)

    _ok(result)
    assert _units(result) == [page_unit(i) for i in range(PAGES)]
    pdf = result.root / "docs" / "book.pdf"
    assert len(_records(result.output_json(pdf))) == PAGES


def test_pdf_page_range(
    fake_llm: FakeLLM, source_case: Callable[[str, Build], RunResult]
) -> None:
    def build(root: Path) -> RunResult:
        pdf = make_scanned_pdf(root / "docs" / "book.pdf", RANGE_PAGES)
        return _extract(fake_llm, root, input=pdf, page_range="2-3")

    result = source_case("pdf_page_range", build)

    _ok(result)
    assert _units(result) == [page_unit(1), page_unit(2)]


def test_mixed_folder(
    fake_llm: FakeLLM, source_case: Callable[[str, Build], RunResult]
) -> None:
    def build(root: Path) -> RunResult:
        folder = make_mixed_folder(root / "mixed")
        return _extract(fake_llm, root, input=folder)

    result = source_case("mixed_folder", build)

    _ok(result)
    out = result.root / "out"
    for stem in ("ledger", "scan", "book"):
        assert out / f"{stem}_output.json" in result.created
    assert {page_unit(0), page_unit(1), page_unit(2)} <= set(_units(result))
    assert record_unit(0) in _units(result)


def test_input_type_text_skips_a_stray_image(
    fake_llm: FakeLLM, source_case: Callable[[str, Build], RunResult]
) -> None:
    def build(root: Path) -> RunResult:
        docs = root / "docs"
        make_text_file(docs / "ledger.txt", 3)
        make_image(docs / "stray.png", 0)
        return _extract(fake_llm, root, input=docs, input_type="text")

    result = source_case("input_type_text", build)

    _ok(result)
    assert _units(result) == [record_unit(0)]
    assert not (result.root / "out" / "stray_output.json").exists()


def test_input_type_pdf_on_a_mixed_folder(
    fake_llm: FakeLLM, source_case: Callable[[str, Build], RunResult]
) -> None:
    def build(root: Path) -> RunResult:
        folder = make_mixed_folder(root / "mixed")
        return _extract(fake_llm, root, input=folder, input_type="pdf")

    result = source_case("input_type_pdf", build)

    _ok(result)
    assert _units(result) == [page_unit(0), page_unit(1)]
    out = result.root / "out"
    assert [path.name for path in result.created if path.parent == out] == [
        "book_output.csv",
        "book_output.json",
        "chronominer.sqlite",
    ]


@pytest.mark.parametrize("output_mode", ["flat", "mirror"])
def test_output_mode(
    output_mode: str,
    fake_llm: FakeLLM,
    source_case: Callable[[str, Build], RunResult],
) -> None:
    def build(root: Path) -> RunResult:
        docs = root / "docs"
        make_text_file(docs / "alpha.txt", 3)
        make_text_file(docs / "sub" / "deep" / "beta.txt", 3)
        return _extract(fake_llm, root, input=docs, output_mode=output_mode)

    result = source_case(f"output_mode_{output_mode}", build)

    _ok(result)
    out = result.root / "out"
    nested = out / "sub" / "deep" / "beta_output.json"
    flat = out / "beta_output.json"
    assert (nested if output_mode == "mirror" else flat) in result.created
    assert (flat if output_mode == "mirror" else nested) not in result.created


@pytest.mark.parametrize(
    ("case", "flags", "selected"),
    [
        ("first_n_chunks", {"first_n_chunks": 2}, (0, 3)),
        ("last_n_chunks", {"last_n_chunks": 1}, (9,)),
    ],
)
def test_chunk_slice(
    case: str,
    flags: dict[str, Any],
    selected: tuple[int, ...],
    fake_llm: FakeLLM,
    source_case: Callable[[str, Build], RunResult],
) -> None:
    def build(root: Path) -> RunResult:
        ledger = make_text_file(root / "docs" / "ledger.txt", RECORDS)
        return _extract(fake_llm, root, input=ledger, **flags)

    result = source_case(case, build)

    _ok(result)
    assert _units(result) == [record_unit(i) for i in selected]
