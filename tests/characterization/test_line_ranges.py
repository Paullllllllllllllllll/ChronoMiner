"""Characterization of line-range generation (``ranges generate``).

Each case builds ledgers, runs one or more stages through ``run_tool`` in the
same root and snapshots every stage. The case runs twice in fresh roots; both
snapshots must match before each stage is compared with
``golden/ranges/generate/<case>/<stage>/``. Besides the standard snapshot, a
stage may record ``outputs/``: the current content of named files that an
earlier stage or the case itself created, such as a rewritten
``_line_ranges.txt``. Each folder with text inputs gets its
``chronominer.sqlite``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chronominer.cli.parser import UsageError, build_parser
from chronominer.common.testing import WriteGuard
from tests.characterization.adapter import (
    DATABASE_NAME,
    RunResult,
    build_argv,
    run_tool,
)
from tests.characterization.conftest import CaseRoot
from tests.characterization.fakes import FakeLLM
from tests.characterization.inputs import (
    heading_lines,
    make_text_file,
    ranges_path,
    read_ranges,
)
from tests.characterization.stages import RunCase, Stage, case_runner, stage

pytestmark = pytest.mark.integration

CASE_PREFIX = "ranges/generate"
RECORDS = 12
TOKENS = 60
STALE_RANGES = "(1, 3)\n(4, 9)\n"


@pytest.fixture
def run_case(case_root: CaseRoot, write_guard: WriteGuard) -> RunCase:
    """Return ``run(case, build)``: run twice, compare, check every stage."""
    return case_runner(CASE_PREFIX, case_root, write_guard)


def _ok(result: RunResult) -> None:
    assert result.exit_code == 0, result.stdout + result.stderr


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _assert_tiles(pairs: list[tuple[int, int]], first: int, last: int) -> None:
    """Assert that *pairs* cover lines *first* to *last* without gaps."""
    assert pairs[0][0] == first
    assert pairs[-1][1] == last
    for (_, end), (start, _) in zip(pairs, pairs[1:], strict=False):
        assert start == end + 1


def test_file_with_the_tokens_option(fake_llm: FakeLLM, run_case: RunCase) -> None:
    def build(root: Path) -> dict[str, Stage]:
        ledger = make_text_file(root / "docs" / "ledger.txt", RECORDS)
        result = run_tool(
            "generate_line_ranges", llm=fake_llm, root=root, input=ledger, tokens=TOKENS
        )
        return {"generate": stage(result)}

    stages = run_case("file_tokens", build)

    result = stages["generate"].result
    _ok(result)
    ledger = result.root / "docs" / "ledger.txt"
    assert result.created == [ledger.with_name(DATABASE_NAME), ranges_path(ledger)]
    pairs = read_ranges(ranges_path(ledger))
    assert len(pairs) > 2
    _assert_tiles(pairs, 1, _line_count(ledger))
    assert result.calls == []
    assert result.models == []


def test_folder_with_the_configured_default(
    fake_llm: FakeLLM, run_case: RunCase
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        docs = root / "docs"
        make_text_file(docs / "ledger.txt", RECORDS)
        make_text_file(docs / "notes.md", 4)
        make_text_file(docs / "sub" / "short.txt", 2)
        (docs / "scan.png").write_bytes(b"not text")
        result = run_tool(
            "generate_line_ranges", llm=fake_llm, root=root, input=docs, verbose=True
        )
        return {"generate": stage(result)}

    stages = run_case("folder_default", build)

    result = stages["generate"].result
    _ok(result)
    docs = result.root / "docs"
    texts = [docs / "ledger.txt", docs / "notes.md", docs / "sub" / "short.txt"]
    databases = [docs / DATABASE_NAME, docs / "sub" / DATABASE_NAME]
    assert sorted(result.created) == sorted(
        [*databases, *(ranges_path(text) for text in texts)]
    )
    for text in texts:
        _assert_tiles(read_ranges(ranges_path(text)), 1, _line_count(text))
    assert len(read_ranges(ranges_path(docs / "ledger.txt"))) < len(
        heading_lines(docs / "ledger.txt")
    )
    assert result.summary is not None
    assert (result.summary["items_total"], result.summary["items_complete"]) == (3, 3)


def test_generation_selects_no_chunks() -> None:
    with pytest.raises(TypeError, match="first_n_chunks"):
        build_argv("generate_line_ranges", {"input": "docs", "first_n_chunks": 2})
    with pytest.raises(UsageError):
        build_parser().parse_args(
            ["ranges", "generate", "--input", "docs", "--pages", "first:2"]
        )


def test_existing_ranges_file_is_overwritten(
    fake_llm: FakeLLM, run_case: RunCase
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        ledger = make_text_file(root / "docs" / "ledger.txt", RECORDS)
        ranges_path(ledger).write_text(STALE_RANGES, encoding="utf-8", newline="\n")
        first = run_tool(
            "generate_line_ranges", llm=fake_llm, root=root, input=ledger, tokens=TOKENS
        )
        stages = {"overwrite": stage(first, outputs=[ranges_path(ledger)])}
        again = run_tool(
            "generate_line_ranges",
            llm=fake_llm,
            root=root,
            input=ledger.parent,
            tokens=TOKENS * 2,
        )
        stages["rerun"] = stage(again, outputs=[ranges_path(ledger)])
        return stages

    stages = run_case("existing", build)

    overwrite = stages["overwrite"].result
    rerun = stages["rerun"].result
    _ok(overwrite)
    _ok(rerun)
    ledger = overwrite.root / "docs" / "ledger.txt"
    assert overwrite.created == [ledger.with_name(DATABASE_NAME)]
    assert rerun.created == []
    pairs = read_ranges(ranges_path(ledger))
    assert pairs != [(1, 3), (4, 9)]
    _assert_tiles(pairs, 1, _line_count(ledger))
    first_pairs = stages["overwrite"].files["outputs/docs/ledger_line_ranges.txt"]
    assert len(first_pairs.splitlines()) > len(pairs)


def test_generation_has_no_force_flag(fake_llm: FakeLLM, case_root: CaseRoot) -> None:
    root = case_root()
    ledger = make_text_file(root / "docs" / "ledger.txt", RECORDS)

    with pytest.raises(TypeError, match="force"):
        run_tool(
            "generate_line_ranges", llm=fake_llm, root=root, input=ledger, force=True
        )
    with pytest.raises(UsageError, match="--force"):
        build_parser().parse_args(["ranges", "generate", "--input", "docs", "--force"])
