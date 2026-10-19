"""The ``_line_ranges.txt`` renderer and its overwrite guard.

Run on Windows with:

    uv --directory <repo> run pytest tests/output/test_line_ranges_render.py -q

Range units are seeded as ``ranges generate`` and ``ranges adjust`` write
them; the file must match the old format byte for byte (the golden of
``ranges/generate/folder_default``), leave out deleted ranges, and a
hand-edited file must survive a render without ``force``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from chronominer.common.store import UnitResult
from chronominer.output.line_ranges import (
    format_range,
    line_ranges_path,
    parse_range,
    range_content,
)
from chronominer.output.render import render_ranges
from chronominer.output.store import KIND_RANGE, open_store
from tests.output.test_store import MODEL, NAME, PROVIDER, Unit, seed_document

GOLDEN = (
    Path(__file__).resolve().parents[1]
    / "characterization"
    / "golden"
    / "ranges"
    / "generate"
    / "folder_default"
    / "generate"
    / "files"
    / "docs"
    / "ledger_line_ranges.txt"
)
RANGES = [(1, 17), (18, 35), (36, 53), (54, 71)]


def _unit(idx: int, content: str | None) -> Unit:
    return Unit(idx, content=content, kind=KIND_RANGE)


def _render(
    root: Path, contents: Sequence[str | None], *, force: bool = False
) -> tuple[Path, list[Path], list[Path]]:
    path = line_ranges_path(root / "docs" / f"{NAME}.txt")

    async def main() -> tuple[list[Path], list[Path]]:
        async with open_store(root) as store:
            await store.begin_run("ranges")
            row = store.document(NAME)
            if row is None:
                units = [_unit(idx, content) for idx, content in enumerate(contents)]
                lease = await seed_document(store, units)
            else:
                lease = await store.acquire(row.doc_id)
                assert lease is not None
                for idx, content in enumerate(contents):
                    await store.write_unit(
                        lease,
                        UnitResult(KIND_RANGE, idx, "done", MODEL, PROVIDER, content),
                    )
            report = await render_ranges(store, NAME, path, lease, force=force)
            return report.written, report.kept

    written, kept = asyncio.run(main())
    return path, written, kept


def test_the_file_matches_the_old_format(tmp_path: Path) -> None:
    path, written, _ = _render(tmp_path, [range_content(*r) for r in RANGES])

    assert written == [path]
    assert path.name == "ledger_line_ranges.txt"
    assert path.read_bytes() == GOLDEN.read_bytes().replace(b"\r\n", b"\n")
    assert path.read_bytes() == b"(1, 17)\n(18, 35)\n(36, 53)\n(54, 71)\n"


def test_deleted_ranges_are_left_out(tmp_path: Path) -> None:
    path, _, _ = _render(tmp_path, ["[1, 18]", "[19, 36]", None, "[55, 71]"])

    assert path.read_text(encoding="utf-8") == "(1, 18)\n(19, 36)\n(55, 71)\n"


def test_the_guard_keeps_a_hand_edited_file_without_force(tmp_path: Path) -> None:
    path, _, _ = _render(tmp_path, [range_content(*r) for r in RANGES])
    path.write_text("(1, 35)\n(36, 71)\n", encoding="utf-8")

    _, written, kept = _render(
        tmp_path, ["[1, 20]", "[21, 35]", "[36, 53]", "[54, 71]"]
    )

    assert written == [] and kept == [path]
    assert path.read_text(encoding="utf-8") == "(1, 35)\n(36, 71)\n"

    _, forced, _ = _render(
        tmp_path, ["[1, 20]", "[21, 35]", "[36, 53]", "[54, 71]"], force=True
    )
    assert forced == [path]
    assert path.read_text(encoding="utf-8").startswith("(1, 20)\n(21, 35)\n")


def test_an_unchanged_rendering_is_not_rewritten(tmp_path: Path) -> None:
    contents = [range_content(*r) for r in RANGES]
    path, _, _ = _render(tmp_path, contents)

    _, written, kept = _render(tmp_path, contents)

    assert written == [] and kept == []
    assert path.read_bytes().count(b"\n") == 4


def test_range_content_round_trips() -> None:
    assert parse_range(range_content(3, 9)) == (3, 9)
    assert format_range(3, 9) == "(3, 9)"
    assert parse_range(None) is None
    assert parse_range("[1]") is None
    assert parse_range("not json") is None
