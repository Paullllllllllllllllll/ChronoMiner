"""Chunk naming, ``--pages`` metadata, text chunks and the held lease."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from chronominer.common.selection import parse_page_range
from chronominer.extract.chunks import (
    _HeldLease,
    build_custom_id,
    chunk_slice,
    custom_id,
    text_chunks,
)
from tests.extract.conftest import make_store, open_handle

_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class TestCustomIds:
    def test_a_safe_stem_is_kept(self) -> None:
        assert build_custom_id("ledger", "-chunk-3") == "ledger-chunk-3"

    def test_an_unsafe_stem_is_sanitized_and_hashed(self) -> None:
        first = build_custom_id("Rechnung 1701 (Zürich)", "-page-7")
        second = build_custom_id("Rechnung_1701__Zürich_", "-page-7")
        assert first.startswith("Rechnung_1701__Z_rich__")
        assert first.endswith("-page-7")
        assert first != second
        assert _SAFE.match(first)

    def test_a_long_stem_is_cut_to_64_characters(self) -> None:
        result = build_custom_id("a" * 120, "-chunk-12")
        assert len(result) == 64
        assert result.endswith("-chunk-12")
        assert _SAFE.match(result)

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("text", "ledger-chunk-3"),
            ("pdf", "ledger-page-3"),
            ("image", "ledger-page-3"),
        ],
    )
    def test_the_unit_word_follows_the_input_type(
        self, kind: str, expected: str
    ) -> None:
        assert custom_id(Path("in/ledger.txt"), kind, 2) == expected


@pytest.mark.parametrize(
    ("pages", "expected"),
    [
        (None, None),
        ("first:2", {"first_n": 2}),
        ("last:1", {"last_n": 1}),
        ("2-3", {"page_range": [2, 3]}),
        ("1,3", {"pages": parse_page_range("1,3").describe()}),
    ],
)
def test_the_chunk_slice_records_the_selection(
    pages: str | None, expected: dict[str, object] | None
) -> None:
    selection = parse_page_range(pages) if pages else None
    assert chunk_slice(selection) == expected


def test_text_chunks_join_their_lines() -> None:
    lines = ["Record 001.", "Item: wheat", "", "Record 002.", "Item: rye"]

    chunks = text_chunks(Path("ledger.txt"), lines, [(1, 3), (4, 5)])

    assert [chunk.text for chunk in chunks] == [
        "Record 001.\nItem: wheat\n",
        "Record 002.\nItem: rye",
    ]
    assert [chunk.image_name for chunk in chunks] == ["1-3", "4-5"]
    assert chunks[1].provenance() == {
        "custom_id": "ledger-chunk-2",
        "chunk_range": [4, 5],
    }


async def test_the_held_lease_is_reused_and_kept(tmp_path: Path) -> None:
    async with make_store(tmp_path / "chronominer.sqlite") as store:
        handle = await open_handle(store, pages=1)
        other = await store.ensure_document("other")
        held = _HeldLease(store, handle.lease)

        assert await held.acquire(handle.doc_id) == handle.lease
        await held.release(handle.lease)
        foreign = await held.acquire(other)
        assert foreign is not None and foreign != handle.lease
        await held.release(foreign)

        assert store.held_leases() == {handle.doc_id: handle.lease}
        await handle.write(0, status="done", content="still leased")
