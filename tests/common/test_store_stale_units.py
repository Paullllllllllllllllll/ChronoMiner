"""Stale units: Store.set_units_stale flags units whose inputs changed, and
Store.delete_units deletes the units a changed plan no longer has.

Run on Windows with:

    uv --directory <repo> run pytest tests/common/test_store_stale_units.py -q
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chronominer.common.sqlite import read_rows
from chronominer.common.store import (
    LEASE_TTL,
    LeaseLostError,
    PlannedUnit,
    ResponseRow,
    Store,
    SubmissionStateError,
    UnitKey,
)
from tests.common.store_child import FakeClock


def _stale(db: Path) -> dict[tuple[str, int], int]:
    return {(row["kind"], row["idx"]): row["stale"] for row in read_rows(db, "units")}


def test_set_units_stale_flags_only_the_given_units(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()
    seen: list[dict[tuple[str, int], int]] = []

    async def main() -> tuple[int, int, int]:
        async with Store(db, "tool", clock=clock) as store:
            await store.begin_run("run")
            doc_id = await store.ensure_document("doc")
            lease = await store.acquire(doc_id)
            assert lease is not None
            planned = [PlannedUnit(idx) for idx in range(4)]
            for kind in ("page", "chunk"):
                await store.plan_units(
                    lease, kind, planned, model="model-a", provider="openai"
                )
            flagged = await store.set_units_stale(lease, "page", [1, 2, 9])
            seen.append(_stale(db))
            cleared = await store.set_units_stale(lease, "page", [1], stale=False)
            seen.append(_stale(db))
            clock.advance(LEASE_TTL + 1.0)
            with pytest.raises(LeaseLostError, match="expired"):
                await store.set_units_stale(lease, "page", [0])
            missing = await _other_kind(store)
            return flagged, cleared, missing

    async def _other_kind(store: Store) -> int:
        doc_id = await store.ensure_document("other")
        lease = await store.acquire(doc_id)
        assert lease is not None
        return await store.set_units_stale(lease, "page", [0])

    flagged, cleared, missing = asyncio.run(main())

    assert flagged == 2
    assert cleared == 1
    assert missing == 0
    expected = {(kind, idx): 0 for kind in ("page", "chunk") for idx in range(4)}
    assert seen[0] == {**expected, ("page", 1): 1, ("page", 2): 1}
    assert seen[1] == {**expected, ("page", 2): 1}
    assert _stale(db) == seen[1]


def test_unit_changes_mark_the_rendered_files_stale(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    output = tmp_path / "doc.txt"
    output.write_text("rendered", encoding="utf-8")

    async def main() -> list[tuple[str, int]]:
        seen: list[tuple[str, int]] = []
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            doc_id = await store.ensure_document("doc")
            lease = await store.acquire(doc_id)
            assert lease is not None

            async def plan() -> int:
                units = [PlannedUnit(idx) for idx in range(2)]
                return await store.plan_units(
                    lease, "page", units, model="model-a", provider="openai"
                )

            changes = {
                "plan": plan,
                "plan nothing new": plan,
                "flag a unit": lambda: store.set_units_stale(lease, "page", [0]),
                "flag no unit": lambda: store.set_units_stale(lease, "page", [9]),
                "reset": lambda: store.reset_units(lease, kind="page"),
                "delete": lambda: store.delete_units(lease, "page", count=1),
                "delete nothing": lambda: store.delete_units(lease, "page", count=1),
            }
            for name, change in changes.items():
                await store.register_output(lease, output, format="txt")
                await change()
                (row,) = read_rows(db, "outputs")
                seen.append((name, row["stale"]))
        return seen

    assert asyncio.run(main()) == [
        ("plan", 1),
        ("plan nothing new", 0),
        ("flag a unit", 1),
        ("flag no unit", 0),
        ("reset", 1),
        ("delete", 1),
        ("delete nothing", 0),
    ]


def test_delete_units_drops_other_kinds_and_units_past_the_count(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> tuple[int, int, int]:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            doc_id = await store.ensure_document("doc")
            lease = await store.acquire(doc_id)
            assert lease is not None
            planned = [PlannedUnit(idx) for idx in range(4)]
            for kind in ("chunk", "text"):
                await store.plan_units(
                    lease, kind, planned, model="model-a", provider="openai"
                )
            text = [UnitKey(doc_id, "text", idx) for idx in range(4)]
            chunk = UnitKey(doc_id, "chunk", 3)
            settled = await store.open_submission(
                provider="openai", model="model-a", units=[text[0], chunk]
            )
            await store.mark_failed(settled.submission_id, "refused")
            await store.append_response(
                ResponseRow(provider="openai", model="model-a", unit=chunk)
            )
            held = await store.open_submission(
                provider="openai", model="model-a", units=[text[1]]
            )
            with pytest.raises(SubmissionStateError, match=held.submission_id):
                await store.delete_units(lease, "chunk")
            await store.mark_failed(held.submission_id, "refused")
            past = await store.delete_units(lease, "chunk", count=3)
            others = await store.delete_units(lease, "chunk")
            again = await store.delete_units(lease, "chunk", count=3)
            return past, others, again

    past, others, again = asyncio.run(main())

    assert (past, others, again) == (5, 0, 0)
    assert [(r["kind"], r["idx"]) for r in read_rows(db, "units")] == [
        ("chunk", 0),
        ("chunk", 1),
        ("chunk", 2),
    ]
    assert read_rows(db, "submission_requests") == []
    assert [(r["kind"], r["idx"]) for r in read_rows(db, "responses")] == [("chunk", 3)]
    (document,) = read_rows(db, "documents")
    assert (document["units_total"], document["units_pending"]) == (3, 3)
