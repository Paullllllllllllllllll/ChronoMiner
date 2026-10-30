"""The store triggers that flag answers stale, and the component step.

Every writer of a chunk's entries or unit, this version's, an older
binary's or a raw SQL write, flags the answers about that chunk's entries;
a changed chunk layout flags the document's chunk answers. An older
ChronoMiner refuses the stepped store, and the tool's component refuses to
step without the classifications component.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.store import (
    Component,
    NewerSchemaError,
    PlannedUnit,
    Store,
    check_schema,
)
from chronominer.output.store import (
    COMPONENT,
    TOOL,
    MissingComponentError,
    database_path,
    open_store,
)
from tests.classify.conftest import (
    Chunk,
    Classify,
    DecisionServer,
    rows,
    seed,
    view,
    write_chunk,
    write_text,
)

BREAD = {"item": "bread", "unit": "lb"}
SALT = {"item": "salt", "unit": "bushel"}
CHUNKS = (
    Chunk(1, 15, entries=(BREAD,)),
    Chunk(16, 30, entries=(SALT,)),
    Chunk(31, 40, status="failed"),
)


@pytest.fixture
def classified(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> tuple[Path, int]:
    """A store whose two entries are classified; return its root and doc_id."""
    text = write_text(tmp_path / "prices.txt")
    doc_id = seed(tmp_path, text, CHUNKS)
    code, summary = classify(
        "entries", "--classifier", "commodity_group", "--input", tmp_path
    )
    assert code == 0, summary
    assert {row["locator"] for row in view(tmp_path)} == {"1:0", "2:0"}
    return tmp_path, doc_id


def _leased(
    root: Path, doc_id: int, work: Callable[[Any, Any], Awaitable[Any]]
) -> None:
    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("extract")
            lease = await store.acquire(doc_id)
            assert lease is not None
            await work(store, lease)

    asyncio.run(main())


def _stale(root: Path) -> dict[str, int]:
    return {
        row["locator"]: row["stale"]
        for row in rows(root, "SELECT locator, stale FROM classification_rows")
    }


def test_extracting_a_chunk_again_flags_its_entries(
    classified: tuple[Path, int],
) -> None:
    root, doc_id = classified

    _leased(root, doc_id, lambda store, lease: write_chunk(store, lease, 0, CHUNKS[0]))

    assert _stale(root) == {"1:0": 1, "2:0": 0}
    assert [row["locator"] for row in view(root)] == ["2:0"]


def test_a_force_reset_flags_every_entry(classified: tuple[Path, int]) -> None:
    root, doc_id = classified

    _leased(root, doc_id, lambda store, lease: store.reset_units(lease))

    assert _stale(root) == {"1:0": 1, "2:0": 1}
    assert view(root) == []


def test_repairing_a_failed_chunk_leaves_the_other_entries(
    classified: tuple[Path, int],
) -> None:
    root, doc_id = classified
    repaired = Chunk(31, 40, entries=({"item": "beer", "unit": "quart"},))

    _leased(root, doc_id, lambda store, lease: write_chunk(store, lease, 2, repaired))

    assert _stale(root) == {"1:0": 0, "2:0": 0}


def test_a_drop_deletes_the_entry_answers(classified: tuple[Path, int]) -> None:
    root, doc_id = classified

    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("drop")
            report = await store.drop([doc_id])
            assert report.dropped == ("prices",)

    asyncio.run(main())

    assert rows(root, "SELECT * FROM classification_rows") == []


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM entry_rows WHERE doc_id = :doc AND idx = 1",
        "UPDATE entry_rows SET entry = '{}' WHERE doc_id = :doc AND idx = 1",
        "UPDATE units SET stale = 1 WHERE doc_id = :doc AND kind = 'chunk' AND idx = 1",
    ],
    ids=["delete", "update", "unit"],
)
def test_a_raw_write_as_an_older_binary_does_flags_the_chunk(
    classified: tuple[Path, int], statement: str
) -> None:
    root, doc_id = classified

    with contextlib.closing(sqlite3.connect(database_path(root))) as connection:
        connection.execute(statement, {"doc": doc_id})
        connection.commit()

    assert _stale(root) == {"1:0": 0, "2:0": 1}


def test_an_older_binary_refuses_the_stepped_store(tmp_path: Path) -> None:
    async def create() -> None:
        async with open_store(tmp_path) as store:
            await store.begin_run("extract")

    asyncio.run(create())
    older = (Component(TOOL, COMPONENT.steps[:2]),)

    with pytest.raises(NewerSchemaError, match="'chronominer' version 3"):
        check_schema(database_path(tmp_path), older)
    with pytest.raises(NewerSchemaError):
        asyncio.run(Store(database_path(tmp_path), TOOL, components=older).open())


def test_a_store_of_the_older_binary_is_stepped_on_opening(tmp_path: Path) -> None:
    older = (Component(TOOL, COMPONENT.steps[:2]),)

    async def main() -> None:
        async with Store(database_path(tmp_path), TOOL, components=older) as store:
            await store.begin_run("extract")
        async with open_store(tmp_path) as store:
            await store.begin_run("classify chunks")

    asyncio.run(main())

    versions = check_schema(database_path(tmp_path)).versions
    assert (versions[TOOL], versions["classifications"]) == (3, 1)
    names = {
        row["name"]
        for row in rows(
            tmp_path, "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        )
    }
    assert {"entry_rows_stale_on_insert", "chunk_units_stale_on_update"} <= names


def test_the_tool_component_refuses_to_step_without_the_answer_table(
    tmp_path: Path,
) -> None:
    alone = Store(database_path(tmp_path), TOOL, components=(COMPONENT,))

    with pytest.raises(MissingComponentError, match="classifications"):
        asyncio.run(alone.open())

    assert TOOL not in check_schema(database_path(tmp_path)).versions


def test_a_changed_chunk_layout_flags_the_chunk_answers(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")
    doc_id = seed(tmp_path, text, CHUNKS)
    other = write_text(tmp_path / "wages.txt")
    seed(tmp_path, other, CHUNKS)
    for path in (text, other):
        code, _ = classify("chunks", "--classifier", "language", "--input", path)
        assert code == 0
    chunk_rows = "SELECT document, stale FROM classification_rows ORDER BY row_id"

    _leased(
        tmp_path, doc_id, lambda store, lease: write_chunk(store, lease, 0, CHUNKS[0])
    )
    assert {row["stale"] for row in rows(tmp_path, chunk_rows)} == {0}

    async def rechunk(store: Any, lease: Any) -> None:
        def drop(connection: sqlite3.Connection) -> None:
            connection.execute("DELETE FROM entry_rows WHERE doc_id = ?", (doc_id,))
            connection.execute(
                "DELETE FROM units WHERE doc_id = ? AND kind = 'chunk'", (doc_id,)
            )

        await store.write(lease, drop)
        layout = [PlannedUnit(0, "1-20", "1-20"), PlannedUnit(1, "21-40", "21-40")]
        await store.plan_units(lease, "chunk", layout, model="m", provider="openai")

    _leased(tmp_path, doc_id, rechunk)

    stale = {(row["document"], row["stale"]) for row in rows(tmp_path, chunk_rows)}
    assert stale == {("prices", 1), ("wages", 0)}
    assert {row["document"] for row in view(tmp_path)} == {"wages"}
