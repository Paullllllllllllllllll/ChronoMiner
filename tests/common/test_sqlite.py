"""The single-owner SQLite writer: concurrency, upserts, rollback and pragmas."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from chronominer.common import sqlite as sqlite_module
from chronominer.common.sqlite import (
    SqliteWriter,
    TableSpec,
    read,
    read_rows,
    table_names,
    to_sql_value,
)

DOCUMENTS = TableSpec(
    name="documents",
    ddl=(
        "CREATE TABLE IF NOT EXISTS documents ("
        "doc_hash TEXT PRIMARY KEY, path TEXT, model TEXT, spec TEXT)"
    ),
    key=("doc_hash",),
)
PAGES = TableSpec(
    name="pages",
    ddl=(
        "CREATE TABLE IF NOT EXISTS pages ("
        "doc_hash TEXT NOT NULL, page_index INTEGER NOT NULL, "
        "status TEXT NOT NULL, tokens INTEGER, ratio REAL, usage TEXT, "
        "PRIMARY KEY (doc_hash, page_index))"
    ),
    key=("doc_hash", "page_index"),
)
TABLES = (DOCUMENTS, PAGES)
PRAGMAS = ("journal_mode", "synchronous", "foreign_keys", "busy_timeout")


def _page(doc: str, index: int, status: str = "done", **extra: Any) -> dict[str, Any]:
    return {"doc_hash": doc, "page_index": index, "status": status, **extra}


async def _write_document(
    writer: SqliteWriter, doc: str, pages: int, status: str = "done"
) -> None:
    async with writer.document() as tx:
        tx.upsert("documents", {"doc_hash": doc, "path": f"{doc}.pdf"})
        for index in range(pages):
            await asyncio.sleep(0)
            tx.upsert("pages", _page(doc, index, status, tokens=index))


def _pragmas(connection: sqlite3.Connection) -> dict[str, Any]:
    return {
        name: connection.execute(f"PRAGMA {name}").fetchone()[0] for name in PRAGMAS
    }


def _assert_store_pragmas(values: dict[str, Any]) -> None:
    assert values["journal_mode"] == "delete"
    assert values["synchronous"] == 2
    assert values["foreign_keys"] == 1
    assert values["busy_timeout"] > 0


def test_concurrent_documents_from_many_tasks_land_without_loss(
    tmp_path: Path,
) -> None:
    db = tmp_path / "out" / "tool.sqlite"

    async def main() -> None:
        async with SqliteWriter(db, TABLES) as writer:
            await asyncio.gather(
                *(_write_document(writer, f"doc{n:02d}", 10) for n in range(20))
            )

    asyncio.run(main())

    assert len(read_rows(db, "documents")) == 20
    pages = read_rows(db, "pages", order_by=["doc_hash", "page_index"])
    assert len(pages) == 200
    assert {(row["doc_hash"], row["page_index"]) for row in pages} == {
        (f"doc{n:02d}", i) for n in range(20) for i in range(10)
    }


def test_rerun_upserts_without_duplicate_rows(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def run(status: str) -> None:
        async with SqliteWriter(db, TABLES) as writer:
            await _write_document(writer, "abc", 3, status)

    asyncio.run(run("failed"))
    asyncio.run(run("done"))

    pages = read_rows(db, "pages", order_by=["page_index"])
    assert [(row["page_index"], row["status"]) for row in pages] == [
        (0, "done"),
        (1, "done"),
        (2, "done"),
    ]
    assert len(read_rows(db, "documents")) == 1


def test_upsert_keeps_columns_the_rerun_does_not_set(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with SqliteWriter(db, TABLES) as writer:
            await writer.upsert("documents", [{"doc_hash": "a", "model": "m1"}])
            await writer.upsert("documents", [{"doc_hash": "a", "path": "a.pdf"}])
            await writer.upsert("documents", [{"doc_hash": "a"}])

    asyncio.run(main())

    assert read_rows(db, "documents") == [
        {"doc_hash": "a", "path": "a.pdf", "model": "m1", "spec": None}
    ]


def test_exception_inside_a_document_leaves_no_rows(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with SqliteWriter(db, TABLES) as writer:
            await _write_document(writer, "kept", 1)
            with pytest.raises(RuntimeError):
                async with writer.document() as tx:
                    tx.upsert("documents", {"doc_hash": "lost"})
                    tx.upsert("pages", _page("lost", 0))
                    raise RuntimeError("page failed")

    asyncio.run(main())

    assert [row["doc_hash"] for row in read_rows(db, "documents")] == ["kept"]
    assert [row["doc_hash"] for row in read_rows(db, "pages")] == ["kept"]


def test_database_error_mid_document_rolls_back_the_whole_document(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with SqliteWriter(db, TABLES) as writer:
            with pytest.raises(sqlite3.IntegrityError):
                async with writer.document() as tx:
                    tx.upsert("documents", {"doc_hash": "lost"})
                    tx.upsert("pages", _page("lost", 0))
                    tx.upsert("pages", {"doc_hash": "lost", "page_index": 1})
            await _write_document(writer, "after", 1)

    asyncio.run(main())

    assert [row["doc_hash"] for row in read_rows(db, "documents")] == ["after"]
    assert [row["doc_hash"] for row in read_rows(db, "pages")] == ["after"]


def test_two_writers_share_one_file_under_the_rollback_journal(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with (
            SqliteWriter(db, TABLES) as first,
            SqliteWriter(db, TABLES, busy_timeout=10.0) as second,
        ):
            await asyncio.gather(
                *(_write_document(first, f"a{n}", 5) for n in range(10)),
                *(_write_document(second, f"b{n}", 5) for n in range(10)),
            )

    asyncio.run(main())

    with contextlib.closing(sqlite3.connect(db)) as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "delete"
    assert len(read_rows(db, "documents")) == 20
    assert len(read_rows(db, "pages")) == 100


def test_flat_values_are_typed_and_nested_values_are_json(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    usage = {"input": 10, "output": 5, "details": [1, 2]}

    async def main() -> None:
        async with SqliteWriter(db, TABLES) as writer:
            await writer.upsert(
                "pages",
                [_page("a", 0, tokens=15, ratio=0.5, usage=usage)],
            )
            await writer.upsert("documents", [{"doc_hash": "a", "path": tmp_path}])

    asyncio.run(main())

    [page] = read_rows(db, "pages")
    assert page["tokens"] == 15 and isinstance(page["tokens"], int)
    assert page["ratio"] == 0.5
    assert json.loads(page["usage"]) == usage
    assert read_rows(db, "documents")[0]["path"] == str(tmp_path)
    assert to_sql_value(("x", "ü")) == '["x", "ü"]'
    assert to_sql_value(True) is True


def test_delete_removes_stale_rows_in_the_same_transaction(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with SqliteWriter(db, TABLES) as writer:
            await _write_document(writer, "a", 4)
            async with writer.document() as tx:
                tx.delete("pages", {"doc_hash": "a"})
                tx.upsert("pages", _page("a", 0))

    asyncio.run(main())

    assert [row["page_index"] for row in read_rows(db, "pages")] == [0]


def test_delete_keeps_the_listed_rows(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with SqliteWriter(db, TABLES) as writer:
            await _write_document(writer, "a", 5)
            await _write_document(writer, "b", 2)
            async with writer.document() as tx:
                tx.delete("pages", {"doc_hash": "a"}, keep={"page_index": [1, 3]})
                tx.delete("pages", {"doc_hash": "b"}, keep={"page_index": []})

    asyncio.run(main())

    rows = read_rows(db, "pages", order_by=["doc_hash", "page_index"])
    assert [(row["doc_hash"], row["page_index"]) for row in rows] == [
        ("a", 1),
        ("a", 3),
    ]


def test_writing_and_reading_leave_only_the_database_file(tmp_path: Path) -> None:
    folder = tmp_path / "out"
    db = folder / "tool.sqlite"

    async def main() -> None:
        async with SqliteWriter(db, TABLES) as writer:
            await _write_document(writer, "a", 1)

    asyncio.run(main())
    assert [path.name for path in folder.iterdir()] == ["tool.sqlite"]
    assert read_rows(db, "documents")[0]["doc_hash"] == "a"
    assert table_names(db) == ["documents", "pages"]

    assert [path.name for path in folder.iterdir()] == ["tool.sqlite"]
    with pytest.raises(FileNotFoundError):
        read_rows(folder / "missing.sqlite", "documents")
    assert not (folder / "missing.sqlite").exists()


def test_invalid_writes_fail_before_reaching_the_database(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with SqliteWriter(db, TABLES) as writer:
            with pytest.raises(ValueError, match="lacks key"):
                await writer.upsert("pages", [{"doc_hash": "a", "status": "done"}])
            with pytest.raises(KeyError):
                await writer.upsert("entries", [{"id": 1}])
            with pytest.raises(ValueError, match="identifier"):
                await writer.upsert("documents", [{"doc_hash": "a", 'x"; --': 1}])
            with pytest.raises(ValueError):
                writer.document().delete("pages", {})

    asyncio.run(main())

    assert read_rows(db, "pages") == []


def test_table_spec_validation() -> None:
    with pytest.raises(ValueError):
        TableSpec(name="pages", ddl="", key=())
    with pytest.raises(ValueError):
        TableSpec(name="bad name", ddl="", key=("id",))


def test_writer_must_be_open(tmp_path: Path) -> None:
    writer = SqliteWriter(tmp_path / "tool.sqlite", TABLES)

    with pytest.raises(RuntimeError, match="not open"):
        asyncio.run(writer.upsert("documents", [{"doc_hash": "a"}]))
    with pytest.raises(RuntimeError, match="not open"):
        asyncio.run(writer.transaction(lambda connection: None))


def test_reopening_keeps_rows_and_lists_tables(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def write(doc: str) -> None:
        async with SqliteWriter(db, TABLES) as writer:
            await _write_document(writer, doc, 1)

    asyncio.run(write("a"))
    asyncio.run(write("b"))

    assert table_names(db) == ["documents", "pages"]
    assert [row["doc_hash"] for row in read_rows(db, "documents")] == ["a", "b"]


def test_transaction_returns_the_result_of_the_function(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    def insert(connection: sqlite3.Connection) -> int:
        cursor = connection.execute(
            "INSERT INTO documents (doc_hash, path) VALUES ('a', 'a.pdf')"
        )
        assert connection.in_transaction
        return int(cursor.lastrowid or 0)

    async def main() -> int:
        async with SqliteWriter(db, TABLES) as writer:
            return await writer.transaction(insert)

    assert asyncio.run(main()) == 1
    assert read_rows(db, "documents")[0]["path"] == "a.pdf"


def test_transaction_rolls_back_and_reraises(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    def fail(connection: sqlite3.Connection) -> None:
        connection.execute("INSERT INTO documents (doc_hash) VALUES ('lost')")
        raise LookupError("no such page")

    async def main() -> None:
        async with SqliteWriter(db, TABLES) as writer:
            with pytest.raises(LookupError, match="no such page"):
                await writer.transaction(fail)
            await writer.upsert("documents", [{"doc_hash": "kept"}])

    asyncio.run(main())

    assert [row["doc_hash"] for row in read_rows(db, "documents")] == ["kept"]


def test_writer_connection_uses_the_rollback_journal(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> dict[str, Any]:
        async with SqliteWriter(db, TABLES, busy_timeout=2.0) as writer:
            return await writer.transaction(_pragmas)

    values = asyncio.run(main())

    _assert_store_pragmas(values)
    assert values["busy_timeout"] == 2000


def test_read_helper_opens_read_only_connections_with_the_pragmas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with SqliteWriter(db, TABLES) as writer:
            await _write_document(writer, "a", 1)

    asyncio.run(main())
    targets: list[tuple[Any, bool]] = []
    real_connect = sqlite3.connect

    def connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        targets.append((database, kwargs.get("uri", False)))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite_module.sqlite3, "connect", connect)

    _assert_store_pragmas(read(db, _pragmas))
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        read(db, lambda c: c.execute("INSERT INTO documents (doc_hash) VALUES ('x')"))
    read_rows(db, "documents")
    table_names(db)

    assert len(targets) == 4
    for database, uri in targets:
        assert uri is True
        assert str(database).startswith("file:")
        assert str(database).endswith("?mode=ro")
    assert [row["doc_hash"] for row in read_rows(db, "documents")] == ["a"]


def test_writer_switches_a_wal_database_to_the_rollback_journal(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "out"
    folder.mkdir()
    db = folder / "tool.sqlite"
    with contextlib.closing(sqlite3.connect(db)) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
        connection.execute(DOCUMENTS.ddl)
        connection.execute("INSERT INTO documents (doc_hash) VALUES ('old')")
        connection.commit()
    assert db.read_bytes()[18:20] == b"\x02\x02"

    async def main() -> dict[str, Any]:
        async with SqliteWriter(db, TABLES) as writer:
            await writer.upsert("documents", [{"doc_hash": "new"}])
            return await writer.transaction(_pragmas)

    _assert_store_pragmas(asyncio.run(main()))

    assert db.read_bytes()[18:20] == b"\x01\x01"
    assert [path.name for path in folder.iterdir()] == ["tool.sqlite"]
    assert [row["doc_hash"] for row in read_rows(db, "documents")] == ["old", "new"]


def test_read_helper_switches_an_idle_wal_database(tmp_path: Path) -> None:
    folder = tmp_path / "out"
    folder.mkdir()
    db = folder / "legacy.sqlite"
    with contextlib.closing(sqlite3.connect(db)) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(DOCUMENTS.ddl)
        connection.execute("INSERT INTO documents (doc_hash) VALUES ('old')")
        connection.commit()

    assert read(db, _pragmas)["journal_mode"] == "delete"
    assert [path.name for path in folder.iterdir()] == ["legacy.sqlite"]


def test_foreign_keys_are_enforced_on_the_writer(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    child = TableSpec(
        name="children",
        ddl=(
            "CREATE TABLE IF NOT EXISTS children (id INTEGER PRIMARY KEY, "
            "doc_hash TEXT NOT NULL REFERENCES documents (doc_hash))"
        ),
        key=("id",),
    )

    async def main() -> None:
        async with SqliteWriter(db, (*TABLES, child)) as writer:
            with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
                await writer.upsert("children", [{"id": 1, "doc_hash": "none"}])

    asyncio.run(main())

    assert read_rows(db, "children") == []


def test_a_failed_commit_rolls_back_and_later_writes_work(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    child = TableSpec(
        name="children",
        ddl=(
            "CREATE TABLE IF NOT EXISTS children (id INTEGER PRIMARY KEY, "
            "doc_hash TEXT NOT NULL REFERENCES documents (doc_hash) "
            "DEFERRABLE INITIALLY DEFERRED)"
        ),
        key=("id",),
    )

    async def main() -> None:
        async with SqliteWriter(db, (*TABLES, child)) as writer:
            with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
                await writer.upsert("children", [{"id": 1, "doc_hash": "none"}])
            await writer.upsert("documents", [{"doc_hash": "kept"}])

    asyncio.run(main())

    assert read_rows(db, "children") == []
    assert [row["doc_hash"] for row in read_rows(db, "documents")] == ["kept"]


def test_queue_size_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="queue_size"):
        SqliteWriter(tmp_path / "tool.sqlite", TABLES, queue_size=0)
