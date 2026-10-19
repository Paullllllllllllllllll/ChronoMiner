"""Schema versions and the refusals of databases the store does not open.

Run on Windows with:

    uv --directory <repo> run pytest tests/common/test_store_versions.py -q
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from pathlib import Path

import pytest

from chronominer.common.sqlite import read_rows, table_names
from chronominer.common.store import (
    CORE_VERSION,
    Component,
    NewerSchemaError,
    Store,
    StoreError,
    UnreadableDatabaseError,
    VersionlessDatabaseError,
    check_schema,
)


def _open(db: Path, components: tuple[Component, ...] = ()) -> None:
    async def main() -> None:
        async with Store(db, "tool", components=components):
            pass

    asyncio.run(main())


def _versionless(db: Path) -> None:
    with contextlib.closing(sqlite3.connect(db)) as connection:
        connection.execute("CREATE TABLE documents (name TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE pages (document TEXT, text TEXT)")
        connection.execute("INSERT INTO documents VALUES ('doc')")
        connection.execute("INSERT INTO pages VALUES ('doc', 'text')")
        connection.commit()


def test_a_versionless_database_is_refused_and_left_unchanged(tmp_path: Path) -> None:
    db = tmp_path / "other.sqlite"
    _versionless(db)
    before = db.read_bytes()

    with pytest.raises(VersionlessDatabaseError) as raised:
        _open(db)

    assert raised.value.exit_code == 2
    assert isinstance(raised.value, StoreError)
    assert "not a tool store" in str(raised.value)
    assert "migrate" not in str(raised.value)
    assert db.read_bytes() == before
    assert not db.with_name(db.name + "-journal").exists()
    with pytest.raises(VersionlessDatabaseError, match="not a tool store"):
        check_schema(db, tool="tool")
    with pytest.raises(VersionlessDatabaseError, match="it is not a store"):
        check_schema(db)


def test_a_newer_core_schema_is_refused_and_left_unchanged(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    _open(db)
    with contextlib.closing(sqlite3.connect(db)) as connection:
        connection.execute(
            "UPDATE meta SET version = ? WHERE component = 'core'",
            (CORE_VERSION + 1,),
        )
        connection.commit()
    before = db.read_bytes()

    with pytest.raises(NewerSchemaError) as raised:
        _open(db)

    assert raised.value.exit_code == 2
    assert db.read_bytes() == before


def test_a_newer_tool_component_is_refused(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    two = Component("tool", ("CREATE TABLE a (x)", "CREATE TABLE b (x)"))
    one = Component("tool", ("CREATE TABLE a (x)",))
    _open(db, (two,))

    with pytest.raises(NewerSchemaError, match="'tool' version 2"):
        _open(db, (one,))


def test_a_file_that_is_not_a_database_is_refused_and_kept(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    content = b"This is a text file, not a database.\n" * 200
    db.write_bytes(content)

    with pytest.raises(UnreadableDatabaseError) as raised:
        _open(db)

    assert raised.value.exit_code == 2
    assert db.read_bytes() == content
    with pytest.raises(UnreadableDatabaseError):
        check_schema(db)


def test_only_writers_create_a_missing_database(tmp_path: Path) -> None:
    db = tmp_path / "sub" / "tool.sqlite"

    state = check_schema(db)

    assert not state.exists
    assert not db.exists()

    async def without_create() -> None:
        await Store(db, "tool").open(create=False)

    with pytest.raises(FileNotFoundError):
        asyncio.run(without_create())
    assert not db.exists()
    _open(db)
    assert check_schema(db).versions == {"core": CORE_VERSION}


def test_a_core_version_1_database_gains_the_batch_request_column(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    _open(db)
    with contextlib.closing(sqlite3.connect(db)) as connection:
        connection.execute("ALTER TABLE submission_requests DROP COLUMN request")
        connection.execute("UPDATE meta SET version = 1 WHERE component = 'core'")
        connection.commit()

    _open(db)

    state = check_schema(db)
    assert state.versions == {"core": CORE_VERSION}
    assert state.tables["submission_requests"][-1] == "request"


V1 = "CREATE TABLE extras (id INTEGER PRIMARY KEY, note TEXT);"
V2 = """
ALTER TABLE extras ADD COLUMN detail TEXT;
CREATE TABLE more (id INTEGER PRIMARY KEY);
CREATE TRIGGER extras_touch AFTER INSERT ON extras
BEGIN
    INSERT INTO more (id) VALUES (NEW.id);
END;
"""


def _seed_v1(db: Path) -> None:
    async def main() -> None:
        async with Store(db, "tool", components=(Component("tool", (V1,)),)) as store:
            await store._write(
                lambda c: c.execute("INSERT INTO extras (note) VALUES ('kept')")
            )

    asyncio.run(main())


def test_a_tool_component_migrates_from_version_one_to_two(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    _seed_v1(db)
    seen: list[str] = []

    def step_three(connection: sqlite3.Connection) -> None:
        seen.append("ran")
        connection.execute("UPDATE extras SET detail = 'filled'")

    _open(db, (Component("tool", (V1, V2)),))

    assert check_schema(db).versions == {"core": CORE_VERSION, "tool": 2}
    assert read_rows(db, "extras") == [{"id": 1, "note": "kept", "detail": None}]
    assert "more" in table_names(db)

    _open(db, (Component("tool", (V1, V2, step_three)),))

    assert seen == ["ran"]
    assert read_rows(db, "extras")[0]["detail"] == "filled"
    _open(db, (Component("tool", (V1, V2, step_three)),))
    assert seen == ["ran"]


def test_a_failing_migration_step_rolls_the_whole_upgrade_back(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    _seed_v1(db)
    broken = "ALTER TABLE extras ADD COLUMN detail TEXT;\nCREATE TABLE extras (x);"

    with pytest.raises(sqlite3.OperationalError, match="already exists"):
        _open(db, (Component("tool", (V1, broken)),))

    assert check_schema(db).versions == {"core": CORE_VERSION, "tool": 1}
    assert read_rows(db, "extras") == [{"id": 1, "note": "kept"}]
