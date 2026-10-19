"""Child process for the SQLite robustness tests.

Usage, from the repository root:

    python tests/common/sqlite_child.py race DB LABEL COUNT
        Open the writer, print READY, wait for a line on stdin, then commit
        COUNT single-row transactions labeled LABEL and print DONE.

    python tests/common/sqlite_child.py hot DB
        Commit five rows labeled ``committed``, then insert rows labeled
        ``killed`` inside one open transaction with a two-page cache until
        pages have spilled to the database file, print READY and sleep until
        the parent kills the process.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chronominer.common.sqlite import SqliteWriter, TableSpec  # noqa: E402

ROWS = TableSpec(
    name="rows",
    ddl=(
        "CREATE TABLE IF NOT EXISTS rows ("
        "id INTEGER PRIMARY KEY, label TEXT NOT NULL, seq INTEGER NOT NULL, "
        "payload BLOB)"
    ),
    key=("id",),
)
SPILL_ROWS = 400
SPILL_BYTES = 4096


def _say(word: str) -> None:
    print(word, flush=True)  # noqa: T201 - the parent reads stdout


async def race(db: Path, label: str, count: int) -> None:
    async with SqliteWriter(db, (ROWS,), busy_timeout=30.0) as writer:
        _say("READY")
        await asyncio.to_thread(sys.stdin.readline)
        for seq in range(count):
            await writer.transaction(
                lambda c, seq=seq: c.execute(
                    "INSERT INTO rows (label, seq) VALUES (?, ?)", (label, seq)
                )
            )
    _say("DONE")


def _spill(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA cache_size = 2")
    for seq in range(SPILL_ROWS):
        connection.execute(
            "INSERT INTO rows (label, seq, payload) VALUES ('killed', ?, "
            "randomblob(?))",
            (seq, SPILL_BYTES),
        )
    _say("READY")
    time.sleep(300)


async def hot(db: Path) -> None:
    async with SqliteWriter(db, (ROWS,)) as writer:
        for seq in range(5):
            await writer.transaction(
                lambda c, seq=seq: c.execute(
                    "INSERT INTO rows (label, seq) VALUES ('committed', ?)", (seq,)
                )
            )
        await writer.transaction(_spill)


def main(argv: list[str]) -> None:
    mode, db = argv[0], Path(argv[1])
    if mode == "race":
        asyncio.run(race(db, argv[2], int(argv[3])))
    elif mode == "hot":
        asyncio.run(hot(db))
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
