"""SQLite storage written by a single owner of the connection.

One database file per output folder. Every connection this module opens uses
the rollback journal (``journal_mode=DELETE``) with ``synchronous=FULL``,
foreign keys on and a busy timeout; WAL is never used, because the SQLite
bundled with current Python releases carries the WAL-reset corruption bug. A
database file found in WAL mode is switched to the rollback journal when the
writer opens it.

Within a process, one dedicated thread owns the writer's connection and runs
every write, which serializes writes submitted from parallel tasks. A write is
a function run inside ``BEGIN IMMEDIATE ... COMMIT``; it commits or rolls back
as a whole. With the rollback journal a commit waits until every open read has
ended, so a write that is still busy after the busy timeout stays in memory
and is retried with backoff, without limit, while a callback warns at most
once per interval. Any other error (a full disk, a constraint) propagates at
once. The writer counts pending writes, and pipelines await
:meth:`SqliteWriter.wait_capacity` to pause new work while writes wait.

Readers open short read-only connections (``file:...?mode=ro``) through
:func:`read_only`. A hot journal left by a killed writer cannot be rolled back
through such a connection; the helper then opens a read-write connection once,
whose first read rolls the journal back, and reads again. A writer waiting to
commit refuses new reads, so a read that is still busy after the busy timeout
is retried with the writer's backoff, without limit. While a writer of this
process has the file open, the read shares its warning callback and interval;
otherwise it logs. The connection then holds one read transaction until it
closes, so its queries see one state and are not refused later.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import re
import sqlite3
import threading
import time
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePath
from types import TracebackType
from typing import Any, TypeVar

DEFAULT_BUSY_TIMEOUT = 30.0
DEFAULT_WARN_INTERVAL = 60.0
DEFAULT_QUEUE_SIZE = 32
RETRY_DELAY_MIN = 0.05
RETRY_DELAY_MAX = 2.0

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BUSY_CODES = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})
_READONLY_ROLLBACK = 776
_WAL_HEADER_OFFSET = 18

logger = logging.getLogger(__name__)

T = TypeVar("T")
BusyCallback = Callable[[Path, float], None]


class DatabaseFullError(sqlite3.OperationalError):
    """The database or its disk is full (``SQLITE_FULL``)."""


@dataclass(frozen=True)
class TableSpec:
    """A table: its name, its DDL and the key columns that identify a row.

    The DDL must declare a primary key or unique constraint on exactly the key
    columns, and should use ``IF NOT EXISTS`` so that reopening a database
    works.
    """

    name: str
    ddl: str
    key: tuple[str, ...]

    def __post_init__(self) -> None:
        _quote(self.name)
        if not self.key:
            raise ValueError(f"Table {self.name!r} declares no key columns.")
        for column in self.key:
            _quote(column)


def _quote(identifier: str) -> str:
    if not _IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def to_sql_value(value: Any) -> Any:
    """Convert a Python value to a column value.

    Mappings, lists and tuples become JSON text; paths become strings; other
    values pass through to the driver.
    """
    if isinstance(value, Mapping | list | tuple):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, PurePath):
        return str(value)
    return value


def _error_code(exc: BaseException) -> int:
    return int(getattr(exc, "sqlite_errorcode", 0) or 0)


def is_busy(exc: BaseException) -> bool:
    """Return True for ``SQLITE_BUSY`` and ``SQLITE_LOCKED`` errors."""
    return isinstance(exc, sqlite3.OperationalError) and (
        (_error_code(exc) & 0xFF) in _BUSY_CODES
    )


def _full_error(exc: BaseException) -> DatabaseFullError | None:
    if not isinstance(exc, sqlite3.Error) or isinstance(exc, DatabaseFullError):
        return None
    if (_error_code(exc) & 0xFF) != sqlite3.SQLITE_FULL:
        return None
    full = DatabaseFullError(str(exc))
    full.sqlite_errorcode = _error_code(exc)
    full.sqlite_errorname = getattr(exc, "sqlite_errorname", "SQLITE_FULL")
    return full


def configure(connection: sqlite3.Connection, busy_timeout: float) -> None:
    """Apply the busy timeout, rollback journal, full sync and foreign keys.

    Switching a WAL database to the rollback journal needs every other
    connection to the file closed; otherwise SQLite reports it busy.
    """
    connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout * 1000)}")
    (mode,) = connection.execute("PRAGMA journal_mode = DELETE").fetchone()
    if str(mode).lower() != "delete":
        raise sqlite3.OperationalError(
            f"Journal mode stays {mode!r}; close other connections to the file."
        )
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA foreign_keys = ON")


@dataclass(frozen=True)
class _Statement:
    sql: str
    params: tuple[Any, ...]


def _upsert_statement(spec: TableSpec, row: Mapping[str, Any]) -> _Statement:
    missing = [column for column in spec.key if column not in row]
    if missing:
        raise ValueError(f"Row for {spec.name!r} lacks key column(s) {missing}.")
    columns = list(row)
    names = ", ".join(_quote(column) for column in columns)
    marks = ", ".join("?" for _ in columns)
    key = ", ".join(_quote(column) for column in spec.key)
    updates = [
        f"{_quote(column)} = excluded.{_quote(column)}"
        for column in columns
        if column not in spec.key
    ]
    action = f"DO UPDATE SET {', '.join(updates)}" if updates else "DO NOTHING"
    sql = (
        f"INSERT INTO {_quote(spec.name)} ({names}) VALUES ({marks}) "
        f"ON CONFLICT ({key}) {action}"
    )
    return _Statement(sql, tuple(to_sql_value(row[column]) for column in columns))


def _delete_statement(
    spec: TableSpec,
    match: Mapping[str, Any],
    keep: Mapping[str, Collection[Any]] | None = None,
) -> _Statement:
    if not match:
        raise ValueError(f"A delete from {spec.name!r} needs at least one column.")
    where = " AND ".join(f"{_quote(column)} = ?" for column in match)
    params = [to_sql_value(value) for value in match.values()]
    kept = {column: list(values) for column, values in (keep or {}).items()}
    if kept and all(kept.values()):
        clauses = []
        for column, values in kept.items():
            marks = ", ".join("?" for _ in values)
            clauses.append(f"{_quote(column)} IN ({marks})")
            params.extend(to_sql_value(value) for value in values)
        where += f" AND NOT ({' AND '.join(clauses)})"
    sql = f"DELETE FROM {_quote(spec.name)} WHERE {where}"
    return _Statement(sql, tuple(params))


def _apply(connection: sqlite3.Connection, statements: Sequence[_Statement]) -> None:
    for statement in statements:
        connection.execute(statement.sql, statement.params)


class DocumentTransaction:
    """Collects the writes of one document and commits them together.

    Use it as an async context manager from :meth:`SqliteWriter.document`.
    Writes are validated when added and sent to the writer on a clean exit;
    an exception inside the block discards them, so no partial rows land.
    """

    def __init__(self, writer: SqliteWriter) -> None:
        self._writer = writer
        self._statements: list[_Statement] = []

    def upsert(self, table: str, row: Mapping[str, Any]) -> None:
        """Insert *row* or update the existing row with the same key."""
        spec = self._writer.table(table)
        self._statements.append(_upsert_statement(spec, row))

    def delete(
        self,
        table: str,
        match: Mapping[str, Any],
        *,
        keep: Mapping[str, Collection[Any]] | None = None,
    ) -> None:
        """Delete the rows whose columns equal the values in *match*.

        With *keep*, rows whose every *keep* column holds one of its listed
        values stay; an empty list keeps nothing.
        """
        spec = self._writer.table(table)
        self._statements.append(_delete_statement(spec, match, keep))

    async def __aenter__(self) -> DocumentTransaction:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        statements, self._statements = self._statements, []
        if exc_type is None and statements:
            await self._writer._execute(statements)


def _runs_on(loop: asyncio.AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


class _BusyWait:
    """Backoff between busy retries on one file, warning once per interval.

    The callback runs on *loop* when it is set and the caller runs on another
    thread; otherwise it runs directly.
    """

    def __init__(
        self, path: Path, warn_interval: float, on_busy: BusyCallback | None
    ) -> None:
        self.path = path
        self.warn_interval = warn_interval
        self.on_busy = on_busy
        self.loop: asyncio.AbstractEventLoop | None = None
        self._last_warning: float | None = None
        self._lock = threading.Lock()

    def wait(self, started: float, delay: float) -> float:
        """Warn when due, sleep *delay* seconds and return the next delay."""
        now = time.monotonic()
        with self._lock:
            last = self._last_warning
            due = last is None or now - last >= self.warn_interval
            if due:
                self._last_warning = now
        if due:
            self._warn(now - started)
        time.sleep(delay)
        return min(delay * 2, RETRY_DELAY_MAX)

    def _warn(self, waited: float) -> None:
        callback = self.on_busy or _log_busy
        loop = self.loop
        if loop is not None and not loop.is_closed() and not _runs_on(loop):
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(callback, self.path, waited)
                return
        callback(self.path, waited)


_OPEN_WRITERS: dict[Path, list[_BusyWait]] = {}
_OPEN_WRITERS_LOCK = threading.Lock()


def _register(busy: _BusyWait) -> None:
    with _OPEN_WRITERS_LOCK:
        _OPEN_WRITERS.setdefault(busy.path.resolve(), []).append(busy)


def _unregister(busy: _BusyWait) -> None:
    key = busy.path.resolve()
    with _OPEN_WRITERS_LOCK:
        waits = _OPEN_WRITERS.get(key, [])
        if busy in waits:
            waits.remove(busy)
        if not waits:
            _OPEN_WRITERS.pop(key, None)


def _busy_wait(path: Path) -> _BusyWait:
    """Return the busy wait of the newest open writer on *path*, else a new one."""
    with _OPEN_WRITERS_LOCK:
        waits = _OPEN_WRITERS.get(path.resolve())
        if waits:
            return waits[-1]
    return _BusyWait(path, DEFAULT_WARN_INTERVAL, None)


class SqliteWriter:
    """Owns one connection to a database file and serializes all writes.

    Open it with ``async with SqliteWriter(path, tables) as writer``. Run a
    write with :meth:`transaction`, or collect upserts with
    ``async with writer.document() as tx`` or :meth:`upsert`. A write whose
    awaiting task is cancelled still runs in the writer thread, committed or
    rolled back as a whole. *on_busy* receives the database path and the
    seconds a busy write, or a busy :func:`read` of the file while the writer
    is open, has waited; it runs on the event loop that opened the writer, at
    most once per *warn_interval*.
    """

    def __init__(
        self,
        path: Path,
        tables: Sequence[TableSpec] = (),
        *,
        busy_timeout: float = DEFAULT_BUSY_TIMEOUT,
        warn_interval: float = DEFAULT_WARN_INTERVAL,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        on_busy: BusyCallback | None = None,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1.")
        self.path = Path(path)
        self._tables = {spec.name: spec for spec in tables}
        self._busy_timeout = busy_timeout
        self._queue_size = queue_size
        self._busy = _BusyWait(self.path, warn_interval, on_busy)
        self._executor: ThreadPoolExecutor | None = None
        self._connection: sqlite3.Connection | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._room: asyncio.Event | None = None
        self._pending = 0

    @property
    def pending(self) -> int:
        """Return the number of writes submitted and not yet finished."""
        return self._pending

    @property
    def queue_size(self) -> int:
        """Return the number of pending writes at which callers should pause."""
        return self._queue_size

    def table(self, name: str) -> TableSpec:
        """Return the declared table *name*; ``KeyError`` when undeclared."""
        try:
            return self._tables[name]
        except KeyError:
            raise KeyError(f"Table {name!r} is not declared for {self.path}") from None

    async def open(self) -> None:
        """Open the database, set its pragmas and create the declared tables."""
        if self._executor is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._busy.loop = self._loop
        self._room = asyncio.Event()
        self._room.set()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sqlite-writer"
        )
        try:
            await self._run(self._connect)
        except BaseException:
            self._executor.shutdown(wait=False)
            self._executor = None
            raise
        _register(self._busy)

    async def close(self) -> None:
        """Wait for the submitted writes, close the connection, stop the thread."""
        executor = self._executor
        if executor is None:
            return
        try:
            await self._run(self._disconnect)
        finally:
            _unregister(self._busy)
            self._executor = None
            executor.shutdown(wait=False)

    async def __aenter__(self) -> SqliteWriter:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    def document(self) -> DocumentTransaction:
        """Return a transaction that commits one document's writes together."""
        return DocumentTransaction(self)

    async def upsert(self, table: str, rows: Iterable[Mapping[str, Any]]) -> None:
        """Upsert *rows* into *table* in one transaction."""
        spec = self.table(table)
        await self._execute([_upsert_statement(spec, row) for row in rows])

    async def transaction(self, func: Callable[[sqlite3.Connection], T]) -> T:
        """Run *func* with the connection inside one transaction; return its result.

        The writer thread runs ``BEGIN IMMEDIATE``, calls *func* and commits;
        an error rolls the transaction back and is re-raised, a full database
        as :class:`DatabaseFullError`. When a statement inside *func* finds
        the database busy, the transaction is rolled back and *func* runs
        again, so *func* must act only through the connection.
        """
        executor = self._executor
        loop = self._loop
        if executor is None or loop is None:
            raise RuntimeError(f"SQLite writer for {self.path} is not open.")
        future = executor.submit(self._transact, func)
        self._pending += 1
        if self._room is not None and self._pending >= self._queue_size:
            self._room.clear()
        future.add_done_callback(self._finished)
        wrapped = asyncio.wrap_future(future, loop=loop)
        wrapped.add_done_callback(_consume)
        return await asyncio.shield(wrapped)

    async def wait_capacity(self) -> None:
        """Return once fewer than ``queue_size`` writes are pending."""
        room = self._room
        if room is None:
            raise RuntimeError(f"SQLite writer for {self.path} is not open.")
        while self._pending >= self._queue_size:
            room.clear()
            await room.wait()

    async def _execute(self, statements: Sequence[_Statement]) -> None:
        await self.transaction(lambda connection: _apply(connection, statements))

    async def _run(self, func: Callable[[], None]) -> None:
        if self._executor is None or self._loop is None:
            raise RuntimeError(f"SQLite writer for {self.path} is not open.")
        await self._loop.run_in_executor(self._executor, func)

    def _finished(self, _future: concurrent.futures.Future[Any]) -> None:
        loop = self._loop
        if loop is None:
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._release)

    def _release(self) -> None:
        self._pending = max(0, self._pending - 1)
        if self._room is not None and self._pending < self._queue_size:
            self._room.set()

    def _connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path, timeout=self._busy_timeout, isolation_level=None
        )
        started = time.monotonic()
        delay = RETRY_DELAY_MIN
        try:
            while True:
                try:
                    configure(connection, self._busy_timeout)
                    for spec in self._tables.values():
                        connection.executescript(spec.ddl)
                    break
                except sqlite3.OperationalError as exc:
                    if not is_busy(exc):
                        raise
                    delay = self._busy.wait(started, delay)
        except BaseException:
            connection.close()
            raise
        self._connection = connection

    def _disconnect(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _transact(self, func: Callable[[sqlite3.Connection], T]) -> T:
        connection = self._connection
        if connection is None:
            raise RuntimeError(f"SQLite writer for {self.path} is not open.")
        started = time.monotonic()
        delay = RETRY_DELAY_MIN
        while True:
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                if not is_busy(exc):
                    raise
                delay = self._busy.wait(started, delay)
                continue
            try:
                result = func(connection)
            except BaseException as exc:
                _rollback(connection)
                if not is_busy(exc):
                    full = _full_error(exc)
                    if full is not None:
                        raise full from exc
                    raise
                delay = self._busy.wait(started, delay)
                continue
            while True:
                try:
                    connection.execute("COMMIT")
                    return result
                except BaseException as exc:
                    if not is_busy(exc):
                        _rollback(connection)
                        full = _full_error(exc)
                        if full is not None:
                            raise full from exc
                        raise
                    if not connection.in_transaction:
                        break
                    delay = self._busy.wait(started, delay)
            delay = self._busy.wait(started, delay)


def _log_busy(path: Path, waited: float) -> None:
    logger.warning(
        "Database %s is busy (another connection holds a transaction); "
        "waited %.0f s, retrying.",
        path,
        waited,
    )


def _rollback(connection: sqlite3.Connection) -> None:
    if connection.in_transaction:
        with contextlib.suppress(sqlite3.Error):
            connection.execute("ROLLBACK")


def _consume(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


def _is_wal_file(path: Path) -> bool:
    with path.open("rb") as handle:
        header = handle.read(_WAL_HEADER_OFFSET + 2)
    return header[_WAL_HEADER_OFFSET : _WAL_HEADER_OFFSET + 2] == b"\x02\x02"


def _open_read_only(path: Path, busy_timeout: float) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(
        uri, uri=True, timeout=busy_timeout, isolation_level=None
    )
    try:
        connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout * 1000)}")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN")
        connection.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except BaseException:
        connection.close()
        raise
    return connection


def _open_recovering(path: Path, busy_timeout: float) -> sqlite3.Connection:
    try:
        return _open_read_only(path, busy_timeout)
    except sqlite3.OperationalError as exc:
        if _error_code(exc) != _READONLY_ROLLBACK:
            raise
    _recover(path, busy_timeout)
    return _open_read_only(path, busy_timeout)


def _recover(path: Path, busy_timeout: float) -> None:
    """Roll back a hot journal and leave WAL mode through a read-write open."""
    connection = sqlite3.connect(path, timeout=busy_timeout, isolation_level=None)
    try:
        configure(connection, busy_timeout)
        connection.execute("SELECT count(*) FROM sqlite_master").fetchone()
    finally:
        connection.close()


@contextlib.contextmanager
def read_only(
    path: Path, *, busy_timeout: float = DEFAULT_BUSY_TIMEOUT
) -> Iterator[sqlite3.Connection]:
    """Open an existing database through a ``mode=ro`` connection; close it after.

    A hot journal left by a killed writer makes the read-only open fail with
    ``SQLITE_READONLY_ROLLBACK``; a read-write connection then opens once,
    rolls the journal back on its first read and closes, and the read-only
    open is repeated. A file still in WAL mode is switched to the rollback
    journal first when no other connection holds it, and read as it is
    otherwise.

    An open still busy after *busy_timeout* (a writer waiting to commit
    refuses new reads) is retried with backoff until it succeeds; warnings
    go to the busy callback of this process's open writer on the file, else
    to the log. The yielded connection holds a read transaction until it
    closes, so its queries see one state and are never refused as busy.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No database at {path}")
    if _is_wal_file(path):
        with contextlib.suppress(sqlite3.OperationalError):
            _recover(path, 0.0)
    busy = _busy_wait(path)
    started = time.monotonic()
    delay = RETRY_DELAY_MIN
    while True:
        try:
            connection = _open_recovering(path, busy_timeout)
            break
        except sqlite3.OperationalError as exc:
            if not is_busy(exc):
                raise
        delay = busy.wait(started, delay)
    try:
        yield connection
    finally:
        connection.close()


def read(
    path: Path,
    queries: Callable[[sqlite3.Connection], T],
    *,
    busy_timeout: float = DEFAULT_BUSY_TIMEOUT,
) -> T:
    """Run *queries* on a short read-only connection and return their result.

    The queries run in one read transaction; a busy database delays the open
    without failing it (see :func:`read_only`).
    """
    with read_only(path, busy_timeout=busy_timeout) as connection:
        return queries(connection)


def table_names(path: Path) -> list[str]:
    """Return the names of the tables in the database at *path*, sorted."""
    rows = read(
        path,
        lambda connection: connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall(),
    )
    return [row[0] for row in rows]


def read_rows(
    path: Path, table: str, *, order_by: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Return the rows of *table* as dicts, ordered by *order_by* or rowid."""
    order = ", ".join(_quote(column) for column in order_by) if order_by else "rowid"
    sql = f"SELECT * FROM {_quote(table)} ORDER BY {order}"

    def rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
        cursor = connection.execute(sql)
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    return read(path, rows)
