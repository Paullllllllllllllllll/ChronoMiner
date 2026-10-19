"""The store schema, run stamping, documents, units and responses.

Run on Windows with:

    uv --directory <repo> run pytest tests/common/test_store.py -q
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.capture import AttemptRecord, use_recorder
from chronominer.common.capture import record_attempt as capture_record
from chronominer.common.sqlite import DatabaseFullError, read, read_rows
from chronominer.common.store import (
    CORE_TABLES,
    CORE_VERSION,
    UNIT_COUNTS_SQL,
    PlannedUnit,
    ResponseRow,
    Store,
    UnitKey,
    UnitResult,
    WriterOptions,
    current_unit,
    unit_context,
)

SPEC = {"model.name": {"value": "model-a", "source": "cli"}}


def _done(idx: int, content: str = "text", **changes: Any) -> UnitResult:
    values: dict[str, Any] = {
        "kind": "page",
        "idx": idx,
        "status": "done",
        "model": "model-a",
        "provider": "openai",
        "content": content,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    values.update(changes)
    return UnitResult(**values)


async def _planned(store: Store, name: str = "doc", pages: int = 3) -> Any:
    doc_id = await store.ensure_document(name, input_path=f"C:/in/{name}.pdf")
    lease = await store.acquire(doc_id)
    assert lease is not None
    units = [PlannedUnit(idx, label=f"page_{idx:04d}.png") for idx in range(pages)]
    await store.plan_units(lease, "page", units, model="model-a", provider="openai")
    return lease


def _schema(db: Path) -> dict[str, Any]:
    def query(connection: sqlite3.Connection) -> dict[str, Any]:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        ]
        keys = {
            table: [
                row[1]
                for row in sorted(
                    connection.execute(f"PRAGMA table_info({table})"),
                    key=lambda r: r[5],
                )
                if row[5]
            ]
            for table in tables
        }
        indexes = {
            row[0]: [
                info[2] for info in connection.execute(f"PRAGMA index_info({row[0]})")
            ]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        }
        return {"tables": tables, "keys": keys, "indexes": indexes}

    return read(db, query)


def test_the_core_tables_have_the_documented_keys_and_indexes(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool"):
            pass

    asyncio.run(main())
    schema = _schema(db)

    assert sorted(schema["tables"]) == sorted(CORE_TABLES)
    assert schema["keys"]["meta"] == ["component"]
    assert schema["keys"]["runs"] == ["run_id"]
    assert schema["keys"]["artifacts"] == ["sha256"]
    assert schema["keys"]["documents"] == ["doc_id"]
    assert schema["keys"]["units"] == ["doc_id", "kind", "idx"]
    assert schema["keys"]["responses"] == ["response_id"]
    assert schema["keys"]["submissions"] == ["submission_id"]
    assert schema["keys"]["submission_requests"] == ["submission_id", "custom_id"]
    assert schema["keys"]["outputs"] == ["doc_id", "path"]
    assert schema["keys"]["notes"] == ["note_id"]
    assert schema["indexes"]["units_by_status"] == ["doc_id", "status"]
    assert schema["indexes"]["responses_by_unit"] == ["doc_id", "kind", "idx"]
    assert read(db, lambda c: c.execute("SELECT * FROM meta").fetchall()) == [
        ("core", CORE_VERSION)
    ]
    unique_names = read(
        db,
        lambda c: [
            row[2] for row in c.execute("PRAGMA index_list(documents)") if row[2]
        ],
    )
    assert unique_names == [1]


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO documents (name, tool, status) VALUES ('d', 't', 'bogus')",
        "INSERT INTO meta (component, version) VALUES ('x', 0)",
        "INSERT INTO notes (level, message, created_at) VALUES ('loud', 'm', 'now')",
    ],
)
def test_check_constraints_reject_invalid_values(
    tmp_path: Path, statement: str
) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool") as store:
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                await store._write(lambda c: c.execute(statement))

    asyncio.run(main())


def test_an_invalid_unit_status_raises_and_leaves_the_unit_pending(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            lease = await _planned(store)
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                await store.write_unit(lease, _done(0, status="finished"))
            unit = store.units(lease.doc_id)[0]
            assert unit.status == "pending"

    asyncio.run(main())


def test_foreign_keys_are_on_in_the_writer(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            pragmas = await store._write(
                lambda c: tuple(
                    c.execute(f"PRAGMA {name}").fetchone()[0]
                    for name in ("foreign_keys", "journal_mode", "synchronous")
                )
            )
            assert pragmas == (1, "delete", 2)
            run_id = store.run_id
            with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
                await store._write(
                    lambda c: c.execute(
                        "INSERT INTO units (doc_id, kind, idx, tool, model, provider, "
                        "run_id) VALUES (999, 'page', 0, 't', 'm', 'p', ?)",
                        (run_id,),
                    )
                )

    asyncio.run(main())


def test_the_resume_count_query_uses_the_unit_status_index(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool"):
            pass

    asyncio.run(main())
    plan = read(
        db,
        lambda c: " ".join(
            str(row[3])
            for row in c.execute("EXPLAIN QUERY PLAN " + UNIT_COUNTS_SQL, (1,))
        ),
    )
    assert "units_by_status" in plan


def test_a_run_stamps_every_row_and_records_its_end(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool", tool_version="9.9.9") as store:
            run_id = await store.begin_run(
                "tool transcribe --input x", spec=SPEC, settings_fingerprint="fp"
            )
            first = await store.store_artifact("prompt", "Transcribe the page.")
            again = await store.store_artifact("prompt", "Transcribe the page.")
            assert first == again
            lease = await _planned(store)
            unit = UnitKey(lease.doc_id, "page", 0)
            response_id = await store.append_response(
                ResponseRow(
                    provider="openai",
                    model="model-a",
                    unit=unit,
                    raw='{"id": "r1"}',
                    raw_kind="http",
                )
            )
            await store.write_unit(lease, _done(0, response_id=response_id))
            await store.end_run(0)
            assert store.run(run_id) is not None

    asyncio.run(main())

    runs = read_rows(db, "runs")
    assert len(runs) == 1
    run = runs[0]
    assert run["tool"] == "tool"
    assert run["tool_version"] == "9.9.9"
    assert run["command"] == "tool transcribe --input x"
    assert '"source": "cli"' in run["spec"]
    assert run["settings_fingerprint"] == "fp"
    assert run["exit_code"] == 0
    assert run["ended_at"] is not None
    assert len(read_rows(db, "artifacts")) == 1
    units = read_rows(db, "units")
    responses = read_rows(db, "responses")
    assert len(units) == 3
    assert len(responses) == 1
    for row in [*units, *responses]:
        assert row["run_id"] == run["run_id"]
        assert row["tool"] == "tool"
        assert row["model"] == "model-a"
        assert row["provider"] == "openai"
    done = [row for row in units if row["status"] == "done"]
    assert done[0]["response_id"] == responses[0]["response_id"]
    assert done[0]["label"] == "page_0000.png"
    assert done[0]["attempts"] == 1


def test_document_status_and_counts_follow_the_units(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> list[tuple[str, int, int, int, int]]:
        seen = []
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            lease = await _planned(store)

            def state() -> tuple[str, int, int, int, int]:
                doc = store.document(lease.doc_id)
                assert doc is not None
                return (
                    doc.status,
                    doc.units_total,
                    doc.units_pending,
                    doc.units_done,
                    doc.units_failed,
                )

            seen.append(state())
            await store.write_unit(lease, _done(0))
            seen.append(state())
            await store.write_unit(lease, _done(1))
            await store.write_unit(
                lease, _done(2, status="failed", content=None, error="boom")
            )
            seen.append(state())
            await store.write_unit(lease, _done(2))
            seen.append(state())
            assert await store.reset_units(lease, idxs=[1, 2]) == 2
            seen.append(state())
            replanned = await store.plan_units(
                lease, "page", [PlannedUnit(0), PlannedUnit(3)], model="m", provider="p"
            )
            assert replanned == 1
            seen.append(state())
        return seen

    assert asyncio.run(main()) == [
        ("pending", 3, 3, 0, 0),
        ("partial", 3, 2, 1, 0),
        ("failed", 3, 0, 2, 1),
        ("complete", 3, 0, 3, 0),
        ("partial", 3, 2, 1, 0),
        ("partial", 4, 3, 1, 0),
    ]


def test_reset_keeps_the_responses_and_counts_attempts(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            lease = await _planned(store, pages=1)
            unit = UnitKey(lease.doc_id, "page", 0)
            for attempt in (1, 2):
                await store.append_response(
                    ResponseRow(
                        provider="openai", model="model-a", unit=unit, attempt=attempt
                    )
                )
            await store.write_unit(lease, _done(0, status="failed", error="x"))
            await store.write_unit(lease, _done(0))
            assert store.units(lease.doc_id)[0].attempts == 2
            await store.reset_units(lease, statuses=["done"])
            row = store.units(lease.doc_id)[0]
            assert (row.status, row.content, row.attempts) == ("pending", None, 0)
            assert len(store.responses(unit=unit)) == 2
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                await store._write(lambda c: c.execute("DELETE FROM responses"))

    asyncio.run(main())


def test_resume_state_reports_counts_lists_and_image_settings(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            lease = await _planned(store, pages=4)
            await store.set_image_settings(lease, {"dpi": 300, "detail": "high"})
            await store.write_unit(lease, _done(0))
            await store.write_unit(lease, _done(1, status="failed", error="x"))
            state = store.resume_state("doc")
            assert state is not None
            assert state.status == "partial"
            assert state.counts == {"done": 1, "failed": 1, "pending": 2}
            assert state.pending == (
                UnitKey(lease.doc_id, "page", 2),
                UnitKey(lease.doc_id, "page", 3),
            )
            assert state.failed == (UnitKey(lease.doc_id, "page", 1),)
            assert state.blocked == ()
            assert state.ready == state.pending
            assert state.image_settings == {"dpi": 300, "detail": "high"}
            assert state.lease_run_id == store.run_id
            assert store.resume_state("missing") is None
            failed = store.failed_units()
            assert [unit.key for unit in failed] == [UnitKey(lease.doc_id, "page", 1)]

    asyncio.run(main())


def test_a_rename_is_one_update_that_keeps_the_rows(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            lease = await _planned(store, "old")
            await store.write_unit(lease, _done(0))
            doc_id = await store.rename_document("old", "new", input_path="C:/in/n.pdf")
            assert doc_id == lease.doc_id
            assert store.document("old") is None
            renamed = store.document("new")
            assert renamed is not None
            assert renamed.input_path == "C:/in/n.pdf"
            assert len(store.units(doc_id)) == 3
            assert await store.ensure_document("new") == doc_id
            with pytest.raises(KeyError):
                await store.rename_document("old", "other")

    asyncio.run(main())


def test_writes_need_a_run(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool") as store:
            with pytest.raises(RuntimeError, match="Begin a run"):
                await store.append_response(ResponseRow(provider="p", model="m"))

    asyncio.run(main())


def test_notes_are_kept_per_document(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            doc_id = await store.ensure_document("doc")
            await store.add_note("model changed", doc_id=doc_id)
            await store.add_note("run note", level="info")
            assert [note["message"] for note in store.notes(doc_id=doc_id)] == [
                "model changed"
            ]
            assert len(store.notes()) == 2

    asyncio.run(main())


def test_the_recorder_attributes_attempts_to_the_current_unit(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    body = b'{"id": "resp_1", "output_text": "caf\xc3\xa9"}'
    image = "data:image/jpeg;base64," + "A" * 2000

    async def main() -> list[int | None]:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            lease = await _planned(store, pages=2)
            unit = UnitKey(lease.doc_id, "page", 1)
            ids = []
            with use_recorder(store.recorder()):
                assert current_unit() is None
                with unit_context(unit):
                    assert current_unit() == unit
                    ids.append(
                        await capture_record(
                            AttemptRecord(
                                provider="openai",
                                model="model-a",
                                role="transcribe",
                                attempt=1,
                                request={"input": [{"image_url": image}]},
                                raw=body.decode("utf-8"),
                                raw_kind="http",
                                http_status=200,
                                latency=0.25,
                            )
                        )
                    )
                assert current_unit() is None
            return ids

    ids = asyncio.run(main())

    rows = read_rows(db, "responses")
    assert [row["response_id"] for row in rows] == ids
    row = rows[0]
    assert (row["doc_id"], row["kind"], row["idx"]) == (1, "page", 1)
    assert row["raw"].encode("utf-8") == body
    assert row["raw_kind"] == "http"
    assert row["http_status"] == 200
    assert "base64" in row["request"]
    assert "A" * 100 not in row["request"]


def test_bytes_that_are_not_utf8_are_stored_as_a_blob(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    body = b"\xff\xfe binary"

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            await store.append_response(
                ResponseRow(provider="p", model="m", raw=body, raw_kind="line")
            )

    asyncio.run(main())
    assert read_rows(db, "responses")[0]["raw"] == body


def test_a_reader_holding_a_transaction_delays_a_unit_write_without_failing(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    warnings: list[float] = []

    async def main() -> float:
        loop = asyncio.get_running_loop()
        options = WriterOptions(
            busy_timeout=0.2,
            warn_interval=0.5,
            on_busy=lambda _path, waited: warnings.append(waited),
        )
        async with Store(db, "tool", writer=options) as store:
            await store.begin_run("run")
            lease = await _planned(store, pages=1)
            reader = sqlite3.connect(db, isolation_level=None)
            reader.execute("BEGIN")
            reader.execute("SELECT count(*) FROM units").fetchone()
            loop.call_later(2.0, lambda: (reader.execute("COMMIT"), reader.close()))
            started = time.monotonic()
            await store.write_unit(lease, _done(0))
            return time.monotonic() - started

    waited = asyncio.run(main())

    assert waited >= 1.5
    assert warnings
    assert [row["status"] for row in read_rows(db, "units")] == ["done"]


def test_a_full_database_fails_the_write_loudly_and_keeps_earlier_rows(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            lease = await _planned(store, pages=1)
            await store.write_unit(lease, _done(0, content="kept"))
            await store._write(
                lambda c: c.execute(
                    "PRAGMA max_page_count = "
                    f"{c.execute('PRAGMA page_count').fetchone()[0] + 1}"
                )
            )
            with pytest.raises(DatabaseFullError):
                await store.append_response(
                    ResponseRow(
                        provider="p", model="m", raw="x" * 200_000, raw_kind="http"
                    )
                )

    asyncio.run(main())

    assert [row["content"] for row in read_rows(db, "units")] == ["kept"]
    assert read_rows(db, "responses") == []
    assert read(db, lambda c: c.execute("PRAGMA integrity_check").fetchone()[0]) == (
        "ok"
    )
