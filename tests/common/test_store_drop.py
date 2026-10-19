"""Dropping documents: rows, rendered files, refusals and document selection.

Run on Windows with:

    uv --directory <repo> run pytest tests/common/test_store_drop.py -q
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from chronominer.common.sqlite import read
from chronominer.common.store import (
    LEASE_TTL,
    Component,
    DocumentRow,
    DocumentSelectionError,
    DropReport,
    KeptFile,
    Lease,
    PlannedUnit,
    RenderTarget,
    ResponseRow,
    Store,
    Submission,
    UnitKey,
    UnitResult,
    select_documents,
)
from tests.common.store_child import FakeClock

EXTRA = Component(
    "extra",
    (
        "CREATE TABLE extra_rows (doc_id INTEGER NOT NULL "
        "REFERENCES documents (doc_id), value TEXT NOT NULL)",
    ),
)
"""A tool component with a content table keyed by ``doc_id``."""

COUNTED = ("units", "responses", "outputs", "notes", "submission_requests")


def _text(connection: sqlite3.Connection, doc_id: int) -> str:
    rows = connection.execute(
        "SELECT content FROM units WHERE doc_id = ? ORDER BY idx", (doc_id,)
    ).fetchall()
    return "\n".join(str(row[0] or "") for row in rows) + "\n"


def _store(db: Path, clock: FakeClock | None = None) -> Store:
    return Store(db, "tool", components=(EXTRA,), clock=clock or FakeClock())


async def _document(store: Store, name: str, out: Path) -> Lease:
    """Store document *name* with two done pages, a response per page, a
    note, a content row and its txt and md files rendered into *out*."""
    doc_id = await store.ensure_document(name, input_path=f"C:/in/{name}.pdf")
    lease = await store.acquire(doc_id)
    assert lease is not None
    await store.plan_units(
        lease, "page", [PlannedUnit(0), PlannedUnit(1)], model="m", provider="p"
    )
    for idx in (0, 1):
        unit = UnitKey(doc_id, "page", idx)
        response_id = await store.append_response(
            ResponseRow(provider="p", model="m", unit=unit, raw="{}", raw_kind="http")
        )
        await store.write_unit(
            lease,
            UnitResult(
                kind="page",
                idx=idx,
                status="done",
                model="m",
                provider="p",
                content=f"{name} page {idx}",
                response_id=response_id,
            ),
        )
    await store.add_note("a note", doc_id=doc_id)
    await store.write(
        lease,
        lambda connection: connection.execute(
            "INSERT INTO extra_rows (doc_id, value) VALUES (?, 'x')", (doc_id,)
        ),
    )
    targets = [
        RenderTarget(out / f"{name}.txt", "txt", _text),
        RenderTarget(out / f"{name}.md", "md", _text),
    ]
    await store.render(lease, targets)
    return lease


def _counts(db: Path, doc_id: int) -> dict[str, int]:
    def query(connection: sqlite3.Connection) -> dict[str, int]:
        return {
            table: int(
                connection.execute(
                    f"SELECT count(*) FROM {table} WHERE doc_id = ?", (doc_id,)
                ).fetchone()[0]
            )
            for table in (*COUNTED, "extra_rows", "documents")
        }

    return read(db, query)


async def _create(submission: Submission) -> str:
    return f"batch-{submission.submission_id[:8]}"


async def _upload(submission: Submission) -> str | None:
    return None


async def _unconfirmed(submission: Submission) -> str:
    raise ConnectionError("no answer")


def test_drop_deletes_every_row_and_the_unchanged_files(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    out = tmp_path / "out"

    async def main() -> tuple[DropReport, int, int]:
        async with _store(db) as store:
            await store.begin_run("run")
            lease = await _document(store, "doc", out)
            kept = await _document(store, "other", out)
            units = [UnitKey(lease.doc_id, "page", idx) for idx in (0, 1)]
            submission = await store.submit(
                provider="p", model="m", units=units, upload=_upload, create=_create
            )
            await store.mark_finalized(submission.submission_id)
            await store.release(lease)
            await store.release(kept)
            (out / "doc.md").write_text("edited by hand\n", encoding="utf-8")
            report = await store.drop([lease.doc_id])
            return report, lease.doc_id, kept.doc_id

    report, doc_id, other_id = asyncio.run(main())

    assert report == DropReport(
        dry_run=False,
        dropped=("doc",),
        deleted_files=(out.resolve() / "doc.txt",),
        kept_files=(
            KeptFile(out.resolve() / "doc.md", "it was changed outside the tool"),
        ),
    )
    assert not (out / "doc.txt").exists()
    assert (out / "doc.md").read_text(encoding="utf-8") == "edited by hand\n"
    assert set(_counts(db, doc_id).values()) == {0}
    assert _counts(db, other_id) == {
        "units": 2,
        "responses": 2,
        "outputs": 2,
        "notes": 1,
        "submission_requests": 0,
        "extra_rows": 1,
        "documents": 1,
    }
    assert (out / "other.txt").exists() and (out / "other.md").exists()
    submissions = read(
        db,
        lambda connection: connection.execute(
            "SELECT status FROM submissions"
        ).fetchall(),
    )
    assert submissions == [("finalized",)]
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        con = sqlite3.connect(db)
        try:
            con.execute("DELETE FROM responses")
        finally:
            con.close()


def test_a_dropped_document_starts_from_scratch(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> tuple[int, int]:
        async with _store(db) as store:
            await store.begin_run("run")
            lease = await _document(store, "doc", tmp_path / "out")
            await store.drop([lease.doc_id])
            doc_id = await store.ensure_document("doc")
            state = store.resume_state("doc")
            assert state is not None and state.counts == {}
            return doc_id, len(store.responses(unit=UnitKey(doc_id, "page", 0)))

    doc_id, responses = asyncio.run(main())

    assert responses == 0
    assert _counts(db, doc_id)["documents"] == 1


def test_a_dry_run_reports_and_changes_nothing(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    out = tmp_path / "out"

    async def main() -> tuple[DropReport, int]:
        async with _store(db) as store:
            await store.begin_run("run")
            lease = await _document(store, "doc", out)
            await store.release(lease)
            (out / "doc.md").unlink()
            return await store.drop([lease.doc_id], dry_run=True), lease.doc_id

    report, doc_id = asyncio.run(main())

    assert report == DropReport(
        dry_run=True,
        dropped=("doc",),
        deleted_files=(out.resolve() / "doc.txt",),
        kept_files=(KeptFile(out.resolve() / "doc.md", "it is missing"),),
    )
    assert (out / "doc.txt").exists()
    assert _counts(db, doc_id)["units"] == 2
    assert report.as_dict()["kept_files"] == [
        {"path": str(out.resolve() / "doc.md"), "reason": "it is missing"}
    ]


def test_a_live_lease_of_another_run_refuses_the_drop(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()

    async def main() -> tuple[DropReport, DropReport, DropReport, int]:
        async with _store(db, clock) as worker, _store(db, clock) as dropper:
            await worker.begin_run("run")
            lease = await _document(worker, "doc", tmp_path / "out")
            refused = await dropper.drop([lease.doc_id])
            own = await worker.drop([lease.doc_id], dry_run=True)
            clock.advance(LEASE_TTL + 1)
            expired = await dropper.drop([lease.doc_id])
            return refused, own, expired, lease.doc_id

    refused, own, expired, doc_id = asyncio.run(main())

    assert refused.dropped == () and refused.deleted_files == ()
    assert [(item.name, item.reason[:4]) for item in refused.refused] == [
        ("doc", "run ")
    ]
    assert "leases it until" in refused.refused[0].reason
    assert own.dropped == ("doc",)
    assert expired.dropped == ("doc",)
    assert _counts(db, doc_id)["documents"] == 0


def test_units_held_by_an_unresolved_batch_refuse_the_drop(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> tuple[DropReport, str, int]:
        async with _store(db) as store:
            await store.begin_run("run")
            lease = await _document(store, "doc", tmp_path / "out")
            unit = UnitKey(lease.doc_id, "page", 0)
            with pytest.raises(ConnectionError):
                await store.submit(
                    provider="p",
                    model="m",
                    units=[unit],
                    upload=_upload,
                    create=_unconfirmed,
                )
            (submission,) = store.unresolved_submissions()
            await store.release(lease)
            return (
                await store.drop([lease.doc_id, 999]),
                submission.submission_id,
                lease.doc_id,
            )

    report, submission_id, doc_id = asyncio.run(main())

    assert report.dropped == ()
    assert [(item.doc_id, item.name, item.reason) for item in report.refused] == [
        (
            doc_id,
            "doc",
            f"batch submission {submission_id} (unresolved) holds its units",
        ),
        (999, None, "no such document in the store"),
    ]
    assert _counts(db, doc_id)["units"] == 2


def test_a_file_another_document_records_is_kept(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    shared = tmp_path / "out" / "combined.txt"

    async def main() -> DropReport:
        async with _store(db) as store:
            await store.begin_run("run")
            first = await _document(store, "a", tmp_path / "out")
            second = await _document(store, "b", tmp_path / "out")
            shared.parent.mkdir(parents=True, exist_ok=True)
            shared.write_text("both\n", encoding="utf-8")
            await store.register_output(first, shared, format="txt")
            await store.register_output(second, shared, format="txt")
            return await store.drop([first.doc_id])

    report = asyncio.run(main())

    assert report.dropped == ("a",)
    assert KeptFile(shared.resolve(), "a document that stays records it too") in (
        report.kept_files
    )
    assert shared.exists()


def _row(doc_id: int, name: str, input_path: str | None) -> DocumentRow:
    return DocumentRow(
        doc_id=doc_id,
        name=name,
        tool="tool",
        input_path=input_path,
        input_sha256=None,
        kind=None,
        status="complete",
        units_total=0,
        units_pending=0,
        units_done=0,
        units_failed=0,
        image_settings=None,
        first_run_id=None,
        last_run_id=None,
        lease_run_id=None,
        lease_until=None,
        lease_generation=0,
        updated_at=None,
    )


DOCUMENTS = [
    _row(3, "gamma", "C:/in/gamma.pdf"),
    _row(1, "alpha", "C:/in/alpha.pdf"),
    _row(2, "beta", None),
]


def _names(selection_documents: tuple[DocumentRow, ...]) -> list[str]:
    return [document.name for document in selection_documents]


def test_select_documents_by_all_number_name_and_file() -> None:
    assert _names(select_documents(DOCUMENTS, select_all=True).documents) == [
        "alpha",
        "beta",
        "gamma",
    ]
    numbered = select_documents(DOCUMENTS, select="1,3,7")
    assert _names(numbered.documents) == ["alpha", "gamma"]
    assert numbered.warnings == (
        "Ignored out-of-range or invalid --select part(s): 7 (valid: 1-3)",
    )
    assert _names(select_documents(DOCUMENTS, select="ET").documents) == ["beta"]
    files = select_documents(DOCUMENTS, files=["gamma.pdf", "beta"])
    assert _names(files.documents) == ["beta", "gamma"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one"),
        ({"select": "1", "select_all": True}, "exactly one"),
        ({"select": "zeta"}, "No stored document matches"),
        ({"select": ""}, "blank"),
        ({"select": "  "}, "blank"),
        ({"files": ["alpha.pdf", "beta.pdf"]}, "'beta.pdf'"),
    ],
)
def test_select_documents_refuses_bad_selections(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(DocumentSelectionError, match=message):
        select_documents(DOCUMENTS, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("extra", "names"),
    [
        (_row(4, "alpha.pdf", None), "alpha, alpha.pdf"),
        (_row(4, "delta", "D:/other/alpha.pdf"), "alpha, delta"),
    ],
)
def test_a_files_entry_that_names_two_documents_is_refused(
    extra: DocumentRow, names: str
) -> None:
    with pytest.raises(DocumentSelectionError, match=rf"names 2 documents \({names}\)"):
        select_documents([*DOCUMENTS, extra], files=["alpha.pdf"])


def test_a_new_document_never_takes_a_dropped_id(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> list[int]:
        async with _store(db) as store:
            await store.begin_run("run")
            first = await store.ensure_document("a")
            last = await store.ensure_document("b")
            await store.drop([last])
            after_drop = await store.ensure_document("c")
            await store.drop([first, after_drop])
        async with _store(db) as store:
            await store.begin_run("run")
            reopened = await store.ensure_document("d")
        return [first, last, after_drop, reopened]

    ids = asyncio.run(main())

    assert ids == [1, 2, 3, 4]
    mark = read(
        db,
        lambda connection: connection.execute(
            "SELECT version FROM meta WHERE component = 'dropped_doc_id'"
        ).fetchone(),
    )
    assert mark == (3,)


def test_a_late_response_for_a_dropped_document_fails_alone(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()

    async def main() -> tuple[int, int]:
        async with _store(db, clock) as worker, _store(db, clock) as dropper:
            await worker.begin_run("run")
            lease = await _document(worker, "doc", tmp_path / "out")
            clock.advance(LEASE_TTL + 1)
            assert (await dropper.drop([lease.doc_id])).dropped == ("doc",)
            unit = UnitKey(lease.doc_id, "page", 0)
            with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
                await worker.append_response(
                    ResponseRow(provider="p", model="m", unit=unit, raw_kind=None)
                )
            return lease.doc_id, await worker.ensure_document("next")

    doc_id, next_id = asyncio.run(main())

    assert next_id > doc_id
    assert _counts(db, doc_id)["responses"] == 0


LINKS = Component(
    "links",
    ("CREATE TABLE links (target INTEGER NOT NULL REFERENCES documents (doc_id))",),
)
"""A tool table that refers to documents without a ``doc_id`` column."""


def test_a_foreign_key_violation_stops_the_drop_before_any_file(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    out = tmp_path / "out"

    async def main() -> int:
        store = Store(db, "tool", components=(EXTRA, LINKS), clock=FakeClock())
        async with store:
            await store.begin_run("run")
            lease = await _document(store, "doc", out)
            await store.write(
                lease,
                lambda connection: connection.execute(
                    "INSERT INTO links (target) VALUES (?)", (lease.doc_id,)
                ),
            )
            with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
                await store.drop([lease.doc_id])
            await store.add_note("the writer still commits", doc_id=lease.doc_id)
            return lease.doc_id

    doc_id = asyncio.run(main())

    assert (out / "doc.txt").exists() and (out / "doc.md").exists()
    assert _counts(db, doc_id) == {
        "units": 2,
        "responses": 2,
        "outputs": 2,
        "notes": 2,
        "submission_requests": 0,
        "extra_rows": 1,
        "documents": 1,
    }
