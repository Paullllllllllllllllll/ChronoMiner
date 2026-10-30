"""The classifications store component and the migration refusal under leases."""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.classifiers import Classifier, parse_classifier
from chronominer.common.classify_store import (
    COMPONENT,
    ChangedInputError,
    DocumentChangedError,
    RowContext,
    answer_rows,
    mark_stale_for_document,
    mark_stale_for_requests,
    record_answers,
    republish_answers,
    reusable_answers,
)
from chronominer.common.decision_runner import (
    Request,
    RequestResult,
    Unit,
    UnitAnswer,
    UnitAnswers,
)
from chronominer.common.decisions import DecisionUsage, Text
from chronominer.common.sqlite import read, read_rows
from chronominer.common.store import (
    LEASE_TTL,
    LeaseLostError,
    MigrationBlockedError,
    Store,
    check_schema,
)
from tests.common.store_child import FakeClock


def _classifier(unit: str = "page", questions: tuple[str, ...] = ("title",)) -> Any:
    body: dict[str, Any] = {
        "name": "front",
        "version": 1,
        "unit": unit,
        "per_request": 2,
        "questions": [
            {"name": name, "type": "predicate", "instructions": f"{name}?"}
            for name in questions
        ]
        + [
            {
                "name": "script",
                "type": "choice",
                "instructions": "Which script?",
                "choices": {"roman": None, "blackletter": None},
            }
        ],
    }
    if unit == "entry":
        body.update(schema="recipes", fields=["title"])
    return parse_classifier(body)


CLASSIFIER: Classifier = _classifier()


def _result(
    texts: tuple[str, ...],
    probability: float = 0.8,
    *,
    start: int = 0,
    document: str = "book",
    classifier: Classifier = CLASSIFIER,
) -> RequestResult:
    units = tuple(
        Unit(f"page {n}", n, (Text(t),)) for n, t in enumerate(texts, start=start)
    )
    request = Request(classifier, document, units)
    answers = tuple(
        UnitAnswers(
            unit,
            tuple(
                UnitAnswer(q.name, "predicate", probability=probability)
                if q.type == "predicate"
                else UnitAnswer(
                    q.name, "choice", "roman", 0.7, {"roman": 0.7, "blackletter": 0.3}
                )
                for q in classifier.questions
            ),
        )
        for unit in units
    )
    return RequestResult(request, answers, "jev", "openrouter", DecisionUsage())


async def _open(db: Path, clock: FakeClock) -> tuple[Store, str]:
    store = Store(db, "tool", components=(COMPONENT,), clock=clock)
    await store.open()
    return store, await store.begin_run("classify")


async def _record(
    store: Store,
    result: RequestResult,
    ctx: RowContext,
    classifier: Classifier = CLASSIFIER,
) -> RequestResult:
    rows = answer_rows(result, classifier, ctx, store.clock())
    await record_answers(store, rows)
    return result


def _view(db: Path) -> list[dict[str, Any]]:
    return read_rows(
        db, "classifications", order_by=("document", "locator", "question")
    )


async def test_page_rows_are_written_without_a_lease_or_document(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    store, run_id = await _open(db, FakeClock())
    try:
        await _record(
            store, _result(("a", "b")), RowContext("book", "fp", run_id=run_id)
        )
        with pytest.raises(ValueError, match="carry no doc_id"):
            answer_rows(_result(("a",)), CLASSIFIER, RowContext("book", "fp", 1), 0.0)
        with pytest.raises(ValueError, match="the request 'book'"):
            answer_rows(_result(("a",)), CLASSIFIER, RowContext("other", "fp"), 0.0)
    finally:
        await store.close()

    state = check_schema(db, (COMPONENT,))
    assert state.versions["classifications"] == 1
    assert read_rows(db, "documents") == []
    assert len(_view(db)) == 4
    assert {row["doc_id"] for row in _view(db)} == {None}


async def test_entry_rows_need_a_live_lease_on_their_document(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()
    entry = _classifier("entry")
    store, run_id = await _open(db, clock)
    try:
        doc_id = await store.ensure_document("book")
        ctx = RowContext("book", "fp", doc_id, run_id=run_id)
        rows = answer_rows(_result(("a",), classifier=entry), entry, ctx, clock())
        with pytest.raises(ValueError, match="need the document's doc_id"):
            answer_rows(
                _result(("a",), classifier=entry), entry, RowContext("book", "f"), 0
            )
        with pytest.raises(ValueError, match="need its lease"):
            await record_answers(store, rows)
        lease = await store.acquire(doc_id)
        assert lease is not None
        assert await record_answers(store, rows, lease=lease) == 2

        clock.advance(LEASE_TTL + 1)
        with pytest.raises(LeaseLostError):
            await record_answers(store, rows, lease=lease)
        assert len(read_rows(db, "classification_rows")) == 2
    finally:
        await store.close()


async def test_the_view_shows_the_latest_row_that_is_not_stale(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()
    store, run_id = await _open(db, clock)
    ctx = RowContext("book", "fp", run_id=run_id)
    try:
        first = await _record(store, _result(("a", "b"), 0.8), ctx)
        clock.advance(1)
        second = await _record(store, _result(("a", "b"), 0.6), ctx)

        rows = _view(db)
        assert len(rows) == 4
        assert {r["probability"] for r in rows if r["kind"] == "predicate"} == {0.6}
        script = next(row for row in rows if row["question"] == "script")
        assert script["value"] == "roman"
        assert script["probabilities"] == '{"blackletter": 0.3, "roman": 0.7}'
        assert script["classifier_sha256"] == CLASSIFIER.sha256

        assert first.request.sha256 == second.request.sha256
        stale = await store.write_unleased(
            lambda c: mark_stale_for_requests(c, "book", [second.request.sha256])
        )
        assert stale == 8
        assert _view(db) == []
    finally:
        await store.close()


async def test_stale_updates_stay_inside_one_document(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    store, run_id = await _open(db, FakeClock())
    try:
        blank = ("blank", "blank")
        await _record(store, _result(blank), RowContext("book", "fp", run_id=run_id))
        other = await _record(
            store, _result(blank, document="other"), RowContext("other", "fp")
        )
        later = await _record(
            store, _result(("c", "d"), start=2), RowContext("book", "fp")
        )

        flagged = await store.write_unleased(
            lambda c: mark_stale_for_document(c, "book", unit="page", ordinal=1)
        )

        assert flagged == 4
        rows = _view(db)
        assert {(r["document"], r["locator"]) for r in rows} == {
            ("book", "page 2"),
            ("book", "page 3"),
            ("other", "page 0"),
            ("other", "page 1"),
        }
        assert {r["request_sha256"] for r in rows} == {
            later.request.sha256,
            other.request.sha256,
        }
        with pytest.raises(ValueError, match="needs its unit kind"):
            await store.write_unleased(
                lambda c: mark_stale_for_document(c, "book", ordinal=1)
            )
    finally:
        await store.close()


async def test_a_classifier_revision_hides_the_old_revisions_answers(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    store, _ = await _open(db, FakeClock())
    revised = _classifier(questions=("dated",))
    try:
        await _record(store, _result(("a",)), RowContext("book", "fp"))
        await _record(
            store,
            _result(("a",), classifier=revised),
            RowContext("book", "fp"),
            revised,
        )

        rows = _view(db)

        assert {row["question"] for row in rows} == {"dated", "script"}
        assert {row["classifier_sha256"] for row in rows} == {revised.sha256}
    finally:
        await store.close()


async def test_rows_of_a_changed_input_leave_the_view(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    store, _ = await _open(db, FakeClock())
    try:
        await _record(store, _result(("a",)), RowContext("book", "fp", None, "sha-1"))
        await _record(
            store, _result(("a",), document="loose"), RowContext("loose", "fp")
        )
        await store.ensure_document("book", input_sha256="sha-1")
        assert {row["document"] for row in _view(db)} == {"book", "loose"}

        await store.ensure_document("book", input_sha256="sha-2")

        assert {row["document"] for row in _view(db)} == {"loose"}
    finally:
        await store.close()


async def test_reuse_needs_all_hashes_the_input_and_a_fresh_row(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    store, _ = await _open(db, FakeClock())
    ctx = RowContext("book", "fp1", input_sha256="sha-1")
    try:
        result = await _record(store, _result(("a", "b")), ctx)
        unit = result.units[0].unit
        key = {
            "unit": "page",
            "locator": unit.locator,
            "classifier": "front",
            "content_sha256": unit.content_sha256,
            "request_sha256": result.request.sha256,
        }

        def lookup(context: RowContext = ctx, **change: str) -> dict[str, Any]:
            return read(db, lambda c: reusable_answers(c, context, **{**key, **change}))

        found = lookup()
        assert set(found) == {"title", "script"}
        assert found["script"]["probabilities"] == {"roman": 0.7, "blackletter": 0.3}
        assert lookup(RowContext("book", "fp2", input_sha256="sha-1")) == {}
        assert lookup(RowContext("book", "fp1", input_sha256="sha-2")) == {}
        assert lookup(content_sha256="0" * 64) == {}
        assert lookup(request_sha256="0" * 64) == {}

        await store.write_unleased(lambda c: mark_stale_for_document(c, "book"))
        assert lookup() == {}
    finally:
        await store.close()


async def _seed_book(store: Store, run_id: str) -> int:
    """Store document ``book`` with page and entry rows, and ``kept`` pages."""
    entry = _classifier("entry")
    doc_id = await store.ensure_document("book")
    await _record(store, _result(("a",)), RowContext("book", "fp"))
    await _record(store, _result(("a",), document="kept"), RowContext("kept", "fp"))
    lease = await store.acquire(doc_id)
    assert lease is not None
    rows = answer_rows(
        _result(("a",), classifier=entry),
        entry,
        RowContext("book", "fp", doc_id, run_id=run_id),
        0.0,
    )
    await record_answers(store, rows, lease=lease)
    await store.release(lease)
    return doc_id


async def test_a_drop_deletes_the_rows_by_name_and_doc_id(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    store, run_id = await _open(db, FakeClock())
    try:
        doc_id = await _seed_book(store, run_id)

        report = await store.drop([doc_id])

        assert report.dropped == ("book",)
        left = read_rows(db, "classification_rows")
        assert {row["document"] for row in left} == {"kept"}
    finally:
        await store.close()


async def test_an_old_writers_plain_delete_takes_the_rows_along(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    store, run_id = await _open(db, FakeClock())
    try:
        doc_id = await _seed_book(store, run_id)
    finally:
        await store.close()

    with contextlib.closing(sqlite3.connect(db)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        connection.commit()

    left = read_rows(db, "classification_rows")
    assert {row["document"] for row in left} == {"kept"}


async def test_unleased_rows_of_a_changed_stored_input_are_refused(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    store, _ = await _open(db, FakeClock())
    try:
        await store.ensure_document("book", input_sha256="sha-a")

        with pytest.raises(ChangedInputError, match="process the changed file"):
            await _record(
                store, _result(("a",)), RowContext("book", "fp", None, "sha-b")
            )
        assert read_rows(db, "classification_rows") == []

        await _record(store, _result(("a",)), RowContext("book", "fp", None, "sha-a"))
        assert len(_view(db)) == 2
    finally:
        await store.close()


async def test_reused_answers_are_republished_after_a_b_a(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()
    store, run_id = await _open(db, clock)
    a, b = RowContext("book", "fp-a"), RowContext("book", "fp-b")
    try:
        first = await _record(store, _result(("x",), 0.8), a)
        clock.advance(1)
        await _record(store, _result(("x",), 0.6), b)
        unit = first.units[0].unit
        found = read(
            db,
            lambda c: reusable_answers(
                c,
                a,
                unit="page",
                locator=unit.locator,
                classifier="front",
                content_sha256=unit.content_sha256,
                request_sha256=first.request.sha256,
            ),
        )
        assert found["title"]["probability"] == 0.8
        assert {r["fingerprint"] for r in _view(db)} == {"fp-b"}

        clock.advance(1)
        assert await republish_answers(store, found, run_id=run_id) == 2

        rows = _view(db)
        assert {r["fingerprint"] for r in rows} == {"fp-a"}
        assert {r["reused"] for r in rows} == {1}
        assert {r["run_id"] for r in rows} == {run_id}
        title = next(r for r in rows if r["question"] == "title")
        assert title["probability"] == 0.8
    finally:
        await store.close()


async def test_answers_of_another_input_version_go_stale(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    store, _ = await _open(db, FakeClock())
    try:
        await _record(
            store, _result(("a", "b", "c")), RowContext("book", "fp", None, "sha-3")
        )
        await _record(
            store, _result(("a",), document="other"), RowContext("other", "fp")
        )
        await _record(
            store, _result(("a", "b")), RowContext("book", "fp", None, "sha-2")
        )

        rows = _view(db)
        assert {r["locator"] for r in rows if r["document"] == "book"} == {
            "page 0",
            "page 1",
        }
        assert {r["input_sha256"] for r in rows if r["document"] == "book"} == {"sha-2"}
        assert {r["locator"] for r in rows if r["document"] == "other"} == {"page 0"}
    finally:
        await store.close()


async def test_answers_of_another_input_go_stale_across_unit_kinds(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    document = _classifier("document")
    store, _ = await _open(db, FakeClock())
    try:
        await _record(
            store,
            _result(("a",), classifier=document),
            RowContext("book", "fp", None, "sha-a"),
            document,
        )
        assert {r["unit"] for r in _view(db)} == {"document"}

        await _record(store, _result(("a",)), RowContext("book", "fp", None, "sha-b"))

        rows = _view(db)
        assert {(r["unit"], r["input_sha256"]) for r in rows} == {("page", "sha-b")}
    finally:
        await store.close()


async def test_a_document_dropped_since_planning_refuses_the_write(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()
    store, run_id = await _open(db, clock)
    try:
        doc_id = await store.ensure_document("book", input_sha256="sha-a")
        ctx = RowContext("book", "fp", None, "sha-a", run_id, seen_doc_id=doc_id)
        first = _result(("a",))
        await record_answers(
            store, answer_rows(first, CLASSIFIER, ctx, clock()), seen_doc_id=doc_id
        )
        unit = first.units[0].unit
        found = read(
            db,
            lambda c: reusable_answers(
                c,
                ctx,
                unit="page",
                locator=unit.locator,
                classifier="front",
                content_sha256=unit.content_sha256,
                request_sha256=first.request.sha256,
            ),
        )
        await store.drop([doc_id])
        rows = answer_rows(_result(("a",)), CLASSIFIER, ctx, clock())

        with pytest.raises(DocumentChangedError, match="dropped or replaced"):
            await record_answers(store, rows, seen_doc_id=doc_id)
        with pytest.raises(DocumentChangedError):
            await republish_answers(store, found, run_id=run_id, seen_doc_id=doc_id)
        assert read_rows(db, "classification_rows") == []

        await store.ensure_document("book", input_sha256="sha-a")
        with pytest.raises(DocumentChangedError):
            await record_answers(store, rows, seen_doc_id=doc_id)
        assert await record_answers(store, rows) == 2
    finally:
        await store.close()


async def test_republished_answers_take_the_current_place(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()
    store, run_id = await _open(db, clock)
    ctx = RowContext("book", "fp")
    try:
        first = await _record(store, _result(("x",)), ctx)
        unit = first.units[0].unit
        found = read(
            db,
            lambda c: reusable_answers(
                c,
                ctx,
                unit="page",
                locator=unit.locator,
                classifier="front",
                content_sha256=unit.content_sha256,
                request_sha256=first.request.sha256,
            ),
        )

        await republish_answers(
            store, found, run_id=run_id, locator="page 5", ordinal=5
        )

        latest = read_rows(db, "classification_rows", order_by=("row_id",))[-2:]
        assert {(r["locator"], r["ordinal"], r["reused"]) for r in latest} == {
            ("page 5", 5, 1)
        }
        assert {(r["locator"], r["ordinal"], r["reused"]) for r in _view(db)} == {
            ("page 5", 5, 1)
        }
    finally:
        await store.close()


async def test_a_migration_waits_for_a_live_lease_of_another_run(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()
    async with Store(db, "tool", clock=clock) as holder:
        await holder.begin_run("transcribe")
        doc_id = await holder.ensure_document("book")
        assert await holder.acquire(doc_id) is not None
        before = check_schema(db).versions

        newer = Store(db, "tool", components=(COMPONENT,), clock=clock)
        with pytest.raises(MigrationBlockedError, match="let it finish") as raised:
            await newer.open()
        assert raised.value.exit_code == 2
        assert "'classifications' 0 to 1" in str(raised.value)
        assert check_schema(db).versions == before

        same = Store(db, "tool", clock=clock)
        await same.open()
        await same.close()

        clock.advance(LEASE_TTL + 1)
        async with Store(db, "tool", components=(COMPONENT,), clock=clock):
            pass
    assert check_schema(db, (COMPONENT,)).versions["classifications"] == 1
