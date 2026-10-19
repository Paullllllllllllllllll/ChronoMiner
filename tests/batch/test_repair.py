"""Repair selection on the store: indices, folders, files, leases, open batches,
changed inputs.

Run on Windows with:

    uv --directory <repo> run pytest tests/batch/test_repair.py -q

The documents are built by ``tests.batch.test_repair_store``; the plan reads
the store through read-only connections and writes nothing.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from chronominer import __main__ as entry
from chronominer.batch import repair as repair_module
from chronominer.batch.repair import RepairPlanError, plan_repair
from chronominer.common.store import UnitKey, file_sha256
from chronominer.output.store import open_store
from chronominer.settings import Settings
from tests.batch.test_repair_store import (
    FakeExtractor,
    rows,
    spec_for,
    store_document,
    tree,
    unit_rows,
)

OUTCOMES = ["failed", "ok", "failed", "ok", "failed"]


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def extractor(monkeypatch: pytest.MonkeyPatch) -> type[FakeExtractor]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(repair_module, "ChunkExtractor", FakeExtractor)
    FakeExtractor.texts = {}
    FakeExtractor.calls = []
    FakeExtractor.created = []
    FakeExtractor.context_images = []
    return FakeExtractor


@pytest.mark.usefixtures("key")
async def test_the_failed_chunks_are_selected(tmp_path: Path) -> None:
    stored = await store_document(tmp_path, OUTCOMES)

    plan = plan_repair(spec_for(stored.output), Settings())

    assert plan.to_repair and plan.failures == [("doc", 0), ("doc", 2), ("doc", 4)]


@pytest.mark.usefixtures("key")
async def test_indices_narrow_the_units(tmp_path: Path) -> None:
    stored = await store_document(tmp_path, OUTCOMES)

    plan = plan_repair(spec_for(stored.output, indices="3-4"), Settings())
    assert [idx for _name, idx in plan.failures] == [4]

    plan = plan_repair(spec_for(stored.output, indices="3"), Settings())
    assert not plan.to_repair
    assert plan.message == "None of the specified indices matched detected failures."


@pytest.mark.usefixtures("key")
async def test_a_folder_finds_the_databases_below_it(tmp_path: Path) -> None:
    first = await store_document(tmp_path / "a", ["failed"], folder="jobs/one")
    second = await store_document(tmp_path / "a", ["ok", "failed"], folder="jobs/b/c")

    plan = plan_repair(spec_for(tmp_path / "a" / "jobs"), Settings())

    assert {document.root for document in plan.documents} == {
        first.root,
        second.root,
    }
    assert sorted(document.failures for document in plan.documents) == [[0], [1]]
    database = plan_repair(spec_for(first.db), Settings())
    assert [document.root for document in database.documents] == [first.root]


@pytest.mark.usefixtures("key")
async def test_the_schema_filter_keeps_the_documents_of_that_schema(
    tmp_path: Path,
) -> None:
    stored = await store_document(tmp_path, ["failed"])

    kept = plan_repair(spec_for(stored.root, schema="wage_records"), Settings())
    other = plan_repair(spec_for(stored.root, schema="trade_records"), Settings())

    assert kept.to_repair
    assert not other.to_repair
    assert other.message == "No failed chunks detected matching your criteria."


@pytest.mark.usefixtures("key")
async def test_files_select_documents_by_name(tmp_path: Path) -> None:
    stored = await store_document(tmp_path, ["failed"])

    for name in ("doc", "doc.pdf", "doc_output.json", "doc_output.csv"):
        plan = plan_repair(spec_for(stored.root), Settings(), files=[name])
        assert plan.failures == [("doc", 0)]
    with pytest.raises(RepairPlanError, match="Specified files not found") as caught:
        plan_repair(spec_for(stored.root), Settings(), files=["missing.pdf"])
    assert caught.value.exit_code == 2


async def test_a_clean_document_plans_nothing_and_needs_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    stored = await store_document(tmp_path, ["ok", "ok"])

    plan = plan_repair(spec_for(stored.output), Settings())

    assert not plan.to_repair
    assert plan.message == "No failed chunks detected matching your criteria."
    assert plan.outcome().selected == 0


def _hold_lease(db: Path, doc_id: int, holder: str = "other-run") -> None:
    connection = sqlite3.connect(db)
    try:
        with connection:
            connection.execute(
                "UPDATE documents SET lease_run_id = ?, lease_until = ? "
                "WHERE doc_id = ?",
                (holder, time.time() + 600, doc_id),
            )
    finally:
        connection.close()


async def test_a_document_leased_by_another_run_is_refused_with_a_note(
    tmp_path: Path, extractor: type[FakeExtractor]
) -> None:
    stored = await store_document(tmp_path, ["ok", "failed"])
    _hold_lease(stored.db, stored.doc_id)
    before = unit_rows(stored.db)
    text = stored.output.read_bytes()

    plan = plan_repair(spec_for(stored.output), Settings())
    assert any("leased by run other-run" in warning for warning in plan.warnings)
    result = await repair_module.run_repair(plan)

    assert (result.repaired, result.failed, result.selected) == (0, 1, 1)
    assert result.message is not None and "another run holds" in result.message
    (note,) = rows(stored.db, "SELECT doc_id, message FROM notes")
    assert note[0] == stored.doc_id and "another run holds" in note[1]
    assert unit_rows(stored.db) == before
    assert stored.output.read_bytes() == text
    assert extractor.calls == []


@pytest.mark.usefixtures("key")
@pytest.mark.parametrize("state", ["submitting", "submitted", "unresolved"])
async def test_units_of_an_open_or_unresolved_submission_are_refused(
    tmp_path: Path, state: str
) -> None:
    stored = await store_document(tmp_path, ["failed", "ok", "failed"])
    async with open_store(stored.root) as store:
        await store.begin_run("extract")
        lease = await store.acquire(stored.doc_id)
        assert lease is not None
        submission = await store.open_submission(
            provider="openai",
            model="gpt-5.6-luna",
            units=[UnitKey(stored.doc_id, "chunk", 0)],
        )
        if state == "submitted":
            await store.record_batch(submission.submission_id, "batch-1")
        elif state == "unresolved":
            await store.mark_unresolved(submission.submission_id, "lost")

    plan = plan_repair(spec_for(stored.output), Settings())
    assert plan.failures == [("doc", 2)]
    assert any("batch check" in warning for warning in plan.warnings)

    plan = plan_repair(spec_for(stored.output, indices="0"), Settings())
    assert not plan.to_repair
    assert plan.message is not None and "batch check" in plan.message


async def test_a_text_whose_input_changed_is_not_repaired_and_exits_1(
    tmp_path: Path, extractor: type[FakeExtractor]
) -> None:
    stored = await store_document(tmp_path, ["ok", "failed", "ok"], kind="text")
    connection = sqlite3.connect(stored.db)
    try:
        with connection:
            connection.execute(
                "UPDATE documents SET input_sha256 = ? WHERE doc_id = ?",
                (file_sha256(stored.source), stored.doc_id),
            )
    finally:
        connection.close()
    text = stored.source.read_text(encoding="utf-8")
    stored.source.write_text("Inserted\nlines\n" + text, encoding="utf-8")
    before = unit_rows(stored.db)

    plan = plan_repair(spec_for(stored.output), Settings())

    assert not plan.to_repair
    (warning,) = plan.warnings
    assert "'doc'" in warning and "extract --force" in warning
    result = await repair_module.run_repair(plan)
    assert (result.repaired, result.failed) == (0, 1)
    assert extractor.calls == []
    assert unit_rows(stored.db) == before
    assert entry.main(["repair", "--input", str(stored.output)]) == 1


def _snapshot(root: Path) -> tuple[list[str], dict[str, bytes]]:
    files = {
        name: (root / name).read_bytes()
        for name in tree(root)
        if (root / name).is_file()
    }
    return tree(root), files


@pytest.mark.usefixtures("key")
@pytest.mark.parametrize("outcomes", [["ok", "failed"], ["ok", "ok"]])
async def test_a_dry_run_writes_nothing(
    tmp_path: Path, extractor: type[FakeExtractor], outcomes: list[str]
) -> None:
    stored = await store_document(tmp_path, outcomes)
    before = _snapshot(tmp_path)

    result = await repair_module.repair_extraction(
        spec_for(stored.output, "dry_run"), Settings()
    )

    assert result.selected == outcomes.count("failed")
    assert result.dry_run is bool(result.selected)
    assert _snapshot(tmp_path) == before
    assert extractor.calls == []


def _versionless(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.execute("CREATE TABLE documents (name TEXT PRIMARY KEY)")
            connection.execute("CREATE TABLE chunks (name TEXT, idx INTEGER)")
            connection.execute("INSERT INTO documents VALUES ('doc')")
    finally:
        connection.close()


def test_a_versionless_database_is_refused_as_not_a_store(tmp_path: Path) -> None:
    output = tmp_path / "doc_output.json"
    output.write_text('{"records": []}\n', encoding="utf-8")
    db = tmp_path / "chronominer.sqlite"
    _versionless(db)
    before = db.read_bytes()

    for target in (output, tmp_path):
        with pytest.raises(RepairPlanError, match="not a chronominer store") as caught:
            plan_repair(spec_for(target), Settings())
        assert caught.value.exit_code == 2
    assert db.read_bytes() == before


def test_a_file_no_store_records_is_a_plan_error(tmp_path: Path) -> None:
    output = tmp_path / "doc_output.json"
    output.write_text('{"records": []}\n', encoding="utf-8")

    with pytest.raises(RepairPlanError, match="not recorded"):
        plan_repair(spec_for(output), Settings())


def test_a_folder_without_a_database_is_a_plan_error(tmp_path: Path) -> None:
    with pytest.raises(RepairPlanError, match="No chronominer.sqlite"):
        plan_repair(spec_for(tmp_path), Settings())


def test_a_missing_file_is_a_plan_error(tmp_path: Path) -> None:
    with pytest.raises(RepairPlanError, match="does not exist"):
        plan_repair(spec_for(tmp_path / "missing.txt"), Settings())
