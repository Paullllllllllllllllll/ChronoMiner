"""Finalization: errors, outcomes, entries, provider files and idempotence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from chronominer.batch.check import check_submissions
from chronominer.batch.finalize import (
    FAILED,
    FINALIZED_OUTCOME,
    MISSING_TEXT,
    PENDING,
    placeholder,
)
from chronominer.batch.submit import submit_batch
from chronominer.common.batch import BatchResultItem, BatchStatus
from chronominer.common.capture import redact_payloads
from chronominer.common.responses import ResponseOutcome
from chronominer.common.sqlite import read
from chronominer.common.usage import Usage
from chronominer.output.store import database_path, open_store
from chronominer.settings import Settings
from tests.batch.conftest import (
    World,
    answer,
    leased_document,
    page_images,
    run_spec,
    text_chunks,
)

OUTPUT = "doc_output.json"


def item(custom_id: str, **values: Any) -> BatchResultItem:
    return BatchResultItem(custom_id=custom_id, **values)


@pytest.mark.parametrize(
    ("provider", "result", "text"),
    [
        (
            "openai",
            item(
                "a",
                error="bad image",
                error_code="invalid_request",
                origin="error_file",
                response={"status_code": 400},
            ),
            "[extraction error; code invalid_request; bad image]",
        ),
        (
            "openai",
            item(
                "a",
                error="bad image",
                error_code="invalid_request",
                response={"status_code": 400},
            ),
            "[extraction error; status 400; code invalid_request; bad image]",
        ),
        (
            "openai",
            item("a", error="HTTP 500", response={"status_code": 500}),
            "[extraction error; status 500]",
        ),
        (
            "anthropic",
            item("a", error="overloaded"),
            "[extraction error: overloaded]",
        ),
        ("google", item("a", error="no result", origin="missing"), MISSING_TEXT),
    ],
)
def test_error_texts_follow_each_provider(
    provider: str, result: BatchResultItem, text: str
) -> None:
    assert placeholder(provider, result) == text


async def submit_store_pages(
    root: Path,
    pages: int = 3,
    *,
    chunks: list[Any] | None = None,
    run: dict[str, Any] | None = None,
) -> str:
    """Submit *pages* pages of document ``doc`` in *root*; return the submission.

    *run* is the effective spec the submitting run records.
    """
    async with open_store(root) as store:
        await store.begin_run("extract", spec=run)
        handle = await leased_document(store)
        outcome = await submit_batch(
            chunks if chunks is not None else page_images(pages),
            document=handle,
            spec=run_spec(root),
            settings=Settings(),
        )
    assert outcome.status == "submitted"
    (submission_id,) = outcome.submission_ids
    return submission_id


def submission_row(root: Path, submission_id: str) -> dict[str, Any]:
    def query(connection: sqlite3.Connection) -> dict[str, Any]:
        cursor = connection.execute(
            "SELECT * FROM submissions WHERE submission_id = ?", (submission_id,)
        )
        names = [column[0] for column in cursor.description]
        return dict(zip(names, cursor.fetchone(), strict=True))

    return read(database_path(root), query)


def unit_rows(root: Path) -> list[tuple[str, str | None, str | None]]:
    return read(
        database_path(root),
        lambda connection: connection.execute(
            "SELECT status, content, error FROM units ORDER BY idx"
        ).fetchall(),
    )


def entries(root: Path) -> list[tuple[int, str]]:
    return read(
        database_path(root),
        lambda connection: connection.execute(
            "SELECT chunk_index, person FROM entries ORDER BY chunk_index"
        ).fetchall(),
    )


async def test_a_completed_batch_is_finalized_once(
    world: World, tmp_path: Path
) -> None:
    await submit_store_pages(tmp_path, pages=2)
    (batch,) = world.batches.values()
    first_id, second_id = batch.bodies
    world.dropped = {second_id}

    first = await check_submissions(tmp_path, Settings())

    (check,) = first.submissions
    assert (check.outcome, check.errors) == (FINALIZED_OUTCOME, 1)
    assert check.outputs == (tmp_path / OUTPUT,)
    assert unit_rows(tmp_path) == [
        ("done", answer(f"text of {first_id}"), None),
        ("failed", None, MISSING_TEXT),
    ]
    assert entries(tmp_path) == [(1, f"text of {first_id}")]
    document = json.loads((tmp_path / OUTPUT).read_text(encoding="utf-8"))
    (record,) = document["records"]
    assert record["custom_id"] == "doc-page-1"
    assert record["response"]["output_text"] == answer(f"text of {first_id}")
    usage = record["response"]["response_data"]["usage"]
    assert list(usage) == ["input_tokens", "output_tokens"]
    assert usage["input_tokens"] == 100
    assert document["_chronominer_metadata"]["failed_chunks"] == [2]
    assert document["_chronominer_metadata"]["schema_name"] == "WageRecords"

    world.calls.clear()
    again = await check_submissions(tmp_path, Settings())
    assert again.submissions == []
    assert world.calls == []


async def test_text_chunks_are_finalized_with_their_ranges(
    world: World, tmp_path: Path
) -> None:
    await submit_store_pages(tmp_path, chunks=list(text_chunks(2)))

    result = await check_submissions(tmp_path, Settings(), formats=("csv",))

    (check,) = result.submissions
    assert check.outcome == FINALIZED_OUTCOME
    assert check.outputs == (tmp_path / OUTPUT, tmp_path / "doc_output.csv")
    document = json.loads((tmp_path / OUTPUT).read_text(encoding="utf-8"))
    assert [record["chunk_range"] for record in document["records"]] == [
        [1, 3],
        [4, 6],
    ]
    assert [index for index, _person in entries(tmp_path)] == [1, 2]


@pytest.mark.parametrize(
    ("formats", "names"),
    [(None, [OUTPUT, "doc_output.csv"]), ((), [OUTPUT])],
    ids=["run formats", "given formats"],
)
async def test_the_check_renders_the_formats_of_the_submitting_run(
    world: World, tmp_path: Path, formats: tuple[str, ...] | None, names: list[str]
) -> None:
    run = {"output.formats": {"value": ["csv"], "source": "flag"}}
    await submit_store_pages(tmp_path, chunks=list(text_chunks(2)), run=run)

    result = await check_submissions(tmp_path, Settings(), formats=formats)

    (check,) = result.submissions
    assert [path.name for path in check.outputs] == names
    assert check.complete == 1


async def test_an_answer_without_text_fails_its_chunk(
    world: World, tmp_path: Path
) -> None:
    await submit_store_pages(tmp_path, pages=2)
    (batch,) = world.batches.values()
    first_id, second_id = batch.bodies
    world.results[second_id] = BatchResultItem(
        custom_id=second_id, success=True, text="  "
    )

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.errors == 1
    statuses = [status for status, _content, _error in unit_rows(tmp_path)]
    assert statuses == ["done", "failed"]


async def test_a_truncated_answer_fails_its_chunk(world: World, tmp_path: Path) -> None:
    await submit_store_pages(tmp_path, pages=2)
    (batch,) = world.batches.values()
    first_id, second_id = batch.bodies
    cut = ResponseOutcome("truncated", "stop_reason=max_tokens")
    result_item = item(second_id, text=answer("text of cut"))
    result_item.settle(cut, "empty")
    world.results[second_id] = result_item

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.errors == 1
    assert unit_rows(tmp_path)[1] == ("failed", None, cut.message)
    assert entries(tmp_path) == [(1, f"text of {first_id}")]


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        (BatchStatus.IN_PROGRESS, PENDING),
        (BatchStatus.EXPIRED, FAILED),
        (BatchStatus.CANCELLED, FAILED),
        (BatchStatus.UNKNOWN, PENDING),
    ],
)
async def test_an_unfinished_batch_writes_no_unit(
    world: World, tmp_path: Path, status: BatchStatus, outcome: str
) -> None:
    world.status = status
    await submit_store_pages(tmp_path, pages=2)

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.outcome == outcome
    assert [row[0] for row in unit_rows(tmp_path)] == ["pending"] * 2
    assert not (tmp_path / OUTPUT).exists()
    assert world.cleaned == []


async def test_a_missing_key_fails_the_submission(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submission_id = await submit_store_pages(tmp_path, pages=1)
    monkeypatch.delenv("OPENAI_API_KEY")

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.outcome == FAILED
    assert "OPENAI_API_KEY" in (check.detail or "")
    assert submission_row(tmp_path, submission_id)["status"] == "submitted"


async def test_each_response_row_holds_the_body_its_line_sent(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    openai = sdk["openai"]
    submission_id = await submit_store_pages(tmp_path, pages=2)
    batch = openai._retrieve(submission_row(tmp_path, submission_id)["batch_id"])
    lines = openai.data[batch.input_file_id].decode("utf-8").splitlines()
    sent = {line["custom_id"]: line["body"] for line in map(json.loads, lines)}
    openai.complete(batch.id)

    await check_submissions(tmp_path, Settings())

    stored = read(
        database_path(tmp_path),
        lambda connection: connection.execute(
            "SELECT custom_id, request, usage FROM responses WHERE submission_id = ?",
            (submission_id,),
        ).fetchall(),
    )
    assert {custom_id for custom_id, _request, _usage in stored} == set(sent)
    for custom_id, request, usage in stored:
        assert json.loads(request) == redact_payloads(sent[custom_id])
        assert list(json.loads(usage)) == list(Usage().as_dict())


async def test_each_result_record_is_stored_as_downloaded(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    openai = sdk["openai"]
    submission_id = await submit_store_pages(tmp_path)
    row = submission_row(tmp_path, submission_id)
    output, errors = openai.complete(row["batch_id"], failed=frozenset({1}))
    order: list[tuple[str, str, bool]] = []

    def at_delete(file_id: str) -> None:
        status = submission_row(tmp_path, submission_id)["status"]
        order.append((file_id, status, (tmp_path / OUTPUT).is_file()))

    openai.on_delete = at_delete

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert (check.outcome, check.errors) == (FINALIZED_OUTCOME, 1)
    assert check.outputs == (tmp_path / OUTPUT,)
    raw = read(
        database_path(tmp_path),
        lambda connection: connection.execute(
            "SELECT custom_id, raw, raw_kind, idx FROM responses "
            "WHERE submission_id = ? ORDER BY response_id",
            (submission_id,),
        ).fetchall(),
    )
    lines = [
        *output.decode("utf-8").splitlines(),
        *errors.decode("utf-8").splitlines(),
    ]
    assert [stored for _cid, stored, _kind, _idx in raw] == lines
    for _cid, stored, kind, _idx in raw:
        assert kind == "line"
        assert stored.encode("utf-8") in output + errors
    assert [idx for _cid, _raw, _kind, idx in raw] == [0, 2, 1]
    units = read(
        database_path(tmp_path),
        lambda connection: connection.execute(
            "SELECT idx, status, content, error, response_id IS NOT NULL "
            "FROM units ORDER BY idx"
        ).fetchall(),
    )
    assert units == [
        (0, "done", answer("Zürich «0»"), None, 1),
        (
            1,
            "failed",
            None,
            "[extraction error; code invalid_image; Bild unlesbar]",
            1,
        ),
        (2, "done", answer("Zürich «2»"), None, 1),
    ]
    assert [person for _index, person in entries(tmp_path)] == [
        "Zürich «0»",
        "Zürich «2»",
    ]
    assert submission_row(tmp_path, submission_id)["status"] == "finalized"
    assert sorted(file_id for file_id, _status, _file in order) == [
        "file-1",
        "file-2",
        "file-3",
    ]
    assert all(status == "finalized" and rendered for _id, status, rendered in order)


async def test_a_running_batch_records_its_status_and_stays_pending(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    submission_id = await submit_store_pages(tmp_path, pages=1)

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.outcome == PENDING
    row = submission_row(tmp_path, submission_id)
    assert (row["status"], row["provider_status"]) == ("submitted", "in_progress")
    assert row["last_checked_at"] is not None
    assert not (tmp_path / OUTPUT).exists()
    assert sdk["openai"].deleted == []


async def test_a_batch_ended_without_results_frees_its_units_and_keeps_files(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    submission_id = await submit_store_pages(tmp_path, pages=2)
    batch = sdk["openai"]._retrieve(submission_row(tmp_path, submission_id)["batch_id"])
    batch.status = "expired"

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.outcome == FAILED
    assert submission_row(tmp_path, submission_id)["status"] == "failed"
    assert sdk["openai"].deleted == []
    async with open_store(tmp_path) as store:
        await store.begin_run("resume")
        handle = await leased_document(store)
        assert handle.todo() == [0, 1]


async def test_a_successful_finalization_deletes_the_provider_files(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    submission_id = await submit_store_pages(tmp_path, pages=1)
    openai = sdk["openai"]
    openai.complete(submission_row(tmp_path, submission_id)["batch_id"])

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.outcome == FINALIZED_OUTCOME
    assert sorted(openai.deleted) == ["file-1", "file-2"]
