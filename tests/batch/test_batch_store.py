"""Batch submission on the store: commit order, refusals, unknown creates.

Run on Windows with:

    uv --directory <repo> run pytest tests/batch/test_batch_store.py -q

The fake provider (``world``) reads the database through ``mode=ro``
connections when its upload and create run, so the oracles see what the
store had committed before each provider call.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from chronominer.batch.submit import (
    BatchSubmissionError,
    SubmitOutcome,
    submit_batch,
)
from chronominer.common.sqlite import read
from chronominer.extract.units import DocumentHandle
from chronominer.output.store import database_path, open_store
from chronominer.settings import Settings
from tests.batch.conftest import (
    World,
    leased_document,
    page_images,
    run_spec,
)


class ApiError(Exception):
    """A provider error carrying an HTTP status."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"Error code: {status_code} - refused")
        self.status_code = status_code


def _submission_rows(db: Path) -> list[dict[str, Any]]:
    def query(connection: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = connection.execute(
            "SELECT submission_id, status, upload_ref, batch_id, part, "
            "request_count FROM submissions ORDER BY part"
        ).fetchall()
        names = ("id", "status", "upload_ref", "batch_id", "part", "count")
        return [dict(zip(names, row, strict=True)) for row in rows]

    return read(db, query)


def _requests(db: Path, submission_id: str) -> list[tuple[str, int]]:
    return read(
        db,
        lambda connection: connection.execute(
            "SELECT custom_id, idx FROM submission_requests "
            "WHERE submission_id = ? ORDER BY idx",
            (submission_id,),
        ).fetchall(),
    )


async def _submit(
    root: Path, *, pages: int = 3, unrendered: frozenset[int] = frozenset()
) -> tuple[SubmitOutcome, DocumentHandle]:
    async with open_store(root) as store:
        await store.begin_run("extract")
        handle = await leased_document(store)
        outcome = await submit_batch(
            page_images(pages, unrendered=unrendered),
            document=handle,
            spec=run_spec(root),
            settings=Settings(),
        )
    return outcome, handle


async def _todo(root: Path) -> list[int]:
    async with open_store(root) as store:
        await store.begin_run("resume")
        handle = await leased_document(store)
        return handle.todo()


async def test_each_step_is_committed_before_the_next_provider_call(
    world: World, tmp_path: Path
) -> None:
    db = database_path(tmp_path)
    world.file_uploads = True
    seen: dict[str, Any] = {}

    def at_upload(submission_id: str) -> None:
        seen["upload"] = (
            _submission_rows(db),
            _requests(db, submission_id),
        )

    def at_create(submission_id: str) -> None:
        seen["create"] = _submission_rows(db)

    world.hooks = {"upload": at_upload, "create": at_create}

    outcome, _handle = await _submit(tmp_path)

    assert outcome.status == "submitted"
    (submission_id,) = outcome.submission_ids
    rows, requests = seen["upload"]
    assert rows == [
        {
            "id": submission_id,
            "status": "submitting",
            "upload_ref": None,
            "batch_id": None,
            "part": 1,
            "count": 3,
        }
    ]
    assert [idx for _custom_id, idx in requests] == [0, 1, 2]
    assert seen["create"][0]["upload_ref"] == "openai-file-1"
    assert seen["create"][0]["batch_id"] is None
    (final,) = _submission_rows(db)
    assert final["status"] == "submitted"
    assert final["batch_id"] == "openai-batch-1"
    assert world.deleted == []


async def test_custom_ids_are_scoped_to_their_submission(
    world: World, tmp_path: Path
) -> None:
    world.max_requests = 2
    outcome, _handle = await _submit(tmp_path)

    db = database_path(tmp_path)
    first, second = outcome.submission_ids
    first_ids = [custom_id for custom_id, _idx in _requests(db, first)]
    second_ids = [custom_id for custom_id, _idx in _requests(db, second)]
    assert [idx for _cid, idx in _requests(db, first)] == [0, 1]
    assert [idx for _cid, idx in _requests(db, second)] == [2]
    assert all(cid.startswith(first[:8] + "-") for cid in first_ids)
    assert all(cid.startswith(second[:8] + "-") for cid in second_ids)
    sent = [list(batch.bodies) for batch in world.batches.values()]
    assert sent == [first_ids, second_ids]
    assert [row["part"] for row in _submission_rows(db)] == [1, 2]


async def test_a_definite_refusal_fails_the_submission_and_deletes_its_upload(
    world: World, tmp_path: Path
) -> None:
    world.file_uploads = True
    world.errors[("openai", "submit")] = [ApiError(400)]

    outcome, _handle = await _submit(tmp_path)

    assert outcome.status == "refused"
    assert "refused" in (outcome.detail or "")
    (row,) = _submission_rows(database_path(tmp_path))
    assert row["status"] == "failed"
    assert world.deleted == ["openai-file-1"]
    assert world.batches == {}
    assert await _todo(tmp_path) == [0, 1, 2]


async def test_an_unknown_create_outcome_keeps_the_upload_and_blocks_the_units(
    world: World, tmp_path: Path
) -> None:
    world.file_uploads = True
    world.lose_create = True

    outcome, _handle = await _submit(tmp_path)

    assert outcome.status == "unresolved"
    assert "batch check" in (outcome.detail or "")
    (row,) = _submission_rows(database_path(tmp_path))
    assert row["status"] == "unresolved"
    assert row["upload_ref"] == "openai-file-1"
    assert row["batch_id"] is None
    assert world.deleted == []
    assert len(world.batches) == 1
    assert await _todo(tmp_path) == []


async def test_a_failed_upload_is_a_refusal(world: World, tmp_path: Path) -> None:
    world.errors[("openai", "upload")] = [ApiError(500)]

    outcome, _handle = await _submit(tmp_path)

    assert outcome.status == "refused"
    (row,) = _submission_rows(database_path(tmp_path))
    assert row["status"] == "failed"
    assert world.batches == {}


async def test_an_unresolved_part_leaves_the_later_parts_unsubmitted(
    world: World, tmp_path: Path
) -> None:
    world.max_requests = 1
    world.errors[("openai", "submit")] = [ApiError(503)]

    outcome, _handle = await _submit(tmp_path)

    assert outcome.status == "unresolved"
    assert len(outcome.submission_ids) == 1
    assert [row["status"] for row in _submission_rows(database_path(tmp_path))] == [
        "unresolved"
    ]
    assert await _todo(tmp_path) == [1, 2]


async def test_chunks_are_planned_with_their_provenance(
    world: World, tmp_path: Path
) -> None:
    outcome, handle = await _submit(tmp_path, unrendered=frozenset({1}))

    assert outcome.status == "submitted"
    rows = read(
        database_path(tmp_path),
        lambda connection: connection.execute(
            "SELECT idx, status, label, source_index, provenance FROM units "
            "ORDER BY idx"
        ).fetchall(),
    )
    assert [row[:4] for row in rows] == [
        (0, "pending", "page_0001", 0),
        (1, "pending", "page_0002", 1),
        (2, "pending", "page_0003", 2),
    ]
    assert json.loads(rows[0][4]) == {
        "custom_id": "doc-page-1",
        "chunk_range": None,
        "image_provenance": {"dpi": 300},
    }
    (submission_id,) = outcome.submission_ids
    assert [idx for _cid, idx in _requests(database_path(tmp_path), submission_id)] == [
        0,
        1,
        2,
    ]
    (batch,) = world.batches.values()
    assert len(batch.bodies) == 2


async def test_unbuildable_requests_still_raise(world: World, tmp_path: Path) -> None:
    with pytest.raises(BatchSubmissionError, match="No chunk"):
        await _submit(tmp_path, pages=1, unrendered=frozenset({0}))
    assert world.calls == []
