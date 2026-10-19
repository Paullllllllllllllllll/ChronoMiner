"""Batch submissions: commit order, unresolved creates, stale rows, queries.

Run on Windows with:

    uv --directory <repo> run pytest tests/common/test_store_submissions.py -q
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.sqlite import read
from chronominer.common.store import (
    LeaseLostError,
    PlannedUnit,
    ResponseRow,
    Store,
    Submission,
    SubmissionRejectedError,
    SubmissionStateError,
    UnitKey,
)
from tests.common.store_child import FakeClock

MINUTE = 60.0


async def _units(store: Store, name: str = "doc", pages: int = 3) -> list[UnitKey]:
    doc_id = await store.ensure_document(name)
    lease = await store.acquire(doc_id)
    assert lease is not None
    planned = [PlannedUnit(idx) for idx in range(pages)]
    await store.plan_units(lease, "page", planned, model="model-a", provider="openai")
    return [UnitKey(doc_id, "page", idx) for idx in range(pages)]


def _ids(submissions: list[Submission]) -> list[str]:
    return [submission.submission_id for submission in submissions]


def test_the_submission_is_committed_before_the_upload(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    seen: list[Any] = []

    async def upload(submission: Submission) -> str:
        def query(connection: sqlite3.Connection) -> Any:
            row = connection.execute(
                "SELECT status, batch_id, request_count FROM submissions "
                "WHERE submission_id = ?",
                (submission.submission_id,),
            ).fetchone()
            requests = connection.execute(
                "SELECT count(*) FROM submission_requests WHERE submission_id = ?",
                (submission.submission_id,),
            ).fetchone()[0]
            return (*row, requests)

        seen.append(read(db, query))
        return "file-1"

    async def create(submission: Submission) -> str:
        seen.append(submission.upload_ref)
        return "batch-1"

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            units = await _units(store)
            submission = await store.submit(
                provider="openai",
                model="model-a",
                units=units,
                upload=upload,
                create=create,
            )
            assert submission.status == "submitted"
            assert submission.batch_id == "batch-1"
            assert submission.upload_ref == "file-1"
            assert submission.request_count == 3
            requests = store.submission_requests(submission.submission_id)
            assert list(requests.values()) == units
            prefix = submission.submission_id[:8]
            assert all(custom_id.startswith(prefix) for custom_id in requests)
            assert store.blocked_units(units[0].doc_id) == units
            state = store.resume_state("doc")
            assert state is not None
            assert state.ready == ()
            assert _ids(store.open_submissions()) == [submission.submission_id]

    asyncio.run(main())

    assert seen == [("submitting", None, 3, 3), "file-1"]


def test_request_bodies_are_kept_per_custom_id_with_payloads_redacted(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    image = "data:image/jpeg;base64," + "A" * 2000

    async def main() -> dict[str, dict[str, Any]]:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            units = await _units(store, pages=2)
            submission = await store.open_submission(
                provider="openai", model="model-a", units=units
            )
            first, second = store.submission_requests(submission.submission_id)
            body = {"model": "model-a", "input": [{"image_url": image}]}
            await store.record_requests(submission.submission_id, {first: body})
            bodies = store.submission_bodies(submission.submission_id)
            assert second not in bodies
            return {first: bodies[first]}

    (kept,) = asyncio.run(main()).values()

    assert kept["model"] == "model-a"
    assert kept["input"][0]["image_url"].startswith("<omitted data:image/jpeg")


def test_an_ambiguous_create_leaves_the_submission_unresolved(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def upload(submission: Submission) -> None:
        return None

    async def lost(submission: Submission) -> str:
        raise TimeoutError("response lost")

    async def rejected(submission: Submission) -> str:
        raise SubmissionRejectedError("400 invalid request")

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            units = await _units(store, pages=2)
            with pytest.raises(TimeoutError):
                await store.submit(
                    provider="anthropic",
                    model="model-a",
                    units=units[:1],
                    upload=upload,
                    create=lost,
                )
            with pytest.raises(SubmissionRejectedError):
                await store.submit(
                    provider="anthropic",
                    model="model-a",
                    units=units[1:],
                    upload=upload,
                    create=rejected,
                )
            unresolved = store.unresolved_submissions()
            assert len(unresolved) == 1
            assert "response lost" in str(unresolved[0].error)
            assert unresolved[0].upload_ref is None
            failed = store.submissions(("failed",))
            assert "400 invalid request" in str(failed[0].error)
            assert store.blocked_units(units[0].doc_id) == units[:1]
            state = store.resume_state(units[0].doc_id)
            assert state is not None
            assert state.ready == (units[1],)

            adopted = await store.record_batch(
                unresolved[0].submission_id, "msgbatch_1"
            )
            assert adopted.status == "submitted"
            assert store.unresolved_submissions() == []

    asyncio.run(main())


def test_a_failed_upload_marks_the_submission_failed(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def upload(submission: Submission) -> str:
        raise OSError("upload refused")

    async def create(submission: Submission) -> str:
        raise AssertionError("create must not run")

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            units = await _units(store, pages=1)
            with pytest.raises(OSError, match="upload refused"):
                await store.submit(
                    provider="openai",
                    model="model-a",
                    units=units,
                    upload=upload,
                    create=create,
                )
            assert [s.status for s in store.submissions()] == ["failed"]
            assert store.blocked_units(units[0].doc_id) == []

    asyncio.run(main())


def test_a_stale_submission_is_marked_unresolved_and_abandon_frees_its_units(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()

    async def main() -> None:
        async with (
            Store(db, "tool", clock=clock) as crashed,
            Store(db, "tool", clock=clock) as later,
        ):
            await crashed.begin_run("crashed")
            await later.begin_run("later")
            units = await _units(crashed)
            opened = await crashed.open_submission(
                provider="openai", model="model-a", units=units[:2]
            )
            assert _ids(later.stale_submissions()) == [opened.submission_id]
            assert await later.mark_stale_submissions() == []

            clock.advance(11 * MINUTE)

            assert await later.mark_stale_submissions() == [opened.submission_id]
            stale = later.submission(opened.submission_id)
            assert stale.status == "unresolved"
            assert "without a provider batch id" in str(stale.error)
            assert later.stale_submissions() == []
            assert later.blocked_units(units[0].doc_id) == units[:2]
            state = later.resume_state("doc")
            assert state is not None
            assert state.blocked == tuple(units[:2])
            assert state.ready == (units[2],)

            with pytest.raises(LeaseLostError):
                await later.abandon_submission(opened.submission_id)
            assert await later.acquire(units[0].doc_id) is not None
            abandoned = await later.abandon_submission(opened.submission_id)
            assert abandoned.status == "abandoned"
            assert later.blocked_units(units[0].doc_id) == []
            assert later.unresolved_submissions() == []
            assert {unit.status for unit in later.units(units[0].doc_id)} == {"pending"}
            with pytest.raises(SubmissionStateError):
                await later.abandon_submission(opened.submission_id)

    asyncio.run(main())


def test_abandon_frees_the_units_of_a_submitted_submission(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            units = await _units(store, pages=2)
            opened = await store.open_submission(
                provider="openai", model="model-a", units=units
            )
            await store.record_batch(opened.submission_id, "batch-1")
            assert store.blocked_units(units[0].doc_id) == units
            abandoned = await store.abandon_submission(opened.submission_id)
            assert (abandoned.status, abandoned.batch_id) == ("abandoned", "batch-1")
            assert store.blocked_units(units[0].doc_id) == []
            state = store.resume_state("doc")
            assert state is not None
            assert state.ready == tuple(units)
            with pytest.raises(SubmissionStateError):
                await store.mark_finalized(opened.submission_id)

    asyncio.run(main())


def test_the_open_unresolved_and_stale_queries_list_exactly_their_rows(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()

    async def main() -> None:
        async with Store(db, "tool", clock=clock) as store:
            await store.begin_run("run")
            units = await _units(store, pages=4)
            made = []
            for unit in units:
                made.append(
                    await store.open_submission(
                        provider="openai", model="model-a", units=[unit]
                    )
                )
                clock.advance(1)
            submitted, unresolved, finalized, submitting = _ids(made)
            await store.record_upload(submitted, "file-a")
            await store.record_batch(submitted, "batch-a", provider_status="validating")
            await store.mark_unresolved(unresolved, "lost")
            await store.record_batch(finalized, "batch-c")
            checked = await store.record_check(
                finalized, provider_status="completed", counts={"completed": 1}
            )
            assert checked.last_checked_at is not None
            assert checked.counts == {"completed": 1}
            done = await store.mark_finalized(finalized, provider_status="completed")
            assert done.finalized_at is not None

            assert _ids(store.open_submissions()) == [submitted, submitting]
            assert _ids(store.unresolved_submissions()) == [unresolved]
            assert _ids(store.stale_submissions()) == [submitting]
            assert _ids(store.submissions(("finalized",))) == [finalized]
            assert store.submission(submitted).provider_status == "validating"
            assert store.blocked_units(units[0].doc_id) == [
                units[0],
                units[1],
                units[3],
            ]

            with pytest.raises(SubmissionStateError):
                await store.mark_finalized(submitting)
            with pytest.raises(SubmissionStateError):
                await store.record_upload(submitted, "file-b")
            cancelled = await store.mark_cancelled(submitted)
            assert cancelled.status == "cancelled"
            with pytest.raises(KeyError):
                store.submission("missing")

    asyncio.run(main())


def test_submission_writes_check_the_lease(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()

    async def main() -> None:
        async with Store(db, "tool", clock=clock) as store:
            await store.begin_run("run")
            units = await _units(store, pages=1)
            opened = await store.open_submission(
                provider="openai", model="model-a", units=units
            )
            clock.advance(11 * MINUTE)
            with pytest.raises(LeaseLostError):
                await store.record_batch(opened.submission_id, "batch-1")
            assert store.submission(opened.submission_id).status == "submitting"
            with pytest.raises(LeaseLostError):
                await store.open_submission(
                    provider="openai", model="model-a", units=units
                )

    asyncio.run(main())


def test_custom_ids_and_batch_ids_are_scoped(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> None:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            units = await _units(store, pages=2)
            with pytest.raises(ValueError, match="unique"):
                await store.open_submission(
                    provider="openai",
                    model="model-a",
                    units=units,
                    custom_ids=["same", "same"],
                )
            first = await store.open_submission(
                provider="openai", model="model-a", units=units, custom_ids=["a", "b"]
            )
            second = await store.open_submission(
                provider="openai", model="model-a", units=units, custom_ids=["a", "b"]
            )
            assert store.submission_requests(second.submission_id) == dict(
                zip(["a", "b"], units, strict=True)
            )
            await store.record_batch(first.submission_id, "batch-1")
            with pytest.raises(sqlite3.IntegrityError):
                await store.record_batch(second.submission_id, "batch-1")

            line = '{"custom_id": "a", "response": {"status_code": 200}}'
            await store.append_response(
                ResponseRow(
                    provider="openai",
                    model="model-a",
                    unit=units[0],
                    submission_id=first.submission_id,
                    custom_id="a",
                    raw=line,
                    raw_kind="line",
                )
            )
            rows = store.responses(submission_id=first.submission_id)
            assert [
                (row["custom_id"], row["raw"], row["raw_kind"]) for row in rows
            ] == [("a", line, "line")]

    asyncio.run(main())
